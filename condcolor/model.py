"""Conditioned UNet (issue #17): greyscale L -> ab, guided by frozen CLIP caption features.

    L (1xHxW)
      encoder   64 @H, 128 @H/2, 256 @H/4, 512 @H/8         (skip copies kept)
      bottleneck 512 @H/16   + cross-attention to the caption
      decoder   512 @H/8 (+xattn), 256 @H/4 (+xattn), 128 @H/2, 64 @H
      head      1x1 conv -> 2 channels -> tanh  = ab / 110

Each level uses residual blocks (GroupNorm, SiLU, 3x3 conv). Cross-attention: image
positions are queries; the 77 caption tokens (512-d, frozen CLIP) are keys and values;
padding tokens are masked out. Only this network trains; CLIP stays frozen.

Self-test (shapes, parameter count, one fp32 training step, peak VRAM, gradient check):
  python -m condcolor.model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_DIM = 512


def norm(c):
    g = 32
    while c % g:  # largest group count <= 32 that divides the channel count
        g -= 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Image features attend to caption tokens; residual, so with no useful text the
    block can learn to pass the image through unchanged."""

    def __init__(self, c, heads=8):
        super().__init__()
        self.norm = norm(c)
        self.attn = nn.MultiheadAttention(c, heads, kdim=TEXT_DIM, vdim=TEXT_DIM, batch_first=True)
        self.last_weights = None  # (B, HW, 77), kept for the cross-attention maps (#19)

    def forward(self, x, text, text_mask):
        b, c, h, w = x.shape
        q = self.norm(x).flatten(2).transpose(1, 2)               # (B, HW, C)
        out, weights = self.attn(q, text, text, key_padding_mask=~text_mask,
                                 need_weights=True, average_attn_weights=True)
        self.last_weights = weights.detach()
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class Down(nn.Module):
    def __init__(self, cin, cout, blocks):
        super().__init__()
        self.blocks = nn.Sequential(ResBlock(cin, cout), *[ResBlock(cout, cout) for _ in range(blocks - 1)])
        self.pool = nn.Conv2d(cout, cout, 3, stride=2, padding=1)

    def forward(self, x):
        h = self.blocks(x)
        return self.pool(h), h


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, blocks, xattn):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.blocks = nn.ModuleList([ResBlock(cin + cskip, cout)] +
                                    [ResBlock(cout, cout) for _ in range(blocks - 1)])
        self.xattn = CrossAttention(cout) if xattn else None

    def forward(self, x, skip, text, text_mask):
        x = torch.cat([self.up(x), skip], 1)
        for b in self.blocks:
            x = b(x)
        if self.xattn is not None:
            x = self.xattn(x, text, text_mask)
        return x


class CondUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 384), blocks=(1, 1, 2, 2), bottleneck=640,
                 xattn_levels=(3, 2), use_text=True):
        """blocks: residual blocks per level (fewer at high resolution, where activations
        are largest). xattn_levels: decoder levels (index into widths) that get
        cross-attention; the bottleneck always has it when use_text. use_text=False builds
        the same UNet with no attention (text-free control; efficiency study #21)."""
        super().__init__()
        self.use_text = use_text
        if isinstance(blocks, int):
            blocks = (blocks,) * len(widths)
        self.stem = nn.Conv2d(1, widths[0], 3, padding=1)
        self.downs = nn.ModuleList()
        c = widths[0]
        for w, nb in zip(widths, blocks):
            self.downs.append(Down(c, w, nb))
            c = w
        self.mid1 = ResBlock(c, bottleneck)
        self.mid_xattn = CrossAttention(bottleneck) if use_text else None
        self.mid2 = ResBlock(bottleneck, bottleneck)
        self.ups = nn.ModuleList()
        c = bottleneck
        for lvl in reversed(range(len(widths))):
            self.ups.append(Up(c, widths[lvl], widths[lvl], blocks[lvl], use_text and lvl in xattn_levels))
            c = widths[lvl]
        self.head = nn.Sequential(norm(c), nn.SiLU(), nn.Conv2d(c, 2, 1))

    def forward(self, L, text=None, text_mask=None):
        x = self.stem(L)
        skips = []
        for d in self.downs:
            x, s = d(x)
            skips.append(s)
        x = self.mid1(x)
        if self.mid_xattn is not None:
            x = self.mid_xattn(x, text, text_mask)
        x = self.mid2(x)
        for u in self.ups:
            x = u(x, skips.pop(), text, text_mask)
        return torch.tanh(self.head(x))

    def xattn_modules(self):
        return [m for m in self.modules() if isinstance(m, CrossAttention)]


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def limit_vram(gb=5.6):
    """On Windows, going past the card's memory silently spills into system RAM and runs
    ~10x slower. Capping the allocator makes that an immediate out-of-memory error instead."""
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total))


def _selftest():
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        limit_vram()
    torch.manual_seed(0)
    net = CondUNet().to(dev)
    attn = sum(count(m) for m in net.xattn_modules())
    print(f"device {dev}; parameters {count(net) / 1e6:.1f}M (cross-attention {attn / 1e6:.1f}M); "
          f"text-free variant {count(CondUNet(use_text=False)) / 1e6:.1f}M")

    b, s = 8, 256
    L = torch.randn(b, 1, s, s, device=dev)
    ab = torch.rand(b, 2, s, s, device=dev) * 2 - 1
    text = torch.randn(b, 77, TEXT_DIM, device=dev)
    mask = torch.zeros(b, 77, dtype=torch.bool, device=dev)
    mask[:, :12] = True
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(4):
        t = time.time()
        out = net(L, text, mask)
        loss = F.smooth_l1_loss(out, ab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"  step {step}: out {tuple(out.shape)} loss {loss.item():.4f} {time.time() - t:.2f}s")
    if dev == "cuda":
        print(f"peak VRAM at batch {b}, {s}px, fp32: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    g = [m.attn.in_proj_weight.grad if m.attn.in_proj_weight is not None else m.attn.k_proj_weight.grad
         for m in net.xattn_modules()]
    print("cross-attention grad norms:", ", ".join(f"{x.norm().item():.2e}" for x in g))


if __name__ == "__main__":
    _selftest()
