"""Overfit sanity check (issue #15, check 2): can the conditioned UNet memorise ten pairs?

A correctly wired model drives training error on ten fixed images to ~0. If it can't, there
is a bug (data, Lab scaling, model, loss, optimiser) and full training would be wasted.

Logged every --log steps:
  loss as mean |ab error| in Lab units (x110), vs the "predict grey" reference
  gradient norm of the cross-attention key/value projections (the caption pathway)
At the end:
  figures/overfit_check.png        grey input / prediction / truth for the ten images
  figures/overfit_loss.png         loss curve
  outputs/overfit/log.csv          step, loss, grad norms
  caption-swap test                same images scored with real / no / shuffled captions

Usage
  python -m condcolor.overfit                  # 10 images, up to 1000 steps
  python -m condcolor.overfit --n 10 --steps 1500 --lr 5e-4
"""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from condcolor import data, text
from condcolor.data import AB_SCALE
from condcolor.model import CondUNet, count, limit_vram

OUT = Path("outputs/overfit")
FIG = Path("figures")


def batch_of(ds):
    items = [ds[i] for i in range(len(ds))]
    return {k: torch.stack([it[k] for it in items]) for k in ("L", "ab", "text", "text_mask")}, \
           [it["id"] for it in items]


def to(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


def caption_grads(net):
    """Norm of the gradient on each cross-attention layer's text-side (key, value) weights."""
    out = []
    for m in net.xattn_modules():
        a = m.attn
        out.append(float(torch.cat([a.k_proj_weight.grad.flatten(), a.v_proj_weight.grad.flatten()]).norm()))
    return out


@torch.no_grad()
def score(net, b):
    pred = net(b["L"], b["text"], b["text_mask"])
    return float((pred - b["ab"]).abs().mean()) * AB_SCALE, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target", type=float, default=1.0, help="stop when mean ab error (Lab units) is below this")
    ap.add_argument("--log", type=int, default=25)
    ap.add_argument("--seed", type=int, default=693)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        limit_vram()
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    ids = data.split_ids("train")
    ids = random.Random(args.seed).sample(ids, args.n)
    table = text.load()
    real, ids = batch_of(data.ColourPairs(ids, table, train=False, caption_mode="real"))
    real = to(real, dev)
    grey_ref = float(real["ab"].abs().mean()) * AB_SCALE

    net = CondUNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"device {dev}; {count(net) / 1e6:.1f}M params; {args.n} images; lr {args.lr}")
    print(f"reference: predicting grey (ab=0) gives mean ab error {grey_ref:.2f} Lab units; target < {args.target}")

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        pred = net(real["L"], real["text"], real["text_mask"])
        loss = F.l1_loss(pred, real["ab"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        err = loss.item() * AB_SCALE
        if step == 1 or step % args.log == 0 or err < args.target:
            g = caption_grads(net)
            rows.append({"step": step, "ab_err_lab": round(err, 4), **{f"grad_xattn{i}": f"{x:.2e}" for i, x in enumerate(g)}})
            print(f"step {step:>5}  ab error {err:7.3f}  caption-path grads " + " ".join(f"{x:.1e}" for x in g)
                  + f"  ({(time.time() - t0) / step:.2f} s/step)", flush=True)
        opt.step()
        if err < args.target:
            print(f"reached target at step {step}")
            break

    with open(OUT / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if dev == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; "
              f"{(time.time() - t0) / step:.2f} s/step; total {(time.time() - t0) / 60:.1f} min")

    # ---- caption-swap test: does the memorised model depend on the caption?
    net.eval()
    results = {}
    for mode in ("real", "none", "shuffled"):
        b, _ = batch_of(data.ColourPairs(ids, table, train=False, caption_mode=mode))
        results[mode], p = score(net, to(b, dev))
        if mode == "real":
            pred_real = p
    print("caption-swap test (mean ab error, Lab units, same 10 images):")
    for mode, e in results.items():
        print(f"  {mode:<9} {e:7.3f}" + ("" if mode == "real" else f"   (+{e - results['real']:.3f} vs real)"))

    # ---- figures
    fig, ax = plt.subplots(3, args.n, figsize=(2 * args.n, 6.4))
    for j in range(args.n):
        L, ab = real["L"][j].cpu(), real["ab"][j].cpu()
        for i, img in enumerate([data.tensors_to_rgb(L, torch.zeros_like(ab)),
                                 data.tensors_to_rgb(L, pred_real[j].cpu()),
                                 data.tensors_to_rgb(L, ab)]):
            ax[i, j].imshow(img)
            ax[i, j].axis("off")
    for i, name in enumerate(["input (L)", "prediction", "truth"]):
        ax[i, 0].set_title(name, loc="left", fontsize=10)
    fig.suptitle(f"Overfit check: {args.n} training images, step {step}, ab error {results['real']:.2f} Lab units")
    fig.tight_layout()
    fig.savefig(FIG / "overfit_check.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["step"] for r in rows], [r["ab_err_lab"] for r in rows], label="training ab error")
    ax.axhline(grey_ref, ls="--", c="gray", label="predict grey")
    ax.axhline(args.target, ls=":", c="green", label="target")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("mean |ab error| (Lab units)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "overfit_loss.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG / 'overfit_check.png'}, {FIG / 'overfit_loss.png'}, {OUT / 'log.csv'}")


if __name__ == "__main__":
    main()
