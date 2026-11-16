"""Train the caption-conditioned UNet (issue #17) on the training split.

  data        train split (456 colour originals), random 256px crop + horizontal flip
  condition   frozen CLIP caption features (cached); with probability --caption-dropout the
              caption is replaced by the empty one, so the model also learns to colour without
              text and the "no caption" condition in #18 is in-distribution
  loss        L1 on ab
  optimiser   AdamW, linear warm-up then cosine decay to ~0 (avoids late-training spikes)
  validation  every --val-every epochs on the val split, full images, through the shared
              protocol (condcolor.protocol); best checkpoint by val dE_ab
  output      checkpoints/<run>/{best,last}.pt, outputs/train/<run>/log.csv, figures/<run>_curve.png

The test split is never touched here.

Usage
  python -m condcolor.train                              # default run "condunet"
  python -m condcolor.train --epochs 80 --run condunet_e80
  python -m condcolor.train --no-text --run unet_notext  # same UNet without cross-attention
  python -m condcolor.train --resume                     # continue from last.pt
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from condcolor import data, text
from condcolor.model import CondUNet, count, limit_vram
from condcolor.predict import predict_rgb
from condcolor.protocol import load_truth, score_image, split_ids, summarise


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in ("L", "ab", "text", "text_mask")}


@torch.no_grad()
def validate(net, table, ids, dev):
    net.eval()
    rows = []
    for iid in ids:
        feat, mask = table[iid]
        truth = load_truth(iid)
        rows.append(score_image(truth, predict_rgb(net, truth, feat, mask, dev=dev)))
    net.train()
    return summarise(rows)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="condunet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300, help="warm-up steps")
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--no-text", action="store_true", help="UNet without cross-attention (control / #21)")
    ap.add_argument("--padding", default="reflect", choices=["reflect", "zeros"],
                    help="conv padding; 'zeros' reproduces the first two runs (corner-patch artifact)")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=693)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        limit_vram()
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path("checkpoints") / args.run
    log_dir = Path("outputs/train") / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    table = text.load()
    train_ids, val_ids = split_ids("train"), split_ids("val")
    ds = data.ColourPairs(train_ids, table, train=True, caption_mode="real", seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate,
                    num_workers=args.workers, persistent_workers=args.workers > 0)
    empty_feat, empty_mask = table[None]

    model_args = {"use_text": not args.no_text, "padding": args.padding}
    net = CondUNet(**model_args).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(dl)
    start_epoch, best, rows = 1, float("inf"), []
    if args.resume and (ckpt_dir / "last.pt").exists():
        c = torch.load(ckpt_dir / "last.pt", map_location=dev)
        net.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start_epoch, best, rows = c["epoch"] + 1, c["best"], c["log"]
        torch.set_rng_state(c["rng"])
        print(f"resumed {args.run} at epoch {start_epoch}")
    print(f"{args.run}: {count(net) / 1e6:.1f}M params, text={'off' if args.no_text else 'on'}, "
          f"{len(train_ids)} train / {len(val_ids)} val, {len(dl)} steps/epoch x {args.epochs} epochs, device {dev}")

    def save(path, epoch):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best,
                    "log": rows, "rng": torch.get_rng_state(), "model_args": model_args,
                    "run": args.run, "args": vars(args)}, path)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        losses = []
        for i, b in enumerate(dl):
            step = (epoch - 1) * len(dl) + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.warmup, args.lr)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            if args.caption_dropout > 0:  # replace some captions by the empty one
                drop = torch.rand(b["L"].shape[0], device=dev) < args.caption_dropout
                b["text"][drop] = empty_feat.to(dev)
                b["text_mask"][drop] = empty_mask.to(dev)
            pred = net(b["L"], b["text"], b["text_mask"])
            loss = F.l1_loss(pred, b["ab"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item() * data.AB_SCALE)

        row = {"epoch": epoch, "train_ab_err": round(float(np.mean(losses)), 3),
               "lr": f"{opt.param_groups[0]['lr']:.2e}", "minutes": round((time.time() - t0) / 60, 1)}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            s = validate(net, table, val_ids, dev)
            row.update({f"val_{k}": round(v, 4) for k, v in s.items() if k != "n"})
            if s["delta_e_ab"] < best:
                best = s["delta_e_ab"]
                save(ckpt_dir / "best.pt", epoch)
                row["best"] = "*"
        rows.append(row)
        save(ckpt_dir / "last.pt", epoch)
        print("  ".join(f"{k} {v}" for k, v in row.items()), flush=True)

    with open(log_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot([r["epoch"] for r in rows], [r["train_ab_err"] for r in rows], label="train |ab error|")
    v = [r for r in rows if "val_delta_e_ab" in r]
    ax.plot([r["epoch"] for r in v], [r["val_delta_e_ab"] for r in v], "o-", label="val dE_ab (full images)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Lab units")
    ax.set_title(args.run)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path("figures") / f"{args.run}_curve.png", dpi=110)
    print(f"best val dE_ab {best:.3f}; checkpoints in {ckpt_dir}; log {log_dir / 'log.csv'}")


if __name__ == "__main__":
    main()
