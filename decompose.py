"""
splits delta E into its chroma bit and its hue bit so we can tell timid apart from just wrong.

the aggregate dE_ab smushes two totally different failures together: painting grey where there
should be colour, and painting confident colour at the wrong hue. CIE94 already splits these into
dC* and dH*, we js never reported them separately.

usage:
  python decompose.py --gt data/matson/images --pred outputs/sd768_real --split splits/colour_split.csv --name sd768_real
  python decompose.py --selftest
"""

import argparse
import csv
import os
import sys

import numpy as np
from skimage import color, io, transform

CHROMA_FLOOR = 5.0


def to_lab(rgb):
    if rgb.ndim == 2:
        rgb = np.dstack([rgb] * 3)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.float64:
        rgb = rgb.astype(np.float64) / 255.0
    return color.rgb2lab(rgb)


def decompose(lab_gt, lab_pred, floor=CHROMA_FLOOR):
    """per-pixel chroma and hue error. gives back a dict of arrays plus the masks."""
    a1, b1 = lab_gt[:, :, 1], lab_gt[:, :, 2]
    a2, b2 = lab_pred[:, :, 1], lab_pred[:, :, 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)

    dL = lab_pred[:, :, 0] - lab_gt[:, :, 0]
    dC = c2 - c1
    dE_ab = np.sqrt((a2 - a1) ** 2 + (b2 - b1) ** 2)

    # dH is whatever dE has left over once you pull out lightness and chroma.
    # clipping bc floating point can shove it a hair below zero
    dH_sq = np.maximum(dE_ab ** 2 - dC ** 2, 0.0)
    dH = np.sqrt(dH_sq)

    h1 = np.arctan2(b1, a1)
    h2 = np.arctan2(b2, a2)
    dh = np.degrees(np.abs(np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))))

    # hue means nothing near the grey axis so only trust it where both sides actually have colour
    hue_mask = (c1 >= floor) & (c2 >= floor)
    # chroma ratio only makes sense where the reference had colour in the first place
    chroma_mask = c1 >= floor

    return {
        "c_true": c1,
        "c_pred": c2,
        "dL": dL,
        "dC": dC,
        "dH": dH,
        "dE_ab": dE_ab,
        "dh_deg": dh,
        "hue_mask": hue_mask,
        "chroma_mask": chroma_mask,
    }


def summarise(d):
    hm, cm = d["hue_mask"], d["chroma_mask"]
    n = d["c_true"].size
    out = {
        "dE_ab": float(np.mean(d["dE_ab"])),
        "pct_gt_chromatic": 100.0 * cm.sum() / n,
        "pct_hue_valid": 100.0 * hm.sum() / n,
    }
    if cm.any():
        out["chroma_ratio"] = float(np.median(d["c_pred"][cm] / d["c_true"][cm]))
        out["dC_signed"] = float(np.mean(d["dC"][cm]))
        out["dC_abs"] = float(np.mean(np.abs(d["dC"][cm])))
    if hm.any():
        w = d["c_true"][hm]
        # weight by true chroma, a hue mistake on a vivid pixel matters way more than on a pale one
        out["hue_err_deg"] = float(np.sum(d["dh_deg"][hm] * w) / np.sum(w))
        out["hue_err_deg_unweighted"] = float(np.mean(d["dh_deg"][hm]))
        out["dH"] = float(np.mean(d["dH"][hm]))
        out["pct_hue_over_60"] = float(100.0 * np.mean(d["dh_deg"][hm] > 60.0))
    return out


EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def find_img(d, img_id):
    for ext in EXTS:
        p = os.path.join(d, img_id + ext)
        if os.path.exists(p):
            return p
    return None


def diagnose(gt_dir, pred_dir, ids):
    """says what is actually sitting on disk instead of js silently skipping everything."""
    print(f"\ngt dir: {gt_dir}  exists={os.path.isdir(gt_dir)}")
    print(f"pred dir: {pred_dir}  exists={os.path.isdir(pred_dir)}")
    if os.path.isdir(pred_dir):
        names = sorted(os.listdir(pred_dir))
        print(f"  {len(names)} things in the pred dir")
        if names:
            print(f"  first few: {names[:6]}")
            exts = sorted({os.path.splitext(n)[1].lower() for n in names})
            print(f"  extensions in there: {exts}")
        else:
            print("  its EMPTY, so whatever made it never actually saved the images")
    if os.path.isdir(gt_dir):
        print(f"  {len(os.listdir(gt_dir))} things in the gt dir")
    hit_gt = sum(1 for i in ids if find_img(gt_dir, i))
    hit_pr = sum(1 for i in ids if find_img(pred_dir, i))
    print(f"  out of {len(ids)} split ids: {hit_gt} have a gt image, {hit_pr} have a prediction")
    if hit_pr == 0 and os.path.isdir(pred_dir) and os.listdir(pred_dir):
        print("  predictions are there but none are named <id>.<ext>, so the naming is different")


def load_pair(gt_path, pred_path):
    gt = io.imread(gt_path)
    pred = io.imread(pred_path)
    lab_gt = to_lab(gt)
    if pred.shape[:2] != gt.shape[:2]:
        pred = transform.resize(pred, gt.shape[:2], anti_aliasing=True, preserve_range=True)
        pred = pred.astype(np.float64) / 255.0 if pred.max() > 1.0 else pred
        lab_pred = color.rgb2lab(pred[:, :, :3] if pred.ndim == 3 else np.dstack([pred] * 3))
    else:
        lab_pred = to_lab(pred)
    # keep the real L, were only judging colour here
    lab_pred[:, :, 0] = lab_gt[:, :, 0]
    return lab_gt, lab_pred


def read_split(path, which="test"):
    ids = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", which).strip() == which:
                ids.append(row["id"].strip())
    return ids


def selftest():
    """checks the decomposition identity and the 60 degree crossover on made up colours."""
    rng = np.random.default_rng(0)
    print("checking dE_ab^2 == dC^2 + dH^2 on random ab pairs")
    a1, b1 = rng.uniform(-60, 60, (64, 64)), rng.uniform(-60, 60, (64, 64))
    a2, b2 = rng.uniform(-60, 60, (64, 64)), rng.uniform(-60, 60, (64, 64))
    L = np.full((64, 64), 50.0)
    g = np.dstack([L, a1, b1])
    p = np.dstack([L, a2, b2])
    d = decompose(g, p, floor=0.0)
    lhs = d["dE_ab"] ** 2
    rhs = d["dC"] ** 2 + d["dH"] ** 2
    err = np.max(np.abs(lhs - rhs))
    print(f"  max abs mismatch: {err:.3e}")
    assert err < 1e-8, "decomposition identity broke"

    print("checking the crossover: grey costs C, right chroma at hue offset theta costs 2C sin(theta/2)")
    C = 30.0
    for theta in (30.0, 59.0, 60.0, 61.0, 120.0):
        grey_cost = C
        hue_cost = 2 * C * np.sin(np.radians(theta) / 2)
        # build it for real and measure it instead of js trusting the formula
        gt = np.dstack([[[50.0]], [[C]], [[0.0]]])
        pred_grey = np.dstack([[[50.0]], [[0.0]], [[0.0]]])
        ang = np.radians(theta)
        pred_hue = np.dstack([[[50.0]], [[C * np.cos(ang)]], [[C * np.sin(ang)]]])
        mg = decompose(gt, pred_grey, floor=0.0)["dE_ab"][0, 0]
        mh = decompose(gt, pred_hue, floor=0.0)["dE_ab"][0, 0]
        assert abs(mg - grey_cost) < 1e-9 and abs(mh - hue_cost) < 1e-9
        winner = "grey" if mg < mh else "coloured"
        print(f"  theta={theta:5.1f}  grey={mg:6.3f}  coloured={mh:6.3f}  metric prefers {winner}")

    print("checking that hue error near grey is unstable (this is why we mask)")
    tiny = np.dstack([[[50.0]], [[0.3]], [[0.0]]])
    tiny2 = np.dstack([[[50.0]], [[-0.3]], [[0.0]]])
    d = decompose(tiny, tiny2, floor=0.0)
    print(f"  two near-grey pixels 0.6 apart in ab: hue error {d['dh_deg'][0, 0]:.1f} deg, dE_ab {d['dE_ab'][0, 0]:.2f}")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt")
    ap.add_argument("--pred")
    ap.add_argument("--split")
    ap.add_argument("--name", default="run")
    ap.add_argument("--which", default="test")
    ap.add_argument("--floor", type=float, default=CHROMA_FLOOR)
    ap.add_argument("--out", default="results")
    ap.add_argument("--joint", action="store_true", help="also dump the 2d (chroma ratio, hue error) histogram")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    for req in ("gt", "pred", "split"):
        if not getattr(args, req):
            sys.exit(f"need --{req} (or js run it with --selftest)")

    ids = read_split(args.split, args.which)
    print(f"{len(ids)} ids in the {args.which} split")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    joint = np.zeros((40, 36), dtype=np.int64)
    missing = []

    for i, img_id in enumerate(ids, 1):
        gt_path = find_img(args.gt, img_id)
        pred_path = find_img(args.pred, img_id)
        if gt_path is None or pred_path is None:
            missing.append(img_id)
            continue
        lab_gt, lab_pred = load_pair(gt_path, pred_path)
        d = decompose(lab_gt, lab_pred, args.floor)
        s = summarise(d)
        s["id"] = img_id
        rows.append(s)
        if args.joint:
            cm = d["chroma_mask"] & d["hue_mask"]
            if cm.any():
                ratio = np.clip(d["c_pred"][cm] / d["c_true"][cm], 0, 2)
                hue = d["dh_deg"][cm]
                h, _, _ = np.histogram2d(ratio, hue, bins=[40, 36], range=[[0, 2], [0, 180]])
                joint += h.astype(np.int64)
        if i % 50 == 0:
            print(f"  {i}/{len(ids)}")

    if missing:
        print(f"skipped {len(missing)} with no gt or no prediction: {missing[:7]}{' ...' if len(missing) > 7 else ''}")
    if not rows:
        diagnose(args.gt, args.pred, ids)
        sys.exit("\nnothing got scored. the diagnosis up there says why")

    keys = [k for k in rows[0] if k != "id"]
    per_image = os.path.join(args.out, f"{args.name}_decomposed.csv")
    with open(per_image, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nscored {len(rows)} images, per-image numbers are in {per_image}\n")
    print(f"{args.name}:")
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            print(f"  {k}: mean {np.mean(vals):.3f}  median {np.median(vals):.3f}")

    if args.joint:
        jp = os.path.join(args.out, f"{args.name}_joint.npy")
        np.save(jp, joint)
        print(f"\njoint histogram saved to {jp}, axes are chroma ratio 0-2 (40 bins) and hue error 0-180 (36 bins)")


if __name__ == "__main__":
    main()
