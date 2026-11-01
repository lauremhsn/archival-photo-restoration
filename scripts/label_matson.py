"""Keyboard tool for hand-labeling degradation types on Matson images.

Shows one image at a time; toggle labels with number keys. Saves after every image
and resumes where you left off. Each labeler writes their own CSV, so files never
conflict; --summary merges them for counts and inter-labeler agreement.

Keys
  1-7        toggle a label (see LABELS)
  a          accept the automatic hints (blur / fade / noise) for this image
  Enter/Space  save and go to the next image
  Backspace  go back one image
  q / Esc    save and quit

Usage
  python scripts/label_matson.py --labeler rola            # label (200 images by default)
  python scripts/label_matson.py --summary                  # counts + agreement across labelers

The image sample is drawn once (random, fixed seed) and stored in data/labels/sample.txt,
so every labeler sees the same images in the same order; the first --overlap images
are the shared set used to measure agreement.
"""

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Team taxonomy for hand labels. Tone is measured in the EDA instead; jpeg is not
# visible enough to label by eye. Edit this list (and only this list) to change columns.
LABELS = ["fade", "noise", "blur", "scratches", "cracks", "dust", "emulsion_loss"]

IMG_DIR = Path("data/matson/images")
LABEL_DIR = Path("data/labels")
SAMPLE = LABEL_DIR / "sample.txt"
WINDOW = "Matson labeling"


# ---------------------------------------------------------------- automatic hints

HINT_FRACTION = 0.2  # flag the most extreme 20% of the sample for each measurable type


def image_stats(gray):
    """Sharpness, contrast and noise of the image centre (plate borders and edge markings excluded).

    lap_var:     variance of the Laplacian; low = few sharp edges (blur)
    range:       2nd-98th percentile intensity spread; low = faded / low contrast
    noise_sigma: noise estimate from the finest detail (median absolute deviation)
    """
    g = cv2.resize(gray, (512, round(512 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    g = g[h // 8: h - h // 8, w // 8: w - w // 8]
    lo, hi = np.percentile(g, [2, 98])
    detail = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    return {"lap_var": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
            "range": int(hi - lo),
            "noise_sigma": round(float(np.median(np.abs(detail)) / 0.6745), 2)}


def hint_cutoffs(ids):
    """Percentile cutoffs over the whole sample. Absolute thresholds don't work on these
    scans (film texture makes everything look 'sharp'), so hints are relative: an image is
    hinted when it is among the blurriest / most faded / noisiest HINT_FRACTION of the sample."""
    stats = [image_stats(cv2.imdecode(np.fromfile(str(IMG_DIR / f"{i}.jpg"), np.uint8), cv2.IMREAD_GRAYSCALE))
             for i in ids if (IMG_DIR / f"{i}.jpg").exists()]
    col = lambda k: np.array([s[k] for s in stats])
    return {"blur": np.percentile(col("lap_var"), 100 * HINT_FRACTION),
            "fade": np.percentile(col("range"), 100 * HINT_FRACTION),
            "noise": np.percentile(col("noise_sigma"), 100 * (1 - HINT_FRACTION))}


def auto_hints(stats, cut):
    """Hints are shown next to the labels and only applied when you press 'a'."""
    return {"blur": stats["lap_var"] <= cut["blur"],
            "fade": stats["range"] <= cut["fade"],
            "noise": stats["noise_sigma"] >= cut["noise"]}


# ---------------------------------------------------------------- storage

def load_sample(n, seed):
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE.exists():
        ids = SAMPLE.read_text(encoding="utf-8").split()
        if len(ids) >= n:
            return ids[:n]
    ids = sorted(p.stem for p in IMG_DIR.glob("*.jpg"))
    if len(ids) < n:
        print(f"note: only {len(ids)} images downloaded so far; sampling from those")
    random.Random(seed).shuffle(ids)
    ids = ids[:n]
    SAMPLE.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


def csv_path(labeler):
    return LABEL_DIR / f"matson_labels_{labeler}.csv"


def load_labels(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save_labels(path, rows):
    fields = ["id", *LABELS, "labeler", "timestamp", "lap_var", "range", "noise_sigma"]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    tmp.replace(path)  # atomic: a crash never leaves a half-written CSV


# ---------------------------------------------------------------- UI

def render(img, iid, pos, total, state, hints, done):
    h, w = img.shape[:2]
    scale = min(900 / h, 1200 / w, 1.0)
    view = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    panel = np.full((view.shape[0], 330, 3), 30, np.uint8)
    y = 30

    def put(text, color=(220, 220, 220), size=0.6, bold=1):
        nonlocal y
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, bold, cv2.LINE_AA)
        y += int(34 * size / 0.6)

    put(f"{pos + 1}/{total}   {iid}", size=0.55)
    put(f"labeled: {done}", (150, 150, 150), 0.5)
    y += 10
    for i, name in enumerate(LABELS, 1):
        on = state[name]
        hint = "  (hint)" if hints.get(name) else ""
        put(f"[{i}] {'X' if on else ' '} {name}{hint}", (90, 230, 90) if on else (200, 200, 200))
    y += 10
    for line in ("a: accept hints", "Enter/Space: next", "Backspace: back", "q/Esc: save & quit"):
        put(line, (150, 150, 150), 0.5)
    return np.hstack([view, panel])


def label(args):
    ids = load_sample(args.n, args.seed)
    path = csv_path(args.labeler)
    rows = load_labels(path)
    pos = next((i for i, iid in enumerate(ids) if iid not in rows), len(ids))
    if pos == len(ids):
        print(f"all {len(ids)} images already labeled in {path}")
        return
    print(f"labeling as '{args.labeler}' -> {path}; starting at image {pos + 1}/{len(ids)}")
    print("computing hint cutoffs over the sample...")
    cut = hint_cutoffs(ids)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while 0 <= pos < len(ids):
        iid = ids[pos]
        gray = cv2.imdecode(np.fromfile(str(IMG_DIR / f"{iid}.jpg"), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  missing {iid}.jpg, skipping")
            pos += 1
            continue
        stats = image_stats(gray)
        hints = auto_hints(stats, cut)
        prev = rows.get(iid)
        state = {n: (prev[n] == "1") if prev else False for n in LABELS}
        color = cv2.imdecode(np.fromfile(str(IMG_DIR / f"{iid}.jpg"), np.uint8), cv2.IMREAD_COLOR)

        while True:
            cv2.imshow(WINDOW, render(color, iid, pos, len(ids), state, hints, len(rows)))
            key = cv2.waitKeyEx(0)
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                save_labels(path, rows)
                cv2.destroyAllWindows()
                print(f"saved {len(rows)} labels to {path}")
                return
            if ord("1") <= key <= ord(str(len(LABELS))):
                name = LABELS[key - ord("1")]
                state[name] = not state[name]
            elif key == ord("a"):
                for name, on in hints.items():
                    if name in state:
                        state[name] = on
            elif key in (13, 32):  # Enter / Space
                rows[iid] = {"id": iid, **{n: int(v) for n, v in state.items()}, "labeler": args.labeler,
                             "timestamp": datetime.now().isoformat(timespec="seconds"), **stats}
                save_labels(path, rows)
                pos += 1
                break
            elif key == 8:  # Backspace
                pos = max(0, pos - 1)
                break
    cv2.destroyAllWindows()
    print(f"done: {len(rows)} labels in {path}")


# ---------------------------------------------------------------- summary

def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def summary(args):
    files = sorted(LABEL_DIR.glob("matson_labels_*.csv"))
    if not files:
        raise SystemExit(f"no label files in {LABEL_DIR}")
    per = {f.stem.removeprefix("matson_labels_"): load_labels(f) for f in files}
    for who, rows in per.items():
        n = len(rows)
        print(f"\n{who}: {n} images")
        for name in LABELS:
            k = sum(r.get(name) == "1" for r in rows.values())
            print(f"  {name:<14} {k:>4}  ({100 * k / max(n, 1):.0f}%)")
    names = list(per)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = per[names[i]], per[names[j]]
            shared = sorted(set(a) & set(b))
            print(f"\nagreement {names[i]} vs {names[j]} on {len(shared)} shared images (Cohen's kappa):")
            for name in LABELS:
                ka = cohen_kappa([a[s][name] == "1" for s in shared], [b[s][name] == "1" for s in shared]) \
                    if shared else float("nan")
                print(f"  {name:<14} {ka:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler", help="your name; labels go to data/labels/matson_labels_<name>.csv")
    ap.add_argument("--n", type=int, default=200, help="sample size (fixed once sample.txt exists)")
    ap.add_argument("--seed", type=int, default=693)
    ap.add_argument("--summary", action="store_true", help="print label counts and inter-labeler agreement")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    elif args.labeler:
        label(args)
    else:
        ap.error("give --labeler NAME to label, or --summary")


if __name__ == "__main__":
    main()
