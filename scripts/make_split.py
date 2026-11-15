"""Build the shared train/val/test split of colour originals (Matson colour slides + autochromes).

These are the paired images for the conditioned models (#16, #17) and Tier A evaluation:
greyscale L is the input, the true colour is the target, the catalog caption is the condition.
Every model, baselines included, must report test numbers on the SAME test ids, so the split
is written to splits/colour_split.csv, which IS committed (data/ is not).

Selection matches baseline.py: rows of matson.csv whose medium contains "color" but not
"hand-color" (hand-coloured prints are excluded: their colour is a painter's guess).

Steps
  1. download any missing images from tile.loc.gov into data/matson/images/ (resumable)
  2. drop ids whose image is missing or unreadable
  3. stratified random split (seed 693): autochromes and slides are split separately so
     each split gets its share of both

Usage
  python scripts/make_split.py                  # download + split (80/10/10)
  python scripts/make_split.py --no-download    # split only what is on disk
"""

import argparse
import csv
import random
import time
from pathlib import Path

import requests
from PIL import Image

CSV = Path("matson.csv")
IMG_DIR = Path("data/matson/images")
OUT = Path("splits/colour_split.csv")
USER_AGENT = "archival-photo-restoration (EECE 693 course project; github.com/lauremhsn/archival-photo-restoration)"
IMAGE_DELAY = 1.0


def is_colour(medium):
    m = medium.lower()
    return "color" in m and "hand-color" not in m


def is_autochrome(medium):
    return "autochrome" in medium.lower()


def download(rows):
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    todo = [r for r in rows if r["image_url"] and not (IMG_DIR / f"{r['id']}.jpg").exists()]
    print(f"{len(rows) - len(todo)} colour images already on disk, {len(todo)} to download")
    for i, r in enumerate(todo, 1):
        url = r["image_url"].split("#")[0]
        for attempt in range(5):
            time.sleep(IMAGE_DELAY)
            try:
                resp = session.get(url, timeout=60)
            except requests.RequestException as e:
                print(f"  network error ({e.__class__.__name__}); retrying")
                time.sleep(2 ** attempt * 5)
                continue
            if resp.status_code == 200:
                tmp = IMG_DIR / f"{r['id']}.jpg.part"
                tmp.write_bytes(resp.content)
                tmp.replace(IMG_DIR / f"{r['id']}.jpg")  # atomic
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt * 30)
                continue
            print(f"  skip {r['id']}: HTTP {resp.status_code}")
            break
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")


def readable(iid):
    p = IMG_DIR / f"{iid}.jpg"
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=693)
    args = ap.parse_args()

    with open(CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if is_colour(r["medium"])]
    print(f"{len(rows)} colour originals in {CSV}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        download(rows)

    ok = [r for r in rows if readable(r["id"])]
    print(f"{len(ok)} usable ({len(rows) - len(ok)} missing or unreadable)")

    rng = random.Random(args.seed)
    out = []
    for group in (True, False):  # autochromes, then slides: stratified
        g = sorted((r for r in ok if is_autochrome(r["medium"]) == group), key=lambda r: r["id"])
        rng.shuffle(g)
        n_test, n_val = round(len(g) * args.test), round(len(g) * args.val)
        for i, r in enumerate(g):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out.append({"id": r["id"], "split": split, "autochrome": int(group),
                        "title": r["title"], "notes": r["notes"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort(key=lambda r: (r["split"], r["id"]))
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "autochrome", "title", "notes"])
        w.writeheader()
        w.writerows(out)

    for s in ("train", "val", "test"):
        n = sum(r["split"] == s for r in out)
        a = sum(r["split"] == s and r["autochrome"] for r in out)
        print(f"  {s:<5} {n:>4}  ({a} autochromes)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
