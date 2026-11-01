"""
Reads page*.json saved from the browser, reports caption coverage, writes
matson.csv, and downloads images from tile.loc.gov.

python ingest.py
python ingest.py --download --limit 300
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from urllib import error, request

OUT_CSV = "matson.csv"
IMG_DIR = "data/matson"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"

PERIOD_NAMES = {
    "israel": "palestine",
    "west bank": "palestine",
    "gaza strip": "palestine",
    "gaza": "palestine",
    "jordan": "transjordan",
    "golan heights": "syria",
}


def normalise_place(text):
    parts = [p.strip() for p in text.split(";") if p.strip()]
    out = []
    for p in parts:
        p = PERIOD_NAMES.get(p.lower(), p)
        if p not in out:
            out.append(p)
    return "; ".join(out)

def as_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "; ".join(p for p in (as_text(x) for x in v) if p)
    if isinstance(v, dict):
        return as_text(v.get("title") or list(v.values()))
    return str(v).strip()


def pick_image(urls):
    if not isinstance(urls, list):
        return ""
    for u in urls:
        if u.endswith("v.jpg") or "v.jpg#" in u:
            return u.split("#")[0]
    jpgs = [u.split("#")[0] for u in urls if ".jpg" in u]
    return jpgs[-1] if jpgs else ""


def extract(rec):
    inner = rec.get("item") or {}
    return {
        "id": as_text(rec.get("id")).rstrip("/").split("/")[-1],
        "title": as_text(rec.get("title")),
        "date": as_text(inner.get("created_published") or rec.get("date")),
        "sort_date": as_text(inner.get("sort_date")),
        "location": normalise_place(as_text(rec.get("location"))),
        "location_raw": as_text(rec.get("location")),
        "subject": as_text(rec.get("subject")),
        "medium": as_text(inner.get("medium")),
        "notes": as_text(inner.get("notes")),
        "image_url": pick_image(rec.get("image_url")),
    }


def load():
    files = sorted(glob.glob("page*.json"))
    if not files:
        sys.exit("No page*.json found. Save the browser pages into this folder first.")

    rows, seen = [], set()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        results = data.get("results") if isinstance(data, dict) else data
        if not results:
            print(f"  {path}: no results key")
            continue
        n = 0
        for rec in results:
            row = extract(rec)
            if row["id"] and row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
                n += 1
        print(f"  {path}: {n} records")
    return rows


def report(rows):
    total = len(rows)
    print(f"\n{total} unique records\n")

    print(f"{'field':<12} {'present':>8} {'rate':>7} {'med words':>10}")
    print("-" * 42)
    for f in ["title", "date", "location", "subject", "notes", "image_url"]:
        vals = [r[f] for r in rows if r[f]]
        wc = int(statistics.median([len(v.split()) for v in vals])) if vals else 0
        print(f"{f:<12} {len(vals):>8} {len(vals)/total*100:>6.1f}% {wc:>10}")
    print("-" * 42)

    rich = [r for r in rows if len(r["title"].split()) >= 5 and r["image_url"]]
    print(f"\nUsable for conditioning (title >=5 words + image): "
          f"{len(rich)} ({len(rich)/total*100:.1f}%)")

    auto = sum(1 for r in rows if "autochrome" in r["medium"].lower())
    col = sum(1 for r in rows if "color" in r["medium"].lower()
              and "hand-color" not in r["medium"].lower())
    hand = sum(1 for r in rows if "hand-color" in r["medium"].lower())
    print(f"Colour originals: {col} (autochromes {auto}, hand-coloured excluded: {hand})")

    print("\nSample titles:")
    for r in rows[:5]:
        print(f"  {r['title'][:90]}")


def fetch(url, tries=3):
    for t in range(tries):
        try:
            req = request.Request(url, headers={"User-Agent": UA})
            with request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            if t < tries - 1:
                time.sleep(2 ** t * 3)
    return None

def download(rows, limit, passes=40, pause=120):
    os.makedirs(IMG_DIR, exist_ok=True)
    todo = [r for r in rows if r["image_url"]][:limit]
    print(f"\nTarget: {len(todo)} images in {IMG_DIR}/")

    for p in range(1, passes + 1):
        missing = [r for r in todo
                   if not os.path.exists(os.path.join(IMG_DIR, f"{r['id']}.jpg"))]
        if not missing:
            break
        print(f"\npass {p}: {len(todo) - len(missing)} done, {len(missing)} to go")
        got = 0
        for i, r in enumerate(missing, 1):
            blob = fetch(r["image_url"])
            if blob:
                with open(os.path.join(IMG_DIR, f"{r['id']}.jpg"), "wb") as fh:
                    fh.write(blob)
                got += 1
            if i % 25 == 0:
                print(f"  {i}/{len(missing)}  (+{got})")
            time.sleep(0.3)
        print(f"  pass {p}: +{got}")
        if got == 0 and p < passes:
            print(f"  nothing came through, waiting {pause}s")
            time.sleep(pause)

    have = sum(1 for r in todo
               if os.path.exists(os.path.join(IMG_DIR, f"{r['id']}.jpg")))
    print(f"\n{have}/{len(todo)} on disk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    rows = load()
    if not rows:
        sys.exit("No records parsed.")

    report(rows)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    if args.download:
        download(rows, args.limit)


if __name__ == "__main__":
    main()