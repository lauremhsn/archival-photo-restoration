"""
Matson caption coverage probe.

python caption_probe.py --pages 3
python caption_probe.py --pages 10 --out matson_meta.json
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from urllib import error, parse, request

COLLECTION = "g-eric-and-edith-matson-photographs"
BASE = f"https://www.loc.gov/collections/{COLLECTION}/"
PER_PAGE = 100
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"

FIELDS = ["title", "subject", "location", "date", "notes", "medium"]


def fetch_page(page):
    params = {
        "fo": "json",
        "at": "results,pagination",
        "c": PER_PAGE,
        "sp": page,
    }
    url = BASE + "?" + parse.urlencode(params)
    req = request.Request(url, headers={"User-Agent": UA})

    for attempt in range(4):
        try:
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 429:
                wait = 20 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {exc.code} on page {page}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"  page {page} failed: {exc}", file=sys.stderr)
            time.sleep(5)
    return None


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [as_text(v) for v in value]
        return " ".join(p for p in parts if p)
    if isinstance(value, dict):
        return " ".join(as_text(v) for v in value.values())
    return str(value).strip()


def extract(item):
    row = {}
    inner = item.get("item") or {}
    for field in FIELDS:
        row[field] = as_text(item.get(field)) or as_text(inner.get(field))

    urls = item.get("image_url") or []
    row["image_url"] = urls[-1] if isinstance(urls, list) and urls else ""
    row["id"] = as_text(item.get("id"))
    return row


def report(rows):
    total = len(rows)
    print()
    print("=" * 64)
    print(f"CAPTION COVERAGE  —  {total} records")
    print("=" * 64)
    print()
    print(f"{'field':<14} {'present':>9} {'rate':>8} {'med len':>9} {'med words':>11}")
    print("-" * 64)

    usable = {}
    for field in FIELDS:
        vals = [r[field] for r in rows if r[field]]
        n = len(vals)
        rate = n / total if total else 0.0
        med_len = int(statistics.median([len(v) for v in vals])) if vals else 0
        med_words = int(statistics.median([len(v.split()) for v in vals])) if vals else 0
        usable[field] = (rate, med_words)
        print(f"{field:<14} {n:>9} {rate*100:>7.1f}% {med_len:>9} {med_words:>11}")

    print("-" * 64)

    n_img = sum(1 for r in rows if r["image_url"])
    print(f"{'image_url':<14} {n_img:>9} {n_img/total*100:>7.1f}%")
    print()

    best = max(FIELDS, key=lambda f: usable[f][0] * min(usable[f][1], 20))
    rate, words = usable[best]
    print(f"Richest text field: {best}  ({rate*100:.0f}% present, ~{words} words median)")
    print()

    print("Sample values from that field:")
    shown = 0
    for r in rows:
        if r[best] and shown < 5:
            text = r[best]
            print(f"  - {text[:160]}{'...' if len(text) > 160 else ''}")
            shown += 1

    rich = sum(1 for r in rows if len(r[best].split()) >= 5)
    print()
    print(f"Records with >=5 words in {best}: {rich} ({rich/total*100:.1f}%)")
    print(f"Records with both text and an image: "
          f"{sum(1 for r in rows if r[best] and r['image_url'])}")
    print()
    print("Read it like this:")
    print("  >=80% with >=5 words: conditioning plan is solid")
    print("  50-80%: workable, train on the subset, report it")
    print("  <50%: rethink the multimodal section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="pages of 100 (default 3)")
    ap.add_argument("--out", default="matson_meta.json", help="where to save records")
    args = ap.parse_args()

    first = fetch_page(1)
    if not first:
        sys.exit("ERROR: could not reach the LoC API")

    pag = first.get("pagination") or {}
    print(f"Collection total: {pag.get('of', 'unknown')} items")
    print(f"Pulling {args.pages} page(s) of {PER_PAGE}")
    print()

    rows = [extract(it) for it in first.get("results", [])]
    print(f"  page 1: {len(rows)} records")

    for page in range(2, args.pages + 1):
        time.sleep(2)
        data = fetch_page(page)
        if not data:
            break
        got = [extract(it) for it in data.get("results", [])]
        if not got:
            break
        rows.extend(got)
        print(f"  page {page}: {len(got)} records")

    if not rows:
        sys.exit("ERROR: no records returned")

    report(rows)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"Saved {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()