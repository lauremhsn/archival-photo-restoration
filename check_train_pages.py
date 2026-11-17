"""
Sanity-checks the hand-saved training pages before ingest.

python check_train_pages.py
python check_train_pages.py --pages "train_pages/page_*.json" --eval-csv matson.csv
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

def expected_pages(n_pages=30, per_page=100, total_items=22061, seed=693):
    """Same shuffle matson_scraper.py uses, so the two can't drift apart."""
    pages = list(range(1, total_items // per_page + 2))
    random.Random(seed).shuffle(pages)
    return pages[:n_pages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="train_pages/page_*.json")
    ap.add_argument("--eval-csv", default="matson.csv")
    ap.add_argument("--n-pages", type=int, default=30)
    args = ap.parse_args()

    expected = expected_pages(args.n_pages)

    files = sorted(glob.glob(args.pages))
    if not files:
        sys.exit(f"nothing matched {args.pages}")

    print(f"found {len(files)} files in {args.pages}\n")

    ids, bad, empty = {}, [], []
    saved_pages = set()
    for path in files:
        base = os.path.basename(path)
        try:
            sp = int("".join(c for c in base if c.isdigit()))
            saved_pages.add(sp)
        except ValueError:
            pass
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            bad.append((base, str(exc)[:60]))
            continue
        results = data.get("results") if isinstance(data, dict) else data
        if not results:
            empty.append(base)
            continue
        for rec in results:
            raw = rec.get("id")
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            iid = str(raw or "").rstrip("/").split("/")[-1]
            if iid:
                ids.setdefault(iid, []).append(base)

    if bad:
        print("these arent json, probably saved the error page. re-save them:")
        for b, why in bad:
            print(f"  {b}: {why}")
        print()
    if empty:
        print("these parsed but theres no results key. re-save them too:")
        for e in empty:
            print(f"  {e}")
        print()

    missing = [p for p in expected if p not in saved_pages]
    extra = sorted(saved_pages - set(expected))
    if missing:
        print(f"still missing {len(missing)} of the {args.n_pages}: {missing}\n")
    if extra:
        print(f"these arent on the list: {extra}\n")

    dupes = {k: v for k, v in ids.items() if len(v) > 1}
    print(f"records: {sum(len(v) for v in ids.values())}")
    print(f"unique ids: {len(ids)}")
    if dupes:
        print(f"{len(dupes)} ids show up on more than one page")

    if os.path.exists(args.eval_csv):
        with open(args.eval_csv, encoding="utf-8") as fh:
            eval_ids = {r["id"] for r in csv.DictReader(fh)}
        overlap = set(ids) & eval_ids
        print(f"\neval set ({args.eval_csv}): {len(eval_ids)} ids")
        print(f"overlap: {len(overlap)}")
        if overlap:
            print("  uh oh, training and eval share ids. these gotta go:")
            for i in sorted(overlap)[:10]:
                print(f"    {i}")
            if len(overlap) > 10:
                print(f"    ... and {len(overlap) - 10} more")
        else:
            print("  clean split, we good")
    else:
        print(f"\ncouldnt find {args.eval_csv} so skipping the overlap check")

    ok = not (bad or empty or missing)
    print("\n" + ("all good, ready for ingest" if ok else "fix the stuff above first"))


if __name__ == "__main__":
    main()