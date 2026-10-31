"""Scrape the Matson Photograph Collection (Library of Congress) via the loc.gov JSON API.

Outputs
  data/matson/metadata.jsonl   one raw API record per line (+ a few derived fields)
  data/matson/images/{id}.jpg  one mid-resolution JPG per item

Resumable: items already in metadata.jsonl / images already on disk are skipped,
so just re-run the same command after a crash or a rate-limit block.

Usage
  python scripts/matson_scraper.py --max-items 50                    # smoke test
  python scripts/matson_scraper.py --max-items 3000 --random-pages   # real run
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

COLLECTION_URL = "https://www.loc.gov/collections/g-eric-and-edith-matson-photographs/"
USER_AGENT = "archival-photo-restoration (EECE 693 course project; github.com/lauremhsn/archival-photo-restoration)"

# loc.gov allows ~20 req/min on collection/search endpoints; exceeding it blocks you for an hour.
API_DELAY = 3.5
IMAGE_DELAY = 1.0
MAX_RETRIES = 5


def get(session, url, params=None, delay=API_DELAY):
    """GET with polite delay and exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        time.sleep(delay)
        try:
            r = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            wait = 2 ** attempt * 10
            print(f"  network error ({e}); retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("Retry-After", 2 ** attempt * 30))
            print(f"  HTTP {r.status_code}; backing off {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 403:
            if "Just a moment" in r.text:
                sys.exit("HTTP 403: loc.gov's Cloudflare bot check is refusing script access "
                         "from this network (not a rate limit). Re-running won't help.")
            sys.exit("HTTP 403 from loc.gov: you are probably blocked for an hour "
                     "(rate limit) — wait, then re-run; progress is kept.")
        r.raise_for_status()
    sys.exit(f"Giving up on {url} after {MAX_RETRIES} attempts; re-run later to resume.")


def item_id(record):
    """'http://www.loc.gov/item/mpc2004001234/PP/' -> 'mpc2004001234'."""
    raw = record.get("id") or record.get("url") or ""
    parts = [p for p in raw.rstrip("/").split("/") if p]
    if "item" in parts and parts.index("item") + 1 < len(parts):
        return parts[parts.index("item") + 1]
    return re.sub(r"[^A-Za-z0-9_-]", "_", parts[-1]) if parts else None


def pick_image(urls, target_width):
    """Choose the largest JPG no wider than target_width (falls back to smallest larger one).

    loc.gov image_url entries look like '...v.jpg#h=512&w=640'.
    """
    cands = []
    for u in urls or []:
        base, _, frag = u.partition("#")
        if not base.lower().endswith((".jpg", ".jpeg")):
            continue
        m = re.search(r"w=(\d+)", frag)
        h = re.search(r"h=(\d+)", frag)
        cands.append((int(m.group(1)) if m else 0, int(h.group(1)) if h else 0, base))
    if not cands:
        return None, 0, 0
    fitting = [c for c in cands if c[0] <= target_width]
    w, h, url = max(fitting) if fitting else min(cands)
    return url, w, h


def master_url(record):
    """TIFF master if listed, else the LoC resource page that links to it (fetch later if needed)."""
    for u in record.get("image_url") or []:
        if u.split("#")[0].lower().endswith((".tif", ".tiff")):
            return u.split("#")[0]
    for res in record.get("resources") or []:
        if res.get("url"):
            return res["url"]
    return None


def is_stereograph(record, w, h):
    text = " ".join(str(record.get(k, "")) for k in ("title", "subject", "description", "notes")).lower()
    return "stereo" in text or (h > 0 and w / h > 1.8)


def load_seen(meta_path):
    seen = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/matson", help="output directory (default: data/matson)")
    ap.add_argument("--max-items", type=int, default=3000)
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--random-pages", action="store_true",
                    help="visit result pages in random order to sample across the whole collection")
    ap.add_argument("--image-width", type=int, default=1024, help="max width of downloaded JPG")
    ap.add_argument("--skip-images", action="store_true", help="metadata only")
    ap.add_argument("--seed", type=int, default=693)
    ap.add_argument("--print-page-urls", action="store_true",
                    help="write the API page URLs to save by hand from a browser to <out>/pages_to_save.txt, then exit")
    ap.add_argument("--pages-dir", help="read metadata from API pages saved as .json files in this folder "
                                        "instead of calling the API (use when loc.gov refuses scripts)")
    ap.add_argument("--total-items", type=int, default=22061,
                    help="collection size, used with --print-page-urls (from the API's pagination.total)")
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    params = {"fo": "json", "c": args.per_page}

    def page_order(total_items):
        pages = list(range(1, total_items // args.per_page + 2))
        if args.random_pages:
            random.Random(args.seed).shuffle(pages)
        return pages

    if args.print_page_urls:
        n_pages = -(-args.max_items // args.per_page)  # ceil
        lines = [f"{COLLECTION_URL}?fo=json&c={args.per_page}&sp={p}   ->  save as page_{p:04d}.json"
                 for p in page_order(args.total_items)[:n_pages]]
        (out / "pages_to_save.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} URLs to {out / 'pages_to_save.txt'}")
        return

    def api_pages():
        first = get(session, COLLECTION_URL, {**params, "sp": 1}).json()
        print(f"collection: {first['pagination']['total']} items")
        for page in page_order(first["pagination"]["total"]):
            yield page, first if page == 1 else get(session, COLLECTION_URL, {**params, "sp": page}).json()

    def saved_pages():
        files = sorted(Path(args.pages_dir).glob("*.json"))
        print(f"reading {len(files)} saved pages from {args.pages_dir}")
        for p in files:
            yield p.name, json.loads(p.read_text(encoding="utf-8-sig"))  # -sig: tolerate a BOM from Windows editors

    # ---- Phase 1: metadata ----
    seen = load_seen(meta_path)
    print(f"{len(seen)} items already in {meta_path}")
    if len(seen) < args.max_items:
        with open(meta_path, "a", encoding="utf-8") as f:
            for page, data in (saved_pages() if args.pages_dir else api_pages()):
                if len(seen) >= args.max_items:
                    break
                new = 0
                for rec in data.get("results", []):
                    iid = item_id(rec)
                    if not iid or iid in seen or len(seen) >= args.max_items:
                        continue
                    url, w, h = pick_image(rec.get("image_url"), args.image_width)
                    rec.update({"_id": iid, "_image_url": url, "_image_w": w, "_image_h": h,
                                "_master_url": master_url(rec),
                                "is_stereograph": is_stereograph(rec, w, h)})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(iid)
                    new += 1
                f.flush()
                print(f"page {page}: +{new} (total {len(seen)})")

    # ---- Phase 2: images ----
    if args.skip_images:
        return
    with open(meta_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    todo = [r for r in records if r.get("_image_url") and not (img_dir / f"{r['_id']}.jpg").exists()]
    no_img = sum(1 for r in records if not r.get("_image_url"))
    print(f"images: {len(records) - len(todo) - no_img} done, {len(todo)} to download, {no_img} without a JPG")

    for i, rec in enumerate(todo, 1):
        r = get(session, rec["_image_url"], delay=IMAGE_DELAY)
        tmp = img_dir / f"{rec['_id']}.jpg.part"
        tmp.write_bytes(r.content)
        tmp.replace(img_dir / f"{rec['_id']}.jpg")  # atomic: no half-written JPGs after a crash
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} images")


if __name__ == "__main__":
    main()
