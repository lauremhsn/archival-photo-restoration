"""
turns "we sampled 3000 records and found no colour" into an actual count, using the collection's own
format facets instead of random sampling.

the rule-of-three bound we had doesnt hold here bc the collection is sorted by format, so the colour
items sit in contiguous blocks and random page sampling can miss them completely. the facets count
every item so theres no sampling left to argue about.

heads up: the loc.gov api 403s us even with a browser UA, same as always. so the route that actually
works is the one we already use for page*.json, save the json from the browser and pass --from-json.
we can also get most of this straight out of page*.json, see medium_census() below.

usage:
  python census.py --from-json data/facets.json
  python census.py --from-pages .
"""

import argparse
import collections
import glob
import json
import re
import time
import urllib.parse
import urllib.request

# same UA ingest.py uses. loc.gov 403s anything that doesnt look like a browser
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")
COLL = "https://www.loc.gov/collections/g-eric-and-edith-matson-photograph-collection/"
PPOC_FORMATS = "https://www.loc.gov/pictures/collection/matpc/index/formats/"

# the collection record spells out which series are colour, so we can js count those
COLOUR_SERIES = ["LC-M303", "LC-M302", "LC-M343", "LC-M342"]
BW_SERIES = ["LC-M353", "LC-M352", "LC-M354", "LC-M351", "LC-M345"]

COLOUR_WORDS = ("color", "colour", "autochrome", "finlay", "transparenc", "slide")

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": COLL,
}


def get(url, params=None, tries=3):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == tries - 1:
                print(f"  gave up on {url}: {e}")
                return None
            time.sleep(2 * (attempt + 1))


def facets():
    print("asking the collection for its facets (this usually 403s, dw about it)")
    d = get(COLL, {"fo": "json", "at": "facets"})
    if not d:
        return None
    return parse_facets(d)


def parse_facets(d):
    out = {}
    for f in d.get("facets", []):
        name = f.get("type") or f.get("title") or "?"
        out[name] = {v.get("title"): v.get("count") for v in f.get("filters", [])}
    return out


def facets_from_file(path):
    """for when the api says no. save the facets json from the browser like we do for page*.json."""
    with open(path, encoding="utf-8") as f:
        return parse_facets(json.load(f))


def medium_census(repo_dir):
    """
    counts media straight out of the page*.json we already committed. no network, no 403s.
    this is the honest version of the census claim: it says what we actually hold.
    """
    recs = {}
    files = sorted(glob.glob(f"{repo_dir}/page*.json"))
    if not files:
        print(f"no page*.json in {repo_dir}")
        return None
    for p in files:
        with open(p, encoding="utf-8") as f:
            for r in json.load(f).get("results", []):
                if r.get("id"):
                    recs[r["id"]] = r
    print(f"{len(files)} page files, {len(recs)} unique records")

    med = collections.Counter()
    series = collections.Counter()
    for r in recs.values():
        it = r.get("item") or {}
        mv = it.get("medium") or it.get("medium_brief") or []
        if isinstance(mv, str):
            mv = [mv]
        med[(mv[0] if mv else "(no medium)")[:80]] += 1
        m = re.match(r"(LC-[A-Za-z0-9]+)", str(r.get("shelf_id") or ""))
        if m:
            series[m.group(1)] += 1

    print("\n== what medium everything is ==")
    for k, n in med.most_common(25):
        tag = "COLOUR" if any(w in k.lower() for w in ("color", "colour")) else ""
        hand = "  <- HAND-COLOURED, not ground truth" if "hand-color" in k.lower() else ""
        print(f"  {n:5d}  {k}{'  ' + tag if tag else ''}{hand}")

    print("\n== call number series ==")
    for k, n in series.most_common(20):
        kind = "colour" if k in COLOUR_SERIES else ("b&w" if k in BW_SERIES else "")
        print(f"  {n:5d}  {k} {kind}")

    auto = sum(n for k, n in med.items() if "autochrome" in k.lower())
    slides2x2 = sum(n for k, n in med.items() if "2 x 2" in k and "color" in k.lower())
    largefmt = sum(n for k, n in med.items()
                   if "transparency" in k.lower() and ("color" in k.lower() or "colour" in k.lower()))
    handcol = sum(n for k, n in med.items() if "hand-color" in k.lower())
    print(f"\nautochromes: {auto}")
    print(f"2x2 colour slides: {slides2x2}")
    print(f"large format colour transparencies: {largefmt}")
    print(f"hand-coloured (must stay OUT of ground truth): {handcol}")
    print("\nsay this in the report fr. the colour set is mostly 2x2 slides, not autochromes,")
    print("and the hand-coloured lantern slides are somebody elses colourisation not a record of colour.")
    return {"medium": dict(med), "series": dict(series)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--from-json", help="a facets json you saved from the browser")
    ap.add_argument("--from-pages", help="repo dir holding page*.json, counts media with no network")
    args = ap.parse_args()

    report = {}

    if args.from_pages:
        report["medium_census"] = medium_census(args.from_pages)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nsaved {args.save}")
        return

    if args.from_json:
        print(f"reading facets from {args.from_json}")
        fc = facets_from_file(args.from_json)
    else:
        fc = facets()

    if fc:
        report["facets"] = fc
        print("\nfacets that mention colour or a transparency format:")
        hits = 0
        for group, vals in fc.items():
            for title, count in (vals or {}).items():
                if title and any(w in title.lower() for w in COLOUR_WORDS):
                    print(f"  [{group}] {title}: {count}")
                    hits += 1
        if not hits:
            print("  nothing matched, print the whole facets dict and look by hand")
        print("\nall facet groups:", ", ".join(fc.keys()))
    else:
        print("facets call failed, as expected. two ways round it:")
        print("  1. open this in the browser, save the json, rerun with --from-json <file>:")
        print(f"     {COLL}?fo=json&at=facets")
        print("  2. or skip it entirely and run --from-pages . which uses page*.json we already have")
        print(f"  (and the format browse page, by eye, says 'Autochromes--Color' is 34: {PPOC_FORMATS})")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
