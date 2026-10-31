# archival-photo-restoration

EECE 693 — Topic 10: Restoration and Historically Grounded Colorization of Archival Photographs.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Data is **never committed** (`data/` is git-ignored). Everyone keeps a local copy with exactly this structure so paths in code work on every machine.

```
data/
  matson/
    images/{id}.jpg             mid-res JPGs from loc.gov (id = LoC item id, e.g. mpc2004001234)
    metadata.jsonl              raw LoC API records, one per line (+ _id, _image_url, _master_url, is_stereograph)
  realold/                      Pik-Fix RealOld (200 real old photos + expert restorations)
    degraded/{name}.png
    restored/{name}.png         same file name as its degraded pair
  clean_source/                 clean color images used to synthesize training pairs (Person Y)
  synthetic/
    clean/{id}.png              clean color target
    degraded/{id}.png           grayscale + simulated degradation (model input)
    masks/{id}.png              binary mask of structured damage (scratches, cracks, tears)
    params.jsonl                degradation types + parameters + seed per {id}
  labels/
    matson_degradation_labels.csv   one row per image, 0/1 column per degradation type, labeler
```

Naming rules: file stems are IDs (no spaces); the same ID links an image across `clean/`, `degraded/`, `masks/` and `params.jsonl`.

## Scripts

| Script | What it does |
|---|---|
| `scripts/matson_scraper.py` | Downloads Matson metadata + images from the loc.gov JSON API. Resumable, rate-limited (~20 req/min API limit — don't remove the delays). |

```
python scripts/matson_scraper.py --max-items 50                     # smoke test
python scripts/matson_scraper.py --max-items 3000 --random-pages    # real run (~1 h)
```

If `www.loc.gov` answers the script with a 403 Cloudflare check (it does from some networks), save the metadata pages from a browser and let the script do the images (`tile.loc.gov` accepts scripts):

```
python scripts/matson_scraper.py --max-items 3000 --random-pages --print-page-urls   # writes data/matson/pages_to_save.txt
# open each URL in a browser, save it as the listed name into data/matson/raw_pages/
python scripts/matson_scraper.py --max-items 3000 --pages-dir data/matson/raw_pages  # builds metadata.jsonl + downloads images
```
