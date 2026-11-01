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
| `scripts/label_matson.py` | Keyboard tool for hand-labeling degradation types on a fixed random sample of Matson images; one CSV per labeler in `data/labels/`, `--summary` gives counts and Cohen's kappa agreement. |
| `scripts/degrade.py` | Degradation simulator: clean color → grayscale → degradations with severity 0–1. Writes `data/synthetic/` (clean, degraded, masks, params.jsonl). Reproducible per image, resumable. |

### Degradation taxonomy

Shared by the simulator and the Matson hand labels (`data/labels/`), so synthetic and real damage can be compared. Every type has a **severity in [0, 1]** (0 = barely visible, 1 = heaviest).

| Type | What it simulates |
|---|---|
| `fade` | fading / contrast loss, lifted blacks |
| `noise` | additive noise, film-grain style |
| `blur` | defocus or motion blur |
| `scratches` | thin curved lines, mostly vertical *(in mask)* |
| `dust` | dust / speckle *(in mask)* |
| `cracks` | glass-plate breaks, long near-straight lines *(in mask)* |
| `emulsion_loss` | ragged patches where the emulsion flaked off *(in mask)* |
| `jpeg` | digitization / compression artifacts |

All of the above are in the default random pool. `tone` (sepia cast) exists but is opt-in only (`--types tone`): it adds color to the colorizer's input, and the real tone of Matson scans is measured in the EDA (distance from neutral gray in Lab) rather than hand-labeled.

```
python scripts/degrade.py --src data/clean_source --out data/synthetic                       # 1-4 random types, random severity
python scripts/degrade.py --src data/clean_source --out data/syn_blur_0.5 --types blur --severity 0.5   # robustness sweep
python scripts/degrade.py --figure path/to/image.jpg                                         # figures/degradation_examples.png
```

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
