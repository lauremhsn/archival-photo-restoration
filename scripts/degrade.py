"""Degradation simulator: turns clean color photos into synthetic "old photo" pairs.

For each clean image: clean color -> grayscale -> degradations applied in a realistic
order, each with a severity in [0, 1]. Saves the clean color target, the degraded input,
a mask of the structured damage and the exact parameters used, so every pair is
reproducible.

Taxonomy (team decision; the same names are used for the Matson hand labels):
  core   fade, noise, blur, scratches, dust
  extra  cracks, emulsion_loss  - Matson glass-plate damage documented by LoC
         jpeg                   - digitization artifacts
  Default random pool = core + extra. `tone` (sepia cast) is opt-in only: it puts
  color into the colorizer's input, and real tone is measured in the EDA instead.
Structured types (scratches, dust, cracks, emulsion_loss) are drawn into the mask.

Severity: 0 = barely visible, 1 = heaviest. Fixed with --severity (e.g. for a
robustness sweep of the colorizer), otherwise drawn uniformly per type and image.

Outputs (see README data layout)
  <out>/clean/{id}.png      clean color target (resized)
  <out>/degraded/{id}.png   grayscale + degradations (model input)
  <out>/masks/{id}.png      255 where structured damage was drawn
  <out>/params.jsonl        {"id", "seed", "types", "severity", "params", "size"} per image

Usage
  python scripts/degrade.py --src data/clean_source --out data/synthetic
  python scripts/degrade.py --src data/clean_source --out data/synthetic_blur_0.5 --types blur --severity 0.5
  python scripts/degrade.py --figure path/to/image.jpg
"""

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

CORE = ["fade", "noise", "blur", "scratches", "dust"]
EXTRA = ["cracks", "emulsion_loss", "jpeg"]
DEFAULT_POOL = CORE + EXTRA  # everything except tone (opt-in via --types / --pool)
STRUCTURED = ["scratches", "dust", "cracks", "emulsion_loss"]
# Order of application: the print/negative fades and tones, the optics blur, film noise,
# then physical damage on top, and JPEG compression from digitization last.
ORDER = ["fade", "tone", "blur", "noise", "scratches", "cracks", "dust", "emulsion_loss", "jpeg"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _scale(mask):
    """Size factor so line widths/speck sizes look the same at any resolution (1.0 at 512 px)."""
    return max(mask.shape) / 512


def _lerp(lo, hi, sev):
    """Map severity 0..1 onto a parameter range."""
    return lo + (hi - lo) * sev


def _composite(img, layer, value):
    """Blend `value` into img where the soft layer (0..1) is set."""
    a = layer[..., None]
    return img * (1 - a) + value * a


# ---------------------------------------------------------------- unstructured

def fade(img, mask, rng, sev):
    """Fading / contrast loss: squashed contrast and lifted blacks, like a faded print."""
    contrast = _lerp(0.9, 0.35, sev)
    lift = _lerp(0.0, 0.25, sev)
    mean = img.mean()
    img = (img - mean) * contrast + mean
    img = img * (1 - lift) + 255 * lift
    return img, {"contrast": contrast, "lift": lift}


def tone(img, mask, rng, sev):
    """Sepia / yellow cast, luminance-preserving."""
    strength = _lerp(0.2, 1.0, sev)
    tint = np.array([1.0, rng.uniform(0.78, 0.92), rng.uniform(0.5, 0.75)])  # RGB
    tint /= tint @ np.array([0.299, 0.587, 0.114])
    toned = img.mean(axis=2, keepdims=True) * tint
    return img * (1 - strength) + toned * strength, {"strength": strength, "tint": tint.tolist()}


def blur(img, mask, rng, sev):
    """Defocus (Gaussian) or camera-shake (motion) blur."""
    s = _scale(mask)
    if rng.random() < 0.6:
        sigma = _lerp(0.4, 3.0, sev) * s
        return cv2.GaussianBlur(img, (0, 0), sigma), {"kind": "gaussian", "sigma": sigma}
    length = max(2, round(_lerp(2, 14, sev) * s))
    angle = rng.uniform(0, 180)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k), {"kind": "motion", "length": length, "angle": angle}


def noise(img, mask, rng, sev):
    """Additive noise, film-grain style: monochrome Gaussian noise with a random grain size."""
    sigma = _lerp(2, 25, sev)
    size = rng.uniform(0, 1.2) * _scale(mask)
    n = rng.normal(0, 1, mask.shape).astype(np.float32)
    if size > 0.3:
        n = cv2.GaussianBlur(n, (0, 0), size)
        n /= n.std() + 1e-6
    return img + sigma * n[..., None], {"sigma": sigma, "grain_size": size}


def jpeg(img, mask, rng, sev):
    """Compression artifacts from digitization."""
    quality = round(_lerp(90, 15, sev))
    bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out.astype(np.float32), {"quality": quality}


# ---------------------------------------------------------------- structured

def scratches(img, mask, rng, sev):
    """Thin curved lines; mostly vertical, like scratches along a film strip."""
    h, w = mask.shape
    s = _scale(mask)
    n = round(_lerp(1, 15, sev))
    opacity = _lerp(0.4, 1.0, sev)
    layer = np.zeros((h, w), np.float32)
    t = np.linspace(0, 1, 60)[:, None]
    for _ in range(n):
        if rng.random() < 0.6:
            x0 = rng.uniform(0, w)
            p0 = np.array([x0, rng.uniform(0, 0.3 * h)])
            p2 = np.array([x0 + rng.normal(0, 0.05 * w), rng.uniform(0.7 * h, h)])
        else:
            p0, p2 = rng.uniform((0, 0), (w, h)), rng.uniform((0, 0), (w, h))
        p1 = (p0 + p2) / 2 + rng.normal(0, 0.1 * max(h, w), 2)
        pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
        thick = max(1, round(rng.uniform(0.6, 2.0) * s))
        cv2.polylines(layer, [pts], False, opacity * float(rng.uniform(0.8, 1.0)), thick, cv2.LINE_AA)
    bright = bool(rng.random() < 0.7)
    mask[layer > 0.2] = 255
    return _composite(img, layer, 255.0 if bright else 0.0), {"n": n, "opacity": opacity, "bright": bright}


def dust(img, mask, rng, sev):
    """Dust / speckle: small light and dark specks."""
    h, w = mask.shape
    s = _scale(mask)
    n = round(_lerp(10, 300, sev))
    light, dark = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for _ in range(n):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (max(1, round(rng.uniform(0.5, 3) * s)), max(1, round(rng.uniform(0.5, 3) * s)))
        cv2.ellipse(light if rng.random() < 0.6 else dark, c, axes, rng.uniform(0, 180), 0, 360,
                    float(rng.uniform(0.6, 1.0)), -1, cv2.LINE_AA)
    mask[(light > 0.2) | (dark > 0.2)] = 255
    img = _composite(img, light, 255.0)
    return _composite(img, dark, 0.0), {"n": n}


def cracks(img, mask, rng, sev):
    """Glass-plate breaks: long near-straight lines entering from an edge, occasionally branching."""
    h, w = mask.shape
    s = _scale(mask)
    n = round(_lerp(1, 4, sev))
    strength = _lerp(0.4, 1.0, sev)
    layer = np.zeros((h, w), np.float32)
    step = 4 * s

    def walk(x, y, ang, length, depth):
        pts = [(x, y)]
        for _ in range(int(length / step)):
            ang += rng.normal(0, 0.06)  # small jitter: glass breaks run nearly straight
            x, y = x + step * np.cos(ang), y + step * np.sin(ang)
            pts.append((x, y))
            if depth < 1 and rng.random() < 0.015:
                walk(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2),
                     length * rng.uniform(0.2, 0.5), depth + 1)
        thick = max(1, round((rng.uniform(0.7, 1.3) if depth == 0 else 0.6) * s))
        cv2.polylines(layer, [np.array(pts, np.int32)], False, strength, thick, cv2.LINE_AA)

    for _ in range(n):
        edge = int(rng.integers(4))
        x, y = [(rng.uniform(0, w), 0), (w - 1, rng.uniform(0, h)),
                (rng.uniform(0, w), h - 1), (0, rng.uniform(0, h))][edge]
        ang = np.arctan2(h / 2 - y, w / 2 - x) + rng.normal(0, 0.5)  # head roughly inward
        walk(x, y, ang, rng.uniform(0.6, 1.5) * max(h, w), 0)
    dark = bool(rng.random() < 0.5)
    mask[layer > 0.2] = 255
    return _composite(img, layer, 0.0 if dark else 255.0), {"n": n, "dark": dark, "strength": strength}


def emulsion_loss(img, mask, rng, sev):
    """Ragged patches where the emulsion flaked off, usually near the plate edges."""
    h, w = mask.shape
    s = _scale(mask)
    k = round(_lerp(1, 3, sev))
    blob = np.zeros((h, w), np.uint8)
    for _ in range(k):
        if rng.random() < 0.7:  # near an edge or corner
            cx = rng.choice([rng.uniform(0, 0.15), rng.uniform(0.85, 1)]) * w
            cy = rng.uniform(0, 1) * h
            if rng.random() < 0.5:
                cx, cy = rng.uniform(0, 1) * w, rng.choice([rng.uniform(0, 0.15), rng.uniform(0.85, 1)]) * h
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = _lerp(0.03, 0.12, sev) * rng.uniform(0.7, 1.0) * max(h, w)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 24))
        radii = r * rng.uniform(0.5, 1.4, 24)
        poly = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1).astype(np.int32)
        cv2.fillPoly(blob, [poly], 1)
    # ragged edges: blur the polygon and threshold it against smoothed noise
    soft = cv2.GaussianBlur(blob.astype(np.float32), (0, 0), 3 * s)
    rough = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5 * s)
    region = (soft + 0.09 * (rough - rough.mean()) / (rough.std() + 1e-6)) > 0.5
    layer = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 0.7 * s)
    light = bool(rng.random() < 0.6)
    fill = (rng.uniform(215, 250) if light else rng.uniform(5, 40)) + rng.normal(0, 6, (h, w, 1))
    mask[region] = 255
    a = layer[..., None]
    return img * (1 - a) + fill * a, {"n": k, "light": light}


FUNCS = {f.__name__: f for f in (fade, tone, blur, noise, jpeg, scratches, dust, cracks, emulsion_loss)}


# ---------------------------------------------------------------- pipeline

def _jsonable(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def degrade(rgb, seed, types=None, severity=None, pool=DEFAULT_POOL):
    """rgb: HxWx3 uint8 clean color image.

    types:    apply exactly these; if None, 1-4 are drawn at random from `pool`.
    severity: one value in [0, 1] for every type; if None, drawn uniformly per type.
    Types are always applied in ORDER. Returns (degraded uint8 HxWx3, mask uint8 HxW, params).
    """
    rng = np.random.default_rng(seed)
    if types is None:
        k = int(rng.integers(1, min(4, len(pool)) + 1))
        types = list(rng.choice(pool, size=k, replace=False))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    img = np.repeat(gray[..., None], 3, axis=2)
    mask = np.zeros(gray.shape, np.uint8)
    sevs, params = {}, {}
    for name in ORDER:
        if name in types:
            sev = float(rng.uniform(0, 1)) if severity is None else float(severity)
            img, p = FUNCS[name](img, mask, rng, sev)
            img = np.clip(img, 0, 255).astype(np.float32)
            sevs[name] = round(sev, 4)
            params[name] = {k: _jsonable(v) for k, v in p.items()}
    applied = [t for t in ORDER if t in types]
    return img.astype(np.uint8), mask, {"seed": seed, "types": applied, "severity": sevs, "params": params}


def imread_rgb(path, max_side):
    # imdecode(fromfile) instead of imread so non-ASCII Windows paths work
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if max_side and max(h, w) > max_side:
        f = max_side / max(h, w)
        bgr = cv2.resize(bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite(path, img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imencode(path.suffix, img)[1].tofile(str(path))


def image_seed(base_seed, image_id):
    """Stable per-image seed: same image + same --seed always gives the same degradation."""
    return zlib.crc32(f"{base_seed}:{image_id}".encode())


def run_batch(args):
    src, out = Path(args.src), Path(args.out)
    for sub in ("clean", "degraded", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    params_path = out / "params.jsonl"
    done = set()
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[:args.limit]
    todo = [p for p in files if p.stem not in done]
    print(f"{len(files)} images in {src}: {len(files) - len(todo)} already done, {len(todo)} to go")

    with open(params_path, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            rgb = imread_rgb(p, args.max_side)
            if rgb is None:
                print(f"  skip unreadable {p.name}")
                continue
            degraded, mask, params = degrade(rgb, image_seed(args.seed, p.stem),
                                             args.types, args.severity, args.pool)
            imwrite(out / "clean" / f"{p.stem}.png", rgb)
            imwrite(out / "degraded" / f"{p.stem}.png", degraded)
            imwrite(out / "masks" / f"{p.stem}.png", mask)
            f.write(json.dumps({"id": p.stem, **params, "size": list(rgb.shape[:2])}) + "\n")
            if i % 50 == 0 or i == len(todo):
                f.flush()
                print(f"  {i}/{len(todo)}")


def make_figure(args):
    """Severity sweep for the report: one row per degradation type, severity 0 -> 1 left to right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = imread_rgb(args.figure, args.max_side)
    if rgb is None:
        raise SystemExit(f"cannot read {args.figure}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rows = args.types or args.pool
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = 1 + len(levels)
    aspect = rgb.shape[0] / rgb.shape[1]
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 2.6, len(rows) * 2.6 * aspect + 0.6),
                             squeeze=False)
    for r, name in enumerate(rows):
        axes[r, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(name, fontsize=12)
        for c, sev in enumerate(levels, 1):
            # same seed across a row so only the severity changes, not the random layout
            img = degrade(rgb, args.seed + r, [name], sev)[0]
            axes[r, c].imshow(img, vmin=0, vmax=255)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("grayscale input", fontsize=11)
    for c, sev in enumerate(levels, 1):
        axes[0, c].set_title(f"severity {sev:g}", fontsize=11)
    fig.tight_layout()
    out = Path(args.figure_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def severity(v):
    v = float(v)
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError("severity must be between 0 and 1")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/clean_source", help="folder of clean color images (searched recursively)")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--max-side", type=int, default=512, help="resize so the long side is at most this (0 = keep)")
    ap.add_argument("--seed", type=int, default=693)
    ap.add_argument("--types", nargs="+", choices=ORDER, help="apply exactly these types to every image")
    ap.add_argument("--pool", nargs="+", choices=ORDER, default=DEFAULT_POOL,
                    help="types to draw 1-4 from at random when --types is not given (default: all except tone)")
    ap.add_argument("--severity", type=severity, help="fixed severity 0-1 for every type (default: random per type)")
    ap.add_argument("--limit", type=int, help="only process the first N images")
    ap.add_argument("--figure", help="make the severity-sweep figure from this one image instead of a batch run")
    ap.add_argument("--figure-out", default="figures/degradation_examples.png")
    args = ap.parse_args()
    make_figure(args) if args.figure else run_batch(args)


if __name__ == "__main__":
    main()
