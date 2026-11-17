"""
full tier A on the colour originals with sd, zero shot, at 768.
runs twice: once with the catalogue title as the prompt, once generic.
the gap between them is the first real measurement of whether metadata does anything.

python sd_tier_a.py
resumes if it dies, just run it again.
"""

import argparse, csv, os, time
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE = "runwayml/stable-diffusion-v1-5"
CONTROLNET = "latentcat/control_v1p_sd15_brightness"
IMG_DIR = "data/matson/images"
GENERIC = "a colour photograph"


def lab(a):
    return skcolor.rgb2lab(np.asarray(a).astype(np.float32) / 255.0)


def recombine(orig, pred):
    o, p = lab(orig), lab(pred)
    return (np.clip(skcolor.lab2rgb(
        np.dstack([o[:, :, 0], p[:, :, 1], p[:, :, 2]])), 0, 1) * 255).astype(np.uint8)


def colourfulness(a):
    x = np.asarray(a).astype(np.float64)
    rg = x[:, :, 0] - x[:, :, 1]
    yb = 0.5 * (x[:, :, 0] + x[:, :, 1]) - x[:, :, 2]
    return float((np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                  + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255.0)


def score(truth, pred):
    t, p = np.asarray(truth), np.asarray(pred)
    lt, lp = lab(t), lab(p)
    return {
        "psnr": round(float(psnr(t, p, data_range=255)), 3),
        "ssim": round(float(ssim(t, p, channel_axis=2, data_range=255)), 4),
        "delta_e": round(float(np.sqrt(((lt - lp) ** 2).sum(axis=2)).mean()), 3),
        "delta_e_ab": round(float(np.sqrt(((lt[:, :, 1:] - lp[:, :, 1:]) ** 2).sum(axis=2)).mean()), 3),
        "colourfulness_pred": round(colourfulness(p), 4),
        "colourfulness_ref": round(colourfulness(t), 4),
    }


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {r["id"] for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cn", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the gpu")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    with open("matson.csv", encoding="utf-8") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}

    todo = []
    for i, m in meta.items():
        med = m.get("medium", "").lower()
        if "color" not in med or "hand-color" in med:
            continue
        p = os.path.join(IMG_DIR, i + ".jpg")
        if os.path.exists(p):
            todo.append((i, p, m.get("title", "").strip()))
    todo.sort()
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} colour originals\n")

    net = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_tiling()
    except AttributeError:
        pass
    pipe.set_progress_bar_config(disable=True)

    fields = ["id", "psnr", "ssim", "delta_e", "delta_e_ab",
              "colourfulness_pred", "colourfulness_ref", "title"]

    for mode in ("title", "generic"):
        out_csv = f"results_tier_a_sd{args.size}_{mode}.csv"
        already = done_ids(out_csv)
        remaining = [t for t in todo if t[0] not in already]
        print(f"=== {mode} prompts === {len(already)} done, {len(remaining)} to go")
        if not remaining:
            print("  nothing left\n")
            continue

        new = not os.path.exists(out_csv)
        fh = open(out_csv, "a", newline="", encoding="utf-8")
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new:
            wr.writeheader()

        t0 = time.time()
        for n, (iid, path, title) in enumerate(remaining, 1):
            prompt = (title[:300] if (mode == "title" and title) else GENERIC)
            truth = Image.open(path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
            g = np.asarray(truth.convert("L"))
            grey = Image.fromarray(np.stack([g] * 3, axis=2))
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            try:
                out = pipe(prompt, image=grey, height=args.size, width=args.size,
                           num_inference_steps=args.steps,
                           controlnet_conditioning_scale=args.cn, generator=gen).images[0]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {iid} oom, skipped")
                continue
            row = score(truth, Image.fromarray(recombine(truth, out)))
            row["id"] = iid
            row["title"] = title
            wr.writerow(row)
            fh.flush()
            if n % 10 == 0 or n == len(remaining):
                el = time.time() - t0
                eta = el / n * (len(remaining) - n) / 60
                print(f"  {n}/{len(remaining)}  {el/n:.1f}s each  ~{eta:.0f} min left")
        fh.close()
        print()

    print("both done. now run:  python sd_compare.py")


if __name__ == "__main__":
    main()
