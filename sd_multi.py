"""
last cheap shot at the structure problem: stack a structural controlnet under
the brightness one so the geometry is locked by edges instead of brightness.

python sd_multi.py
"""

import argparse
import csv
import glob
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import color as skcolor, feature
from skimage.metrics import structural_similarity as ssim

BASE = "runwayml/stable-diffusion-v1-5"
BRIGHT = "latentcat/control_v1p_sd15_brightness"
CANNY = "lllyasviel/control_v11p_sd15_canny"
SOFT = "lllyasviel/control_v11p_sd15_softedge"
OUT_DIR = "outputs/sd_multi"


def lab(a):
    return skcolor.rgb2lab(np.asarray(a).astype(np.float32) / 255.0)


def recombine(orig, pred):
    o, p = lab(orig), lab(pred)
    return (np.clip(skcolor.lab2rgb(
        np.dstack([o[:, :, 0], p[:, :, 1], p[:, :, 2]])), 0, 1) * 255).astype(np.uint8)


def structure_score(orig, pred):
    return float(ssim(lab(orig)[:, :, 0], lab(pred)[:, :, 0], data_range=100.0))


def chroma(a):
    l = lab(a)
    return float(np.sqrt(l[:, :, 1] ** 2 + l[:, :, 2] ** 2).mean())


def canny_map(grey_arr, lo=0.1, hi=0.2):
    e = feature.canny(grey_arr.astype(np.float32) / 255.0, sigma=2.0,
                      low_threshold=lo, high_threshold=hi)
    return Image.fromarray(np.stack([(e * 255).astype(np.uint8)] * 3, axis=2))


def label_strip(img, text, h=22):
    out = Image.new("RGB", (img.width, img.height + h), (20, 20, 20))
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((4, 5), text, fill=(235, 235, 235))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="a colour photograph")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs the gpu")

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    dt = torch.float16
    print("loading controlnets")
    bright = ControlNetModel.from_pretrained(BRIGHT, torch_dtype=dt)
    struct, struct_name, prep = None, None, None
    try:
        from controlnet_aux import HEDdetector
        struct = ControlNetModel.from_pretrained(SOFT, torch_dtype=dt)
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        prep, struct_name = (lambda g: hed(Image.fromarray(g))), "softedge"
    except Exception as exc:
        print(f"  no softedge ({str(exc)[:70]}), falling back to canny")
        struct = ControlNetModel.from_pretrained(CANNY, torch_dtype=dt)
        prep, struct_name = canny_map, "canny"
    print(f"  structural control: {struct_name}")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=[struct, bright], torch_dtype=dt,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.enable_attention_slicing()

    # structural weight, brightness weight
    WEIGHTS = [(0.5, 0.8), (0.8, 0.5), (1.0, 0.3), (1.2, 0.2)]

    files = sorted(glob.glob("data/matson/images/*.jpg"))[:args.n]
    if not files:
        raise SystemExit("no images in data/matson/images/")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"ready. {len(files)} images x {len(WEIGHTS)} weightings\n")

    rows = []
    t0 = time.time()
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        orig = Image.open(path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        g = np.asarray(orig.convert("L"))
        grey = Image.fromarray(np.stack([g] * 3, axis=2))
        smap = prep(g).resize((args.size, args.size))
        if smap.mode != "RGB":
            smap = smap.convert("RGB")

        panels = [label_strip(grey, "original"), label_strip(smap, struct_name)]
        print(stem)
        for ws, wb in WEIGHTS:
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            out = pipe(args.prompt, image=[smap, grey], num_inference_steps=args.steps,
                       controlnet_conditioning_scale=[ws, wb], generator=gen).images[0]
            s = structure_score(orig, out)
            fixed = recombine(orig, out)
            tag = f"{struct_name} {ws} / bright {wb}"
            panels.append(label_strip(Image.fromarray(fixed), f"{tag}  ssim {s:.2f}"))
            rows.append({"image": stem, "struct": struct_name, "w_struct": ws,
                         "w_bright": wb, "structure_ssim": round(s, 4),
                         "chroma_raw": round(chroma(out), 2),
                         "chroma_fixed": round(chroma(fixed), 2)})
            print(f"  {tag:<28} structure {s:.3f}  chroma {chroma(out):.1f}")

        w, h = panels[0].size
        sheet = Image.new("RGB", (w * len(panels), h), (20, 20, 20))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * w, 0))
        sheet.save(f"{OUT_DIR}/{stem}_multi.png")
        print(f"  -> {OUT_DIR}/{stem}_multi.png\n")

    with open(f"{OUT_DIR}/multi.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"done in {time.time()-t0:.0f}s\n")
    print(f"{'weighting':<28} {'structure':>10} {'chroma':>8}")
    for ws, wb in WEIGHTS:
        sel = [r for r in rows if r["w_struct"] == ws and r["w_bright"] == wb]
        print(f"{struct_name+' '+str(ws)+' / bright '+str(wb):<28} "
              f"{np.mean([r['structure_ssim'] for r in sel]):>10.3f} "
              f"{np.mean([r['chroma_fixed'] for r in sel]):>8.1f}")
    print("\nwe need structure above ~0.8 AND chroma near 13 (thats what the real")
    print("colour originals average). if nothing hits both, this route is done and")
    print("the negative result goes in the report instead.")


if __name__ == "__main__":
    main()
