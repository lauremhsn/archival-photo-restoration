"""
Feasibility check for the pretrained route. Loads SD 1.5 + a recolor ControlNet,
colorizes one greyscale scan, reports peak VRAM.

python sd_check.py
python sd_check.py --image data/matson/images/<id>.jpg --prompt "Jerusalem. Olivet"
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor

CANDIDATES = [
    "latentcat/control_v1p_sd15_brightness",
    "ioclab/control_v1p_sd15_brightness",
]
BASE = "runwayml/stable-diffusion-v1-5"
OUT_DIR = "outputs/sd_check"


def recombine(orig_rgb, pred_rgb):
    """Keep the original luminance, take only the predicted chrominance."""
    o = skcolor.rgb2lab(orig_rgb.astype(np.float32) / 255.0)
    p = skcolor.rgb2lab(np.asarray(pred_rgb).astype(np.float32) / 255.0)
    merged = np.dstack([o[:, :, 0], p[:, :, 1], p[:, :, 2]])
    return (np.clip(skcolor.lab2rgb(merged), 0, 1) * 255).astype(np.uint8)


def load_pipe(controlnet_id, dtype):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    net = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE, controlnet=net, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="a colour photograph of Jerusalem")
    ap.add_argument("--controlnet", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no cuda here, this needs the gpu")
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu: {name}  {total:.1f} GB")
    dtype = torch.float16

    path = args.image
    if path is None:
        hits = sorted(glob.glob("data/matson/images/*.jpg"))
        if not hits:
            raise SystemExit("no images found, pass --image")
        path = hits[0]
    print(f"image: {path}")

    img = Image.open(path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    grey = np.asarray(img.convert("L"))
    control = Image.fromarray(np.stack([grey] * 3, axis=2))

    ids = [args.controlnet] if args.controlnet else CANDIDATES
    pipe = None
    for cid in ids:
        try:
            print(f"trying controlnet: {cid}")
            pipe = load_pipe(cid, dtype)
            print(f"  loaded {cid}")
            break
        except Exception as exc:
            print(f"  nope: {str(exc)[:160]}")
    if pipe is None:
        raise SystemExit("couldnt load any of the controlnets, check the ids or the network")

    pipe = pipe.to(dev)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.time()
    out = pipe(args.prompt, image=control, num_inference_steps=args.steps,
               generator=gen).images[0]
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    raw = np.asarray(out)
    fixed = recombine(np.asarray(img), out)
    Image.fromarray(raw).save(f"{OUT_DIR}/{stem}_raw.png")
    Image.fromarray(fixed).save(f"{OUT_DIR}/{stem}_lfixed.png")

    def chroma(a):
        lab = skcolor.rgb2lab(a.astype(np.float32) / 255.0)
        return float(np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).mean())

    lo = skcolor.rgb2lab(np.asarray(img).astype(np.float32) / 255.0)[:, :, 0]
    lr = skcolor.rgb2lab(raw.astype(np.float32) / 255.0)[:, :, 0]

    print(f"\n{args.steps} steps in {secs:.1f}s  ({secs/args.steps:.2f}s/step)")
    print(f"peak vram {peak:.2f} GB of {total:.1f}")
    print(f"mean chroma: raw {chroma(raw):.1f}  after L fix {chroma(fixed):.1f}")
    print(f"luminance drift from the vae: {np.abs(lo - lr).mean():.1f} L units")
    print(f"\nwrote {OUT_DIR}/{stem}_raw.png and {stem}_lfixed.png")


if __name__ == "__main__":
    main()