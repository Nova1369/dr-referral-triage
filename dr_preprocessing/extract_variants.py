#!/usr/bin/env python3
"""
extract_variants.py - ResNet-50 features of altered photos: the Graham-filter ablation and the robustness tests.

Same network, weights, input size and resizing as extract_features.py. The only difference is one change made
to the 512 px photo before it is resized to 224 px:
  graham   Ben Graham's local-colour normalisation (dr_preprocess.enhance_graham): the colour-step ablation
  blur     Gaussian blur, radius 3 px at 512 px (about 1.3 px at 224 px): a slightly out-of-focus photo
  dark     every pixel at half its brightness: an under-exposed photo
  jpeg     re-saved as JPEG at quality 20: strong compression, as on a slow telemedicine link
  hflip    mirrored left-right: a left eye looks like a right eye (additional work: mirror-averaged features, mirror_fix.py)

--datasets chooses the photos: all three datasets for graham and hflip (they change training too), APTOS only for
the robustness tests (only test photos are altered, so only APTOS is needed).

Output: features/resnet50_224_<variant>.npz with X (photos x 2,048) and uid, in manifest order for the chosen
datasets. Resumable like extract_features.py: parts of 256 photos, --time-budget stops cleanly and the same
command continues.

Usage (from dr_preprocessing/):
  python extract_variants.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --variant graham
  python extract_variants.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth --variant blur
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import dr_preprocess as dp
from extract_features import IMAGENET_MEAN, IMAGENET_STD, PART, build_model, pick_device, sha256

VARIANTS = ("graham", "blur", "dark", "jpeg", "hflip")
DEFAULT_DATASETS = {"graham": "aptos,idrid,messidor2", "blur": "aptos", "dark": "aptos", "jpeg": "aptos",
                    "hflip": "aptos,idrid,messidor2"}


def alter(im: Image.Image, variant: str) -> Image.Image:
    if variant == "graham":
        return Image.fromarray(dp.enhance_graham(np.asarray(im)))
    if variant == "blur":
        return im.filter(ImageFilter.GaussianBlur(radius=3))
    if variant == "dark":
        return Image.fromarray(np.round(np.asarray(im, np.float32) * 0.5).astype(np.uint8))
    if variant == "hflip":
        return im.transpose(getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT)
    if variant == "jpeg":
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=20)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(variant)


def load_batch(paths, size, variant):
    out = np.empty((len(paths), 3, size, size), np.float32)
    for i, p in enumerate(paths):
        im = alter(Image.open(p).convert("RGB"), variant)
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC, reducing_gap=3.0)
        a = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = a.transpose(2, 0, 1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="ResNet-50 features of altered photos.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="../data_512")
    p.add_argument("--weights", required=True)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--datasets", default=None, help="comma-separated; default depends on the variant")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--time-budget", type=float, default=0)
    args = p.parse_args(argv)

    import torch
    t0 = time.time()
    data = Path(args.data)
    man = pd.read_csv(data / "manifest.csv")
    datasets = (args.datasets or DEFAULT_DATASETS[args.variant]).split(",")
    man = man[man["dataset"].isin(datasets)]
    paths = [str(data / r) for r in man["out_path"]]
    uids = np.array(man["uid"].astype(str).tolist())
    tag = f"resnet50_{args.size}_{args.variant}"
    feat_dir = data / "features"
    parts_dir = feat_dir / f"parts_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    weights = Path(args.weights)
    key = {"weights_sha256": sha256(weights), "input_size": args.size, "variant": args.variant,
           "n_images": len(uids), "uid_sha256": hashlib.sha256("\n".join(uids).encode()).hexdigest()}
    key_file = parts_dir / "run.json"
    if key_file.exists() and json.loads(key_file.read_text()) != key:
        sys.exit(f"{parts_dir} was made with other settings or photos; rename or delete it, then run again.")
    key_file.write_text(json.dumps(key, indent=2))
    model = build_model(weights, pick_device("cpu"))

    starts = list(range(0, len(paths), PART))
    todo = [s for s in starts if not (parts_dir / f"part_{s:05d}.npy").exists()]
    print(f"{args.variant}: {len(paths)} photos ({','.join(datasets)}), {len(starts) - len(todo)}/{len(starts)} parts done")
    t1, n_done = time.time(), 0
    for s in todo:
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        chunk = paths[s:s + PART]
        feats = []
        with torch.inference_mode():
            for b in range(0, len(chunk), args.batch):
                x = torch.from_numpy(load_batch(chunk[b:b + args.batch], args.size, args.variant))
                feats.append(model(x.contiguous(memory_format=torch.channels_last)).float().numpy())
        f = np.concatenate(feats).astype(np.float32)
        if not np.isfinite(f).all():
            sys.exit(f"non-finite features in part starting at {s}")
        tmp = parts_dir / f"part_{s:05d}.tmp.npy"
        np.save(tmp, f)
        os.replace(tmp, parts_dir / f"part_{s:05d}.npy")
        n_done += len(chunk)
        print(f"  part {s:05d} done  {n_done / (time.time() - t1):.1f} photos/s", flush=True)

    missing = [s for s in starts if not (parts_dir / f"part_{s:05d}.npy").exists()]
    if missing:
        print(f"time budget reached: {len(missing)} parts left; run the same command again")
        sys.exit(3)
    X = np.concatenate([np.load(parts_dir / f"part_{s:05d}.npy") for s in starts])
    np.savez(feat_dir / f"{tag}.npz", X=X, uid=uids)
    (feat_dir / f"{tag}.json").write_text(json.dumps({**key, "torch": torch.__version__,
                                                        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    print(f"saved {feat_dir / (tag + '.npz')} {X.shape} ({(time.time() - t0) / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
