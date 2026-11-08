#!/usr/bin/env python3
"""
extract_features.py - ResNet-50 features for every image in manifest.csv (step 2 of the mid report).

Each preprocessed 512 x 512 photo is resized to --size (default 224), normalised with the ImageNet
mean and standard deviation, and passed through an ImageNet-pretrained ResNet-50 [He et al. 2016].
The 2,048 numbers after the global average pooling layer are the image's features.

PCA is deliberately NOT applied here: it is fitted later inside each training fold (part of the
model pipeline), so the test folds and the external datasets never influence it.

Output (in DATA/features/):
  resnet50_<size>.npz    X: float32 array (n_images x 2048), uid: image ids in manifest order
  resnet50_<size>.json   settings, weights file and its SHA-256, library versions, timing

Runs are resumable: finished batches are kept in DATA/features/parts_resnet50_<size>/ and skipped.
With --time-budget the script stops cleanly after that many seconds; run it again to continue.
The parts folder records the weights, input size and image list (run.json); parts made with other
settings are never mixed in.

Usage:
  python extract_features.py --data ../data_512 --weights ../models/resnet50_ram-a26f946b.pth
  python extract_features.py --data ../data_512 --weights ... --size 512 --device mps   # Apple GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PART = 256                       # images per saved part file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_key(weights_sha: str, size: int, uids) -> dict:
    """What the saved parts depend on; a parts folder is only reused when all of it matches."""
    return {"weights_sha256": weights_sha, "input_size": int(size), "n_images": len(uids),
            "uid_sha256": hashlib.sha256("\n".join(uids).encode()).hexdigest()}


def load_batch(paths, size):
    """Read images, resize with antialiasing to size x size, normalise; returns N x 3 x size x size."""
    out = np.empty((len(paths), 3, size, size), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC, reducing_gap=3.0)
        a = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = a.transpose(2, 0, 1)
    return out


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(weights: Path, device):
    import torch
    import torchvision
    model = torchvision.models.resnet50()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing:
        sys.exit(f"The weights file does not match ResNet-50 (missing {missing[:5]} ...)")
    model.fc = torch.nn.Identity()             # keep the 2,048 pooled features
    return model.eval().to(device).to(memory_format=torch.channels_last)


def main(argv=None):
    p = argparse.ArgumentParser(description="ImageNet ResNet-50 features for the preprocessed fundus images.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="../data_512", help="output folder of dr_preprocess.py (has manifest.csv)")
    p.add_argument("--weights", required=True, help="ImageNet ResNet-50 weights (.pth state dict)")
    p.add_argument("--size", type=int, default=224, help="input size for ResNet-50 (224 = ImageNet size)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps (Apple GPU)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = all cores)")
    p.add_argument("--time-budget", type=float, default=0, help="stop after this many seconds (0 = no limit)")
    args = p.parse_args(argv)

    import torch
    t0 = time.time()
    data = Path(args.data)
    man = pd.read_csv(data / "manifest.csv")
    paths = [str(data / r) for r in man["out_path"]]
    uids = np.array(man["uid"].astype(str).tolist())      # fixed-width text: loads without allow_pickle
    tag = f"resnet50_{args.size}"
    feat_dir = data / "features"
    parts_dir = feat_dir / f"parts_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if args.threads or (os.cpu_count() or 1) > 1:
        torch.set_num_threads(args.threads or os.cpu_count())
    device = pick_device(args.device)
    weights = Path(args.weights)
    weights_sha = sha256(weights)
    key = run_key(weights_sha, args.size, uids.tolist())
    key_file = parts_dir / "run.json"
    if key_file.exists():
        if json.loads(key_file.read_text()) != key:
            sys.exit(f"{parts_dir} was made with other weights, input size or images.\n"
                     "Delete or rename that folder, then run again.")
    elif any(parts_dir.glob("part_*.npy")):
        sys.exit(f"{parts_dir} has no run.json (made by an older version of this script).\n"
                 "Delete or rename that folder, then run again.")
    else:
        key_file.write_text(json.dumps(key, indent=2))
    model = build_model(weights, device)

    starts = list(range(0, len(paths), PART))
    todo = [s for s in starts if not (parts_dir / f"part_{s:05d}.npy").exists()]
    print(f"{len(paths)} images, {len(starts) - len(todo)}/{len(starts)} parts already done; "
          f"ResNet-50 at {args.size} px on {device}")
    t1, n_done = time.time(), 0
    for s in todo:
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        chunk = paths[s:s + PART]
        feats = []
        with torch.inference_mode():
            for b in range(0, len(chunk), args.batch):
                x = torch.from_numpy(load_batch(chunk[b:b + args.batch], args.size))
                x = x.to(device).contiguous(memory_format=torch.channels_last)
                feats.append(model(x).float().cpu().numpy())
        f = np.concatenate(feats).astype(np.float32)
        if not np.isfinite(f).all():
            sys.exit(f"non-finite features in part starting at {s}")
        tmp = parts_dir / f"part_{s:05d}.tmp.npy"
        np.save(tmp, f)
        os.replace(tmp, parts_dir / f"part_{s:05d}.npy")    # never leave a half-written part
        n_done += len(chunk)
        rate = n_done / (time.time() - t1)
        left = len(paths) - sum((parts_dir / f"part_{q:05d}.npy").exists() * len(paths[q:q + PART]) for q in starts)
        print(f"  part {s:05d} done  {rate:.1f} img/s  {left} images left (~{left / rate / 60:.1f} min)", flush=True)

    missing = [s for s in starts if not (parts_dir / f"part_{s:05d}.npy").exists()]
    if missing:
        print(f"\nTime budget reached: {len(missing)} parts left. Run the same command again to continue.")
        sys.exit(3)

    X = np.concatenate([np.load(parts_dir / f"part_{s:05d}.npy") for s in starts])
    assert X.shape == (len(paths), 2048), X.shape
    np.savez(feat_dir / f"{tag}.npz", X=X, uid=uids)
    meta = {
        "features": f"{tag}.npz", "n_images": int(X.shape[0]), "dim": int(X.shape[1]),
        "model": "torchvision ResNet-50 architecture, global-average-pooled output (fc removed)",
        "weights_file": weights.name, "weights_sha256": weights_sha, "uid_sha256": key["uid_sha256"],
        "input_size": args.size, "resize": "PIL bicubic with antialiasing from the 512 px images",
        "normalisation": "ImageNet mean/std", "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "order": "same row order as manifest.csv (column uid)",
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    (feat_dir / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {feat_dir / (tag + '.npz')}  shape {X.shape}  ({(time.time() - t0) / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
