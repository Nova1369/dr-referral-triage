"""Export (or check) the small checkpoint of the final referral model.

The model is the one every rule (C1, C2, C3) is built on: a 3-class logistic regression (no referral /
routine / urgent), C = 0.003, fitted on all APTOS 2019 photos, on standardised 2,048-d ResNet-50 features
(see extract_features.py). The checkpoint holds only numbers (scaler mean and scale, coefficients,
intercepts) - no images and no per-image data - so it is a few tens of kB and lives in the repository.

    python export_checkpoint.py export --data ../data_512 --out ../checkpoints/lr_aptos_ref3.npz
    python export_checkpoint.py check  --data ../data_512 --ckpt ../checkpoints/lr_aptos_ref3.npz

`check` reloads the checkpoint, refits the model from the features, and confirms the two agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C_LR = 0.003
CLASSES = ["no_referral", "routine", "urgent"]


def fit(data: Path, features: str):
    man = pd.read_csv(data / "manifest.csv")
    with np.load(data / "features" / features) as z:
        X, uid = z["X"], z["uid"]
    assert (uid == man["uid"].astype(str).to_numpy()).all(), "features do not match manifest.csv"
    apt = (man["dataset"] == "aptos").to_numpy()
    y = man["ref3"].to_numpy(int)[apt]
    scaler = StandardScaler().fit(X[apt])
    model = LogisticRegression(C=C_LR, max_iter=5000).fit(scaler.transform(X[apt]), y)
    return scaler, model, X, man


def predict(ckpt, X):
    Z = (X - ckpt["mean"]) / ckpt["scale"]
    s = Z @ ckpt["coef"].T + ckpt["intercept"]
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "check"])
    ap.add_argument("--data", default="../data_512")
    ap.add_argument("--features", default="resnet50_224.npz")
    ap.add_argument("--out", default="../checkpoints/lr_aptos_ref3.npz")
    ap.add_argument("--ckpt", default="../checkpoints/lr_aptos_ref3.npz")
    a = ap.parse_args()
    data = Path(a.data)
    scaler, model, X, man = fit(data, a.features)

    if a.mode == "export":
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_,
                            intercept=model.intercept_, classes=np.array(CLASSES))
        meta = {"model": "LogisticRegression", "C": C_LR, "max_iter": 5000, "classes": CLASSES,
                "train_set": "APTOS 2019 train (all photos)", "n_train": int((man["dataset"] == "aptos").sum()),
                "features": "ResNet-50 (timm resnet50_ram-a26f946b.pth) global-average-pooled, 224 px, 2048-d",
                "backbone_sha256": "a26f946bdcb2e618263d03d8b60b66a26f7788dbadc1940c315b2bf62020a3e7",
                "sklearn": sklearn.__version__, "numpy": np.__version__,
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", out, "and", out.with_suffix(".json"))
    else:
        ck = dict(np.load(a.ckpt))
        P_ck = predict(ck, X)
        P_fit = model.predict_proba(scaler.transform(X))
        d = float(np.abs(P_ck - P_fit).max())
        agree = float((P_ck.argmax(1) == P_fit.argmax(1)).mean())
        print(f"max |p_checkpoint - p_refit| = {d:.2e}; same argmax on {agree:.4%} of {len(X)} photos")
        print("OK" if d < 0.01 else "WARNING: larger difference than expected (library versions?)")


if __name__ == "__main__":
    main()
