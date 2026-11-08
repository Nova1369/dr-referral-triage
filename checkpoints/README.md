# Checkpoints

| File | Size | What it is |
|---|---|---|
| `lr_aptos_ref3.npz` (+ `.json`) | 55 kB | The final model behind C1-C3: 3-class logistic regression (no referral / routine / urgent), C = 0.003, fitted on all 3,662 APTOS photos, with its feature standardiser (mean, scale). Made by `src/export_checkpoint.py export`; `check` refits and compares (max difference in probability 6e-7). |
| `resnet50_ram-a26f946b.pth` | 102 MB, **not committed** | Frozen ImageNet ResNet-50 backbone from timm [22]. Too large for git and not ours to re-host, so it is linked: https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth, SHA-256 `a26f946bdcb2e618263d03d8b60b66a26f7788dbadc1940c315b2bf62020a3e7`. `bash scripts/download_weights.sh` fetches and checks it. |

No network was fine-tuned, so these two files are the complete trained state of the project. The conformal
thresholds are not a checkpoint: they are refitted from calibration photos by design (C1 per fold, C2 per
clinic) and are written to `results/c1/thresholds.csv`.

Using the checkpoint on features `X` (n x 2048) from `src/extract_features.py`:

```python
import numpy as np
ck = np.load("checkpoints/lr_aptos_ref3.npz")
s = ((X - ck["mean"]) / ck["scale"]) @ ck["coef"].T + ck["intercept"]
p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)   # P(no referral, routine, urgent)
```
