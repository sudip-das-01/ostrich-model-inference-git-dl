from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision import models


def discover_input_path() -> str:
    dataset_dir = Path("dataset")
    preferred = dataset_dir / "input.csv"
    if preferred.exists():
        return str(preferred)

    matches = sorted(dataset_dir.glob("input.*"))
    if not matches:
        raise FileNotFoundError("Missing input file. Expected dataset/input.<ext> (e.g. dataset/input.csv).")

    for p in matches:
        if p.suffix.lower() == ".csv":
            return str(p)

    if len(matches) == 1:
        return str(matches[0])

    raise FileNotFoundError(f"Multiple candidate inputs found: {[str(p) for p in matches]}")


def discover_model_path() -> str:
    model_dir = Path("model")
    preferred = model_dir / "model.pt"
    if preferred.exists():
        return str(preferred)

    matches = sorted(model_dir.glob("model.*"))
    if not matches:
        raise FileNotFoundError("Missing model file. Expected model/model.<ext> (e.g. model/model.pt).")

    # Prefer common torch extensions if multiple matches exist
    for ext in (".pt", ".pth"):
        for p in matches:
            if p.suffix.lower() == ext:
                return str(p)

    if len(matches) == 1:
        return str(matches[0])

    raise FileNotFoundError(f"Multiple candidate models found: {[str(p) for p in matches]}")


def _load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(num_classes=10).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_images_from_csv(df: pd.DataFrame) -> tuple[list[int], torch.Tensor]:
    """
    Expects:
      - dataset/images.npy with shape (N, 3, 32, 32) uint8
      - dataset/input.csv containing an `idx` column
    """
    if "idx" not in df.columns:
        raise ValueError("DL repo expects dataset/input.csv to include an 'idx' column.")
    idxs = [int(x) for x in df["idx"].tolist()]
    images_u8 = np.load("dataset/images.npy")  # (N,3,32,32) uint8
    if images_u8.ndim != 4 or images_u8.shape[1:] != (3, 32, 32):
        raise ValueError(f"Unexpected dataset/images.npy shape: {images_u8.shape} (expected N,3,32,32)")

    x = images_u8[idxs].astype(np.float32) / 255.0
    xs = torch.from_numpy(x)
    # Normalize same as training
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
    xs = (xs - mean) / std
    return idxs, xs


def main() -> None:
    os.makedirs("output", exist_ok=True)

    model_path = Path(os.environ.get("DL_MODEL_PATH") or discover_model_path())

    input_path = discover_input_path()
    df = pd.read_csv(input_path)
    if any(c in df.columns for c in ("target", "prediction")):
        raise ValueError("Input dataset must not contain 'target' or 'prediction' columns.")

    max_rows_env = os.environ.get("DL_MAX_ROWS")
    if max_rows_env:
        try:
            n = int(max_rows_env)
            if n > 0:
                df = df.head(n).copy()
        except Exception:
            pass

    ids, x = _load_images_from_csv(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(model_path, device)
    batch_size = int(os.environ.get("DL_BATCH_SIZE", "256"))
    batch_size = max(1, batch_size)

    preds_all: list[int] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds_all.extend(int(p) for p in preds)

    pd.DataFrame({"idx": ids, "target": preds_all}).to_csv("output/output.csv", index=False)


if __name__ == "__main__":
    main()

