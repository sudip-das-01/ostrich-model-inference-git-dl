from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


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
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class _ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, *, dataset_root: Path) -> None:
        if "image_path" in df.columns:
            col = "image_path"
        elif "path" in df.columns:
            col = "path"
        elif "image" in df.columns:
            col = "image"
        else:
            raise ValueError(
                "DL repo expects dataset/input.csv to include an 'image_path' column "
                "with relative paths like 'images/000001.png'."
            )

        self.ids = [int(x) for x in (df["idx"].tolist() if "idx" in df.columns else range(len(df)))]
        self.paths = [str(x) for x in df[col].tolist()]
        self.dataset_root = dataset_root
        self.tx = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        rel = self.paths[i]
        p = Path(rel)
        img_path = p if p.is_absolute() else (self.dataset_root / p)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image file referenced in CSV: {rel} (resolved: {img_path})")
        img = Image.open(img_path).convert("RGB")
        x = self.tx(img)
        return torch.tensor(self.ids[i], dtype=torch.long), x


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(model_path, device)
    batch_size = int(os.environ.get("DL_BATCH_SIZE", "256"))
    batch_size = max(1, batch_size)

    ds = _ImagePathDataset(df, dataset_root=Path("dataset"))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ids_all: list[int] = []
    preds_all: list[int] = []
    with torch.inference_mode():
        for ids_batch, xb in dl:
            ids_all.extend([int(x) for x in ids_batch.cpu().tolist()])
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds_all.extend(int(p) for p in preds)

    pd.DataFrame({"idx": ids_all, "target": preds_all}).to_csv("output/output.csv", index=False)


if __name__ == "__main__":
    main()

