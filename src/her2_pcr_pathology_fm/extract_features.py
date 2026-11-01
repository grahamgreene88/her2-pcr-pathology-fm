"""Extract tile embeddings with a frozen foundation model.

For each slide: read tiles at the coordinates from tiling.py, resize to 224 px, apply the encoder's own preprocessing,
embed, and save features + coords to one .h5 file per slide.
"""

import argparse
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

TILE_PX = 224


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


## Encoder
def load_lunit():
    model = timm.create_model(
        "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
        pretrained=True,
        num_classes=0,
    ).eval()
    cfg = timm.data.resolve_model_data_config(model)
    cfg["crop_pct"] = 1.0  # don't crop away tile edges
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, lambda m, x: m(x)


ENCODERS = {
    "lunit_vits16": {
        "load": load_lunit,
        "model_id": "1aurent/vit_small_patch16_224.lunit_dino",
    }
}


## Reading tiles
class TileDataset(Dataset):
    def __init__(self, slide_path, coords, tile_l0, transform):
        self.slide_path = str(slide_path)
        self.coords = coords
        self.tile_l0 = int(tile_l0)
        self.transform = transform
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)
        x, y = self.coords[i]
        tile = self.slide.read_region(
            (int(x), int(y)), 0, (self.tile_l0, self.tile_l0)
        ).convert("RGB")
        if tile.size != (TILE_PX, TILE_PX):
            tile = tile.resize((TILE_PX, TILE_PX), Image.BICUBIC)
        return self.transform(tile)


@torch.inference_mode()
def embed_slide(
    model,
    encode,
    transform,
    slide_path,
    coords,
    tile_l0,
    device,
    dtype,
    batch_size,
    num_workers,
) -> np.ndarray:
    ds = TileDataset(slide_path, coords, tile_l0, transform)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats = [encode(model, x.to(device, dtype=dtype)).float().cpu() for x in dl]
    return torch.cat(feats).numpy().astype(np.float16)


## Main loop
def run(
    encoder="lunit_vits16",
    data_root=Path("data"),
    tiles_root=Path("features/tiles"),
    out_root=Path("features"),
    batch_size=64,
    num_workers=4,
    half=False,
):
    device = get_device()
    dtype = torch.float16 if half else torch.float32
    model, transform, encode = ENCODERS[encoder]["load"]()
    model = model.to(device, dtype=dtype)
    print(f"encoder={encoder} device={device} dtype={dtype}")

    manifest = pd.read_csv(data_root / "manifest.csv", dtype={"patient_id": str})
    for row in tqdm(list(manifest.itertuples()), desc=encoder):
        out = out_root / encoder / row.cohort / f"{row.patient_id}.h5"
        if out.exists():
            continue  # resumable

        with h5py.File(tiles_root / row.cohort / f"{row.patient_id}.h5") as f:
            coords = f["coords"][:]
            tile_l0 = int(f.attrs["tile_level0_px"])

        t0 = time.perf_counter()
        feats = embed_slide(
            model,
            encode,
            transform,
            data_root / row.slide_path,
            coords,
            tile_l0,
            device,
            dtype,
            batch_size,
            num_workers,
        )
        rate = len(coords) / (time.perf_counter() - t0)

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as f:
            f.create_dataset("features", data=feats)
            f.create_dataset("coords", data=coords)
            f.attrs.update(
                encoder=encoder,
                model_id=ENCODERS[encoder]["model_id"],
                feature_dim=feats.shape[1],
                tile_px=TILE_PX,
                tile_level0_px=tile_l0,
                precision="fp16" if half else "fp32",
            )
        tmp.rename(out)  # atomic: no half-written files
        tqdm.write(
            f"{row.cohort}/{row.patient_id}: {len(coords)} tiles, {rate:.0f} tiles/s"
        )


if __name__ == "__main__":  # required for DataLoader workers on macOS
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="lunit_vits16", choices=ENCODERS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--half", action="store_true")
    a = p.parse_args()
    run(
        encoder=a.encoder,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        half=a.half,
    )
