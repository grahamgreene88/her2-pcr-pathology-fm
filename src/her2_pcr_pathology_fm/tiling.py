"""Tissue masking and tile-coordinate generation.
Saves tile coordinates (not images) per slide with no stain normalisation.
"""

from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

TILE_PX = 224  # tile size the encoders receive
TARGET_MPP = 0.5  # microns per pixel, about 20x
MIN_TISSUE = 0.5  # keep tiles that are at least 50% tissue
MASK_PX_PER_TILE = 16  # each tile covers a 16 x 16 block of the mask
SAT_FLOOR = 15  # minimum saturation threshold (0-255)
SAT_CEILING = 25  # maximum, so pale tissue isn't dropped on densely stained slides


def tissue_mask(
    slide: openslide.OpenSlide, tile_l0: int, sat_ceil: float = SAT_CEILING
):
    """Binary tissue mask at a resolution where one tile = MASK_PX_PER_TILE pixels."""
    W, H = slide.dimensions
    scale = MASK_PX_PER_TILE / tile_l0
    size = (round(W * scale), round(H * scale))
    # exact size for the grid
    thumb = slide.get_thumbnail(size).convert("RGB").resize(size)

    sat = np.asarray(thumb.convert("HSV"))[..., 1]
    thresh = max(float(threshold_otsu(sat)), SAT_FLOOR)
    thresh = min(max(float(threshold_otsu(sat)), SAT_FLOOR), sat_ceil)
    mask = sat > thresh

    # one tile's worth of mask pixels
    min_area = MASK_PX_PER_TILE**2
    mask = closing(mask, footprint=disk(2))
    mask = remove_small_objects(mask, max_size=min_area - 1)
    mask = remove_small_holes(mask, max_size=min_area - 1)
    return mask, thresh, thumb


def tile_coords(mask: np.ndarray, tile_l0: int):
    """Top-left (x, y) level-0 coordinates of grid tiles with enough tissue."""
    k = MASK_PX_PER_TILE
    ny, nx = mask.shape[0] // k, mask.shape[1] // k
    blocks = mask[: ny * k, : nx * k].reshape(ny, k, nx, k)
    # tissue fraction per grid cell
    frac = blocks.mean(axis=(1, 3))
    rows, cols = np.nonzero(frac >= MIN_TISSUE)
    coords = np.stack([cols * tile_l0, rows * tile_l0], axis=1)
    return coords.astype(np.int64), frac[rows, cols].astype(np.float32)


def save_qc_png(thumb: Image.Image, coords: np.ndarray, tile_l0: int, path: Path):
    img = thumb.copy()
    draw = ImageDraw.Draw(img)
    k = MASK_PX_PER_TILE
    for x, y in coords:
        x0, y0 = x / tile_l0 * k, y / tile_l0 * k
        draw.rectangle([x0, y0, x0 + k - 1, y0 + k - 1], outline=(0, 200, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def tile_slide(slide_path: Path, out_path: Path, qc_path: Path) -> None:
    slide = openslide.OpenSlide(str(slide_path))
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp is None:
        raise ValueError(f"{slide_path.name}: no microns-per-pixel metadata")
    mpp = float(mpp)
    # e.g. 225 px at 0.4986 mpp
    tile_l0 = round(TILE_PX * TARGET_MPP / mpp)

    mask, thresh, thumb = tissue_mask(slide, tile_l0)
    coords, frac = tile_coords(mask, tile_l0)
    slide.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("tissue_frac", data=frac)
        f.attrs.update(
            tile_px=TILE_PX,
            target_mpp=TARGET_MPP,
            slide_mpp=mpp,
            tile_level0_px=tile_l0,
            otsu_threshold=thresh,
            min_tissue=MIN_TISSUE,
            mask_tissue_frac=float(mask.mean()),
        )
    tmp.rename(out_path)
    save_qc_png(thumb, coords, tile_l0, qc_path)


def run(
    data_root=Path("data"),
    out_root=Path("features/tiles"),
    qc_root=Path("results/qc_tiling"),
) -> pd.DataFrame:
    manifest = pd.read_csv(data_root / "manifest.csv", dtype={"patient_id": str})
    for row in manifest.itertuples():
        out = out_root / row.cohort / f"{row.patient_id}.h5"
        if out.exists():
            continue  # resumable
        print(f"tiling {row.cohort}/{row.patient_id}")
        tile_slide(
            data_root / row.slide_path,
            out,
            qc_root / row.cohort / f"{row.patient_id}.png",
        )

    qc = []
    for row in manifest.itertuples():
        with h5py.File(out_root / row.cohort / f"{row.patient_id}.h5") as f:
            qc.append(
                {
                    "patient_id": row.patient_id,
                    "cohort": row.cohort,
                    "label": row.label,
                    "n_tiles": len(f["coords"]),
                    **{
                        k: f.attrs[k]
                        for k in (
                            "tile_level0_px",
                            "otsu_threshold",
                            "mask_tissue_frac",
                        )
                    },
                }
            )
    qc = pd.DataFrame(qc)
    qc.to_csv(qc_root / "tiling_qc.csv", index=False)
    print(qc.groupby("cohort")["n_tiles"].describe())
    return qc


if __name__ == "__main__":
    run()
