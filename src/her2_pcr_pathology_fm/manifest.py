"""Build and validate the slide manifest: one row per slide, joined to its label."""

from pathlib import Path

import openslide
import pandas as pd

# Differences between cohorts, rest of script is shared
COHORTS = {
    "yale": {
        "id_col": "Patient",
        "label_col": "Responder",
        "label_map": {"responder": 1, "nonresponder": 0},
        "slide_to_id": lambda p: p.stem,  # O09-03495.svs -> O09-03495
    },
    "impress": {
        "id_col": "Unique ID",
        "label_col": "pCR",
        "label_map": {"1": 1, "0": 0},
        "slide_to_id": lambda p: p.stem.split("_")[0],  # 061_HE.svs -> 061
    },
}

"Build and validate the slide manifest: one row per slide, joined to its label." ""
from pathlib import Path

import openslide
import pandas as pd

# Per-cohort differences live here; everything else is shared.
COHORTS = {
    "yale": {
        "id_col": "Patient",
        "label_col": "Responder",
        "label_map": {"responder": 1, "nonresponder": 0},
        "slide_to_id": lambda p: p.stem,  # O09-03495.svs -> O09-03495
    },
    "impress": {
        "id_col": "Unique ID",
        "label_col": "pCR",
        "label_map": {"1": 1, "0": 0},
        "slide_to_id": lambda p: p.stem.split("_")[0],  # 061_HE.svs -> 061
    },
}


def load_labels(cohort_dir: Path, cohort: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cohort_dir / f"{cohort}_labels.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[cfg["id_col"]])  # drops blank export rows
    df["patient_id"] = df[cfg["id_col"]].str.strip()
    df["label_raw"] = df[cfg["label_col"]].str.strip().str.lower()
    df["label"] = df["label_raw"].map(cfg["label_map"])
    return df[["patient_id", "label_raw", "label"]]


def slide_info(path: Path) -> dict:
    s = openslide.OpenSlide(str(path))
    w, h = s.dimensions
    mpp = s.properties.get(openslide.PROPERTY_NAME_MPP_X)
    vendor = s.properties.get(openslide.PROPERTY_NAME_VENDOR)
    s.close()
    return {
        "width": w,
        "height": h,
        "mpp": float(mpp) if mpp else None,
        "vendor": vendor,
    }


def build_cohort(data_root: Path, cohort: str, cfg: dict) -> pd.DataFrame:
    cohort_dir = data_root / cohort
    slides = sorted(
        p for p in (cohort_dir / "slides").glob("*.svs") if not p.name.startswith("._")
    )
    rows = [
        {
            "patient_id": cfg["slide_to_id"](p),
            "cohort": cohort,
            "slide_path": str(p.relative_to(data_root)),
            **slide_info(p),
        }
        for p in slides
    ]
    slides_df = pd.DataFrame(rows)
    labels = load_labels(cohort_dir, cohort, cfg)

    # Raise data issues
    dup = slides_df["patient_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"{cohort}: multiple slides per patient: "
            f"{slides_df.loc[dup, 'patient_id'].tolist()}"
        )
    if labels["patient_id"].duplicated().any():
        raise ValueError(f"{cohort}: duplicate patient IDs in label file")

    df = slides_df.merge(labels, on="patient_id", how="left")
    if df["label_raw"].isna().any():
        raise ValueError(
            f"{cohort}: slides with no label row: "
            f"{df.loc[df['label_raw'].isna(), 'patient_id'].tolist()}"
        )
    if df["label"].isna().any():
        raise ValueError(
            f"{cohort}: unmapped label values: "
            f"{df.loc[df['label'].isna(), 'label_raw'].unique().tolist()}"
        )

    # Extra checks
    extra = set(labels["patient_id"]) - set(slides_df["patient_id"])
    print(f"{cohort}: {len(df)} slides; {len(extra)} label rows without a slide")
    off_mpp = df[(df["mpp"] < 0.45) | (df["mpp"] > 0.55)]
    if len(off_mpp):
        print(f"{cohort}: WARNING, unexpected mpp:\n{off_mpp[['patient_id', 'mpp']]}")

    return df.drop(columns="label_raw")


def build_manifest(data_root: Path) -> pd.DataFrame:
    df = pd.concat(
        [build_cohort(data_root, c, cfg) for c, cfg in COHORTS.items()],
        ignore_index=True,
    )
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    data_root = Path("data")
    manifest = build_manifest(data_root)
    print(manifest.groupby(["cohort", "label"]).size().unstack())
    manifest.to_csv(data_root / "manifest.csv", index=False)
