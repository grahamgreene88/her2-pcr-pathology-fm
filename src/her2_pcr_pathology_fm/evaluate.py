"""Evaluation harness: patient-level repeated CV, out-of-fold predictions, AUC with uncertainty."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def load_bags(
    cohort, encoder="lunit_vits16", data_root=Path("data"), feat_root=Path("features")
):
    """One bag per patient: an (n_tiles, dim) float32 array."""
    manifest = pd.read_csv(data_root / "manifest.csv", dtype={"patient_id": str})
    m = manifest[manifest.cohort == cohort].reset_index(drop=True)
    bags = []
    for r in m.itertuples():
        with h5py.File(feat_root / encoder / cohort / f"{r.patient_id}.h5") as f:
            bags.append(f["features"][:].astype(np.float32))
    return m.patient_id.to_numpy(), m.label.to_numpy(), bags


def run_cv(
    fit_predict, bags, y, ids, n_splits=5, n_repeats=10, seed=123
) -> pd.DataFrame:
    """Out-of-fold predictions for every patient in every repeat."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    rows = []
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        scores = fit_predict(
            [bags[i] for i in tr], y[tr], [bags[i] for i in te], seed=seed + k
        )
        rows += [
            {
                "repeat": k // n_splits,
                "fold": k % n_splits,
                "patient_id": ids[i],
                "label": int(y[i]),
                "score": float(s),
            }
            for i, s in zip(te, scores)
        ]
    return pd.DataFrame(rows)


def _patient_avg(oof):
    return oof.groupby("patient_id").agg(
        label=("label", "first"), score=("score", "mean")
    )


def bootstrap_auc(y, s, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def summarise(oof, name, n_boot=2000, seed=123) -> dict:
    per_repeat = np.array(
        [roc_auc_score(g.label, g.score) for _, g in oof.groupby("repeat")]
    )
    avg = _patient_avg(oof)
    lo, hi = bootstrap_auc(avg.label.to_numpy(), avg.score.to_numpy(), n_boot, seed)
    return {
        "method": name,
        "auc_mean": per_repeat.mean(),
        "auc_sd": per_repeat.std(),
        "auc_patient_avg": roc_auc_score(avg.label, avg.score),
        "ci_low": lo,
        "ci_high": hi,
        "n_patients": len(avg),
        "n_repeats": len(per_repeat),
    }


def paired_bootstrap(oof_a, oof_b, n_boot=2000, seed=123) -> dict:
    """AUC(a) - AUC(b) on the same patients, with a bootstrap CI."""
    a = _patient_avg(oof_a)
    b = _patient_avg(oof_b)["score"].reindex(a.index)
    y, sa, sb = a.label.to_numpy(), a.score.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    d = np.array(d)
    return {
        "auc_diff": roc_auc_score(y, sa) - roc_auc_score(y, sb),
        "ci_low": np.percentile(d, 2.5),
        "ci_high": np.percentile(d, 97.5),
        "frac_boot_le_0": (d <= 0).mean(),
    }


def run(
    method,
    train_cohort="yale",
    encoder="lunit_vits16",
    n_repeats=10,
    seed=0,
    out_root=Path("results/cv"),
):
    from her2_pcr_pathology_fm.models import METHODS

    ids, y, bags = load_bags(train_cohort, encoder)
    oof = run_cv(METHODS[method], bags, y, ids, n_repeats=n_repeats, seed=seed)
    out = out_root / encoder / train_cohort
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / f"{method}_oof.csv", index=False)
    summary = summarise(oof, method, seed=seed) | {
        "encoder": encoder,
        "train_cohort": train_cohort,
        "seed": seed,
    }
    pd.DataFrame([summary]).to_csv(out / f"{method}_summary.csv", index=False)
    print(summary)
    return oof, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--train-cohort", default="yale")
    p.add_argument("--encoder", default="lunit_vits16")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.method, a.train_cohort, a.encoder, a.n_repeats, a.seed)
