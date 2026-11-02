"""Methods with a shared signature: fit_predict(train_bags, y_train, test_bags, seed) -> scores."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tile_count(train_bags, y_train, test_bags, seed=123):
    """No training: score = number of tiles (the tissue-amount shortcut)."""
    return np.array([len(b) for b in test_bags], dtype=float)


def meanpool_lr(train_bags, y_train, test_bags, seed=123, n_components=32):
    """Mean-pool tiles -> scale -> PCA -> balanced logistic regression, fit on training folds only."""
    Xtr = np.stack([b.mean(axis=0) for b in train_bags])
    Xte = np.stack([b.mean(axis=0) for b in test_bags])
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(class_weight="balanced", max_iter=5000),
    )
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]


METHODS = {"tile_count": tile_count, "meanpool_lr": meanpool_lr}
