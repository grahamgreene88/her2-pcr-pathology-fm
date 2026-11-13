"""Methods with a shared signature: fit_predict(train_bags, y_train, test_bags, seed) -> scores."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch


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


class ABMIL(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=128, attn_dim=64, dropout=0.25):
        super().__init__()
        # projection layer to reduce embedding dimension from 384 to 128
        self.proj_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)  # (384, 128)
        )
        self.attn_branch_u = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.Sigmoid()
        )
        self.attn_branch_v = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.linear_head = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        h = self.proj_layer(X)
        sig_side = self.attn_branch_u(h)
        tan_side = self.attn_branch_v(h)
        a_scores = self.attn_w(sig_side * tan_side)
        a = F.softmax(a_scores, dim=0)
        # slide/patient embedding
        z = torch.sum(a * h, dim=0)
        logits = self.linear_head(z)
        return logits, a


METHODS = {"tile_count": tile_count, "meanpool_lr": meanpool_lr}
