"""Methods with a shared signature: fit_predict(train_bags, y_train, test_bags, seed) -> scores."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from her2_pcr_pathology_fm.utils import get_device


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


@torch.no_grad()
def evaluate_abmil(model, bags, labels, criterion, device):
    model.eval()
    losses, probs = [], []
    for bag, label in zip(bags, labels):
        bag_t = torch.from_numpy(bag).to(device)
        label_t = torch.tensor([float(label)], device=device)
        logit, _ = model(bag_t)
        loss = criterion(logit, label_t).item()
        losses.append(loss)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
    mean_loss = np.mean(losses)
    auc = roc_auc_score(labels, probs)
    return mean_loss, auc, probs


@torch.no_grad()
def predict_abmil(model, bags, device):
    model.eval()
    probs = []
    for bag in bags:
        bag_t = torch.from_numpy(bag).to(device)
        logit, _ = model(bag_t)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
    probs = np.array(probs)
    return probs


def train_abmil(
    train_bags,
    train_y,
    val_bags=None,
    val_y=None,
    seed=1337,
    n_epochs=20,
    lr=1e-4,
    weight_decay=0.01,
    dropout=0.25,
    device="cpu",
    track_train=True,
):
    train_y = np.asarray(train_y)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ABMIL(dropout=dropout).to(device)
    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(n_epochs):
        model.train()
        for i in rng.permutation(len(train_bags)):
            bag, label = train_bags[i], train_y[i]
            # Convert bags, labels to tensors
            bag_t = torch.from_numpy(bag).to(device)
            label_t = torch.tensor([float(label)], device=device)

            # forward pass
            logit, _ = model(bag_t)
            # calculate loss
            loss = criterion(logit, label_t)

            # backward pass
            # clearing gradients from previous step
            optimizer.zero_grad(set_to_none=True)
            # calculate gradients for all params
            loss.backward()
            # use gradients to update parameters to minimize loss
            optimizer.step()

        # Evaluate at the end of each epoch
        row = {"epoch": epoch}
        if track_train:
            row["mean_train_loss"], row["train_auc"], _ = evaluate_abmil(
                model, train_bags, train_y, criterion, device
            )
        if val_bags is not None:
            row["mean_val_loss"], row["val_auc"], _ = evaluate_abmil(
                model, val_bags, val_y, criterion, device
            )
        history.append(row)
    return model, history


def abmil_fit_predict(train_bags, y_train, test_bags, seed=1337, device=None):
    if device is None:
        device = get_device()
    model, _ = train_abmil(
        train_bags,
        y_train,
        seed=seed,
        n_epochs=20,
        lr=1e-4,
        weight_decay=0.01,
        dropout=0.25,
        device=device,
        track_train=False,
    )
    probs = predict_abmil(model, test_bags, device=device)
    return probs


METHODS = {
    "tile_count": tile_count,
    "meanpool_lr": meanpool_lr,
    "abmil": abmil_fit_predict,
}
