"""Frozen prompt-grouped holdout (~20% of prompt groups) + grouped K-fold on the development set."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import LABELS, SEED


def make_splits(train: pd.DataFrame, holdout_frac=0.2, n_folds=5, seed=SEED) -> pd.DataFrame:
    """Returns DataFrame[id, prompt_id, split('holdout'|'dev'), fold(-1 for holdout)].
    Groups are prompt_id; duplicates/near-duplicates across prompts are merged by caller via `group` column if present."""
    groups = train["group"] if "group" in train else train["prompt_id"]
    y = train["model"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
    n_hold_splits = int(round(1 / holdout_frac))
    sgk = StratifiedGroupKFold(n_splits=n_hold_splits, shuffle=True, random_state=seed)
    dev_idx, hold_idx = next(sgk.split(train, y, groups))
    split = np.array(["dev"] * len(train), dtype=object)
    split[hold_idx] = "holdout"
    fold = np.full(len(train), -1)
    sgk2 = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
    for k, (_, va) in enumerate(sgk2.split(train.iloc[dev_idx], y[dev_idx], groups.iloc[dev_idx])):
        fold[dev_idx[va]] = k
    out = pd.DataFrame(
        {"id": train["id"], "prompt_id": train["prompt_id"], "split": split, "fold": fold}
    )
    assert set(train.loc[split == "holdout", "model"]) == set(LABELS), (
        "holdout lacks class coverage"
    )
    for k in range(n_folds):
        assert set(train.loc[fold == k, "model"]) == set(LABELS), f"fold {k} lacks class coverage"
    return out
