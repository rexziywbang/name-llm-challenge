"""Evaluate the frozen text baseline on prompt-grouped development folds."""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from attribution import LABELS
from attribution.baseline import StyleLR, TfidfLinear, blend_log
from attribution.splits import make_splits


def evaluate(train):
    required = {"id", "prompt_id", "text", "model"}
    if missing := required.difference(train.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if train.id.isna().any() or not train.id.is_unique:
        raise ValueError("Every passage needs a unique id")
    if train.prompt_id.isna().any() or set(train.model) != set(LABELS):
        raise ValueError("Expected complete prompt ids and all 18 competition labels")
    splits = make_splits(train)
    texts = train.text.fillna("").to_numpy(dtype=str)
    targets = train.model.map(dict(zip(LABELS, range(len(LABELS))))).to_numpy()
    groups = train.prompt_id.to_numpy()
    folds = splits.fold.to_numpy()
    predictions = np.full(len(train), -1)
    for fold in sorted(set(folds) - {-1}):
        fit = (folds >= 0) & (folds != fold)
        valid = folds == fold
        lexical = TfidfLinear().fit(texts[fit], targets[fit], groups[fit])
        style = StyleLR().fit(texts[fit], targets[fit])
        probabilities = blend_log(
            [lexical.predict_proba(texts[valid]), style.predict_proba(texts[valid])],
            [1.0, 0.5],
        )
        predictions[valid] = probabilities.argmax(axis=1)
        score = balanced_accuracy_score(targets[valid], predictions[valid])
        print(f"Fold {fold}: {score:.4f} ({valid.sum()} passages)")
    dev = folds >= 0
    score = balanced_accuracy_score(targets[dev], predictions[dev])
    print(f"Grouped development balanced accuracy: {score:.4f}")
    return splits, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", help="Official training CSV")
    parser.add_argument("--splits", help="Optional destination for the fold assignments")
    args = parser.parse_args()
    splits, _ = evaluate(pd.read_csv(args.train_csv, dtype={"id": str}))
    if args.splits:
        splits.to_csv(args.splits, index=False)
