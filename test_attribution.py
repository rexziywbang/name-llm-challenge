"""Small offline checks for the split, end-to-end CPU path and release scripts."""

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attribution import LABELS
from attribution.splits import make_splits
from evaluate import evaluate
from kernels.ensemble import native_log_probabilities, blend_e17
from scipy.special import softmax
from kernels.build_kernel import RECIPES, ROOT, assemble


def synthetic_passages():
    # Every prompt has one answer per author, as in 149 of the 150 official groups.
    return pd.DataFrame(
        [
            (f"{p}-{k}", p, f"Topic {p}: a passage written in style {k}.", label)
            for p in range(30)
            for k, label in enumerate(LABELS)
        ],
        columns=["id", "prompt_id", "text", "model"],
    )


class AttributionTests(unittest.TestCase):
    def test_no_prompt_crosses_a_fold_or_holdout(self):
        splits = make_splits(synthetic_passages())
        self.assertTrue(splits.groupby("prompt_id").fold.nunique().eq(1).all())
        self.assertEqual(set(splits.fold), {-1, 0, 1, 2, 3, 4})
        self.assertTrue(splits.loc[splits.split == "holdout", "fold"].eq(-1).all())

    def test_cpu_pipeline_handles_missing_passage(self):
        rows = synthetic_passages()
        rows.loc[0, "text"] = None
        _, score = evaluate(rows)
        self.assertTrue(np.isfinite(score))

    def test_standalone_workers_match_sources(self):
        for key, (_, name, filename) in RECIPES.items():
            nodes = ast.parse(assemble(key)).body
            (value,) = [
                n.value
                for n in nodes
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name
            ]
            self.assertEqual(ast.literal_eval(value), (ROOT / filename).read_text())

    def test_manifest_permutation_and_five_fold_pooling(self):
        rng = np.random.default_rng(8)
        logits = rng.normal(size=(5, 7, 18))
        order = rng.permutation(18)
        labels = np.array(LABELS)[order]
        scales = np.linspace(0.7, 1.2, 5)
        result, _, _ = native_log_probabilities(logits[:, :, order], labels, LABELS, scales)
        reference = softmax(logits * scales[:, None, None], axis=2).mean(axis=0)
        np.testing.assert_allclose(np.exp(result), reference)

    def test_blend_matches_weighted_geometric_mean(self):
        members = [
            np.log([p]) for p in ([0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.55, 0.45], [0.2, 0.8])
        ]
        _, scores = blend_e17(*members)
        reference = np.prod(
            [np.exp(m) ** w for m, w in zip(members, [1.2 / 7, 1.8 / 7, 0.6 / 7, 0.6 / 7, 0.4])],
            axis=0,
        )
        np.testing.assert_allclose(softmax(scores, axis=1), reference / reference.sum())

    def test_paired_lora_result(self):
        result = json.loads((Path(__file__).parent / "results/e8_folds.json").read_text())
        two_stage, control = result["ts"], result["ctl"]
        self.assertAlmostEqual(
            100 * (two_stage["plain_oof"] - control["plain_oof"]), 5.83, places=2
        )
        self.assertTrue(
            all(two_stage["plain_fold"][f] > v for f, v in control["plain_fold"].items())
        )


if __name__ == "__main__":
    unittest.main()
