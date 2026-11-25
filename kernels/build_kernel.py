"""Assemble the readable driver and worker into a single offline Kaggle script."""

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECIPES = {
    "e8": ("train_two_stage.py", "WORKER", "lora_worker.py"),
    "e17": ("predict_e17.py", "NATIVE_WORKER_SOURCE", "native_embedding.py"),
}


def assemble(recipe):
    driver, variable, worker = RECIPES[recipe]
    source = (ROOT / driver).read_text()
    worker_source = (ROOT / worker).read_text()
    marker = f"{variable} = None  # Filled by build_kernel.py for standalone Kaggle execution."
    if source.count(marker) != 1:
        raise ValueError(f"Missing or ambiguous worker marker in {driver}")
    assembled = source.replace(marker, f"{variable} = {worker_source!r}")
    if recipe == "e17":
        blend_import = "from ensemble import blend_e17, native_log_probabilities"
        if assembled.count(blend_import) != 1:
            raise ValueError("Missing ensemble import")
        assembled = assembled.replace(blend_import, (ROOT / "ensemble.py").read_text())
    ast.parse(worker_source)
    ast.parse(assembled)
    return assembled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=RECIPES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assemble(args.recipe))
