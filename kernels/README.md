# GPU experiments

These are offline Kaggle workloads. Start with the CPU evaluation in the root README for a quick local run. Each driver here runs as a script. The build step embeds the worker so that a single Kaggle script can launch a second process with its own GPU allocation.

## E17 · reproduce the submitted ensemble

1. Run `python kernels/build_kernel.py e17 build/predict_e17.py` from the repository root.
2. Create a Python script notebook in the competition and use the generated script as its source. Attach the competition data and the exact three datasets and four base model versions in [the input manifest](e17_assets.json). The DeBERTa weights and readout are in the attached dataset bundles. Select two T4 GPUs, disable internet, and run the script in the Kaggle environment with PyTorch, Transformers, safetensors, scikit-learn and a Parquet engine.
3. Inspect `/kaggle/working/submission.parquet`. The driver checks that row IDs retain input order, every ID appears once, and predicted labels belong to the competition vocabulary. It writes `e17_probs.npz` beside the output for inspecting member disagreements. Running the notebook and submitting its output for scoring are separate actions.

The native worker runs first and releases its GPU memory before the other models load. It reuses one backbone across five fold states. The driver loads the later models sequentially, remaps head labels by name, and uses the fixed blend in [ensemble.py](ensemble.py). E17’s historical 0.663 was measured on the hidden public split. The small interactive test file only exercises inference.

## E8 · train a controlled LoRA comparison

Run `python kernels/build_kernel.py e8 build/train_two_stage.py`. Attach the competition, the Qwen3-8B-Base model from the E17 manifest and an input folder containing the following two files. The public release supplies the training driver and fold results. Reconstructing the historical pool requires preparing these data separately.

| Input | Contents |
| --- | --- |
| `official_splits.csv` | Output of the root evaluation command with `--splits`. The trainer joins on official passage ID. |
| `pool_instyle.parquet` | One row per external passage, with `text`, `model` and `pid` (a stable identifier for the shared external prompt). |

The original pool combined the six sources below, removed duplicate passages and overlap with official text, and used prompts covered by at least 12 distinct competition labels. Excluding Prvsiyan’s `train_regen` split removed regenerations of official prompts. The resulting pool contained 11,928 passages from 732 prompts. Check both counts and prompt provenance before interpreting a reconstructed run.

On two T4s, GPU 0 trains the shared pool initialization, the five two-stage folds and the final adapter. GPU 1 trains the five official-only controls. Each worker writes per-job completion records and validation features in `lora_out/`. Use a fresh working directory when changing inputs or hyperparameters. The trainer resumes completed jobs in the same directory. The historical jobs used PyTorch, Transformers, PEFT and safetensors.

## E4 · stress-test prompt centering

Use [predict_centered.py](predict_centered.py) as a standalone Kaggle script. Attach the competition and `qwen-lm/qwen2.5/transformers/0.5b/1`. It trains the lexical, style and hidden-state heads on the official CSV and emits a submission. To supply local paths set `NTL_INPUT_DIR`, `NTL_MODEL_DIR` and `NTL_OUTPUT_PATH`. The historical sibling-count ablation is [recorded here](../results/e4_sibling_control.md).

## Sources

- Competition and data by Yao Yan, Martyna Plomecka and Ashley Oldacre. [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- Qwen backbones and embeddings by the Qwen team. The q8v5/q14v5 adapter bundles come from the public Name That LLM community. [Adapters by dhruvtibarewal](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) · [Handwriting blend by nishantdahal](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend)
- The native embedding expert and its weights come from CHEN Han (lightningv08). The worker adapts the upstream Apache-2.0 implementation for offline inference. This edition separates the worker from the generated single-file script and reformats it. [Inference assets](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) · [Apache-2.0 terms](Apache-2.0.txt)
- The LoRA24 training recipe builds on lucasboesen and lightningv08. The two-stage training idea comes from nishantdahal and the high-coverage prompt filter from zbigniewkirchner. [LoRA training](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) · [Five-fold training](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training) · [Two-stage training](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- The likelihood comparison used reference features from nakagawaren0805 and a readout from estopadilla. The project’s contribution here was testing conditional routing, investigating batch-position dependence and applying the replication criterion. [GLM reference features](https://www.kaggle.com/datasets/nakagawaren0805/nllm-glm-logits-train)

### External training pool

| Contributor | Dataset |
| --- | --- |
| seyitkaangunes | [Extra API passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) |
| zbigniewkirchner | [Extra GPT passages](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-extra-gpt-passages) |
| lucasboesen | [Free proxy generations](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations) |
| abdullah1afify | [NTL generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations) |
| dophuchung | [Proxy generations](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations) |
| prvsiyan | [Extended Name That LLM dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) |
