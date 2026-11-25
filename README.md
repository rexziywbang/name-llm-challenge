# Which LLM wrote this passage?

A model can answer a question hundreds of other models know how to answer. Can its choice of words, punctuation and phrasing identify it on a **new prompt**? This project studies generator attribution across 18 models in Kaggle’s *Name That LLM*. The text-only baseline scored **0.377** public leaderboard balanced accuracy. The strongest submitted ensemble scored **0.663**. The metric gives each model’s recall equal weight.

The most informative experiment was a control. Removing the average representation of passages that answered the same prompt improved validation accuracy by **9.21 percentage points** when 18 answers traveled together. With only two answers available, the gain fell to **0.20 points**. The public leaderboard improved by **2.6 points**. How passages are batched at evaluation time can change the apparent value of a method. [Sibling-count ablation](results/e4_sibling_control.md)

## What to read

| Question | Implementation or evidence |
| --- | --- |
| How much signal survives a change of prompt? | [CPU evaluation](evaluate.py) and [lexical and style models](attribution/baseline.py) |
| Can a small reference LM recognize another generator from the value of extra context? | [Context-ablation scorer](attribution/context_gain.py) and [the negative result](EXPERIMENTS.md#context-as-a-fingerprint) |
| Does training on additional in-style passages help on unseen official prompts? | [Two-stage LoRA worker](kernels/lora_worker.py) and [paired fold results](results/e8_folds.json) |
| Which components earned a place in the final ensemble? | [Inference driver](kernels/predict_e17.py), [native embedding expert](kernels/native_embedding.py) and [submission history](EXPERIMENTS.md#submission-history) |
| How can a likelihood scorer reproduce the signal yet disagree on top-1 predictions? | [GLM replication study](EXPERIMENTS.md#when-likelihood-replication-and-top-1-agreement-diverge) |

## Run the CPU evaluation

From the repository root with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
```

The tests exercise all 18 classes on synthetic passages without downloading weights or competition data. To evaluate the baseline, download the official training CSV through Kaggle and run:

```bash
python evaluate.py data/train.csv --splits data/official_splits.csv
```

The command fits the text vectorizers, SVM temperature and style classifier within each training fold. It prints five fold scores and pooled out-of-fold balanced accuracy for the fixed blend. The historical audit found 150 prompts across 2,699 passages and zero cross-prompt near-duplicate pairs at the recorded eight-word shingle threshold. The split reserved 539 rows for holdout and grouped the remaining 2,160 by prompt. The historical development score was 0.3625. Package versions can affect splits or fitted coefficients. [Competition and data](https://www.kaggle.com/competitions/name-that-llm)

## Run the GPU code

The [GPU guide](kernels/README.md) lists the input assets, how to build single-file offline scripts, and the extra data preparation needed for the LoRA comparison. Model weights, official passages and generated intermediate features are obtained separately. The scores above come from the historical competition runs. This public edition includes the training and inference code, aggregate results and offline regression checks.

## Project and upstream credits

Aarush Das (`rexziywbang`) maintains this research portfolio. Development used Devin for implementation, experiment execution and debugging, and Codex for the public release. The work evaluates published techniques alongside project-specific hypotheses, ablations and replication checks. CHEN Han (lightningv08) developed the native embedding expert used in the best ensemble. The LoRA training work builds on the recipes of lucasboesen, lightningv08 and nishantdahal. [Sources and component credits](kernels/README.md#sources)
