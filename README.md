# Which LLM wrote this passage?

Give 18 language models the same prompt, hide their names, and read one answer. Can you tell which model wrote it? In Kaggle’s *Name That LLM*, the classifier has to make that call on **prompts it hasn’t seen during training**. My baseline used word fragments, punctuation and other style cues. It scored **0.377** on the public leaderboard. The best ensemble documented here reached **0.663**. Both scores are balanced accuracy, the average recall across the 18 generators.

**Where this work started.** I created `namethatllm`, the main research repo, on **October 27, 2026**. I’m still competing in *Name That LLM*, so I’m keeping that repo private. This public repo shares selected findings, the code behind several experiments, and a few debugging stories worth showing. [Kaggle competition](https://www.kaggle.com/competitions/name-that-llm)

## Three things that surprised me

- **Who else answered the prompt mattered.** I tried subtracting the average hidden representation of answers to the same inferred prompt. The idea was to remove what everyone was talking about and leave more of *how* each model said it. With all 18 answers available, centering improved held-out balanced accuracy by **9.21 percentage points**. When I kept just two answers per prompt, the gain fell to **0.20 points**. The classifier’s apparent edge depended on which other passages it could see at prediction time. [Sibling-count ablation](results/e4_sibling_control.md)
- **The next 10,792 training passages bought less than a point.** A two-stage LoRA recipe trained first on 11,928 external passages, then on official data. It beat the official-only recipe on all five held-out folds. I kept the two-stage recipe fixed and widened the pool to 22,720 passages, relaxing the rule for how many generators had to answer each prompt. Balanced accuracy went from **0.5111 to 0.5199**. Four folds improved, but the run missed the one-point gain I’d set as the bar for a submission. [Training-pool experiment](EXPERIMENTS.md#what-did-the-next-10792-passages-buy)
- **One passage got a different likelihood when I moved it in the batch.** The scorer reproduced the same bad token scores on a rerun. Reversing the order made the anomaly disappear. Scoring one passage at a time brought both GLM likelihood correlations with the reference above **0.99**, yet the 18-way classifier still agreed with the reference’s winning label on only **79.9%** of rows. Looking at the gap between first and second place explained why small score changes mattered. [Likelihood debugging notes](EXPERIMENTS.md#when-likelihood-replication-and-top-1-agreement-diverge)

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

I, Aarush Das (`rexziywbang`), maintain this research portfolio. Development used Devin for most of the experiment execution and debugging, while research taste and implementation were guided/motivated by me. The work evaluates published techniques alongside project-specific hypotheses, ablations and replication checks. CHEN Han (lightningv08) developed the native embedding expert used in the best ensemble. The LoRA training work builds on the recipes of lucasboesen, lightningv08 and nishantdahal. [Sources and component credits](kernels/README.md#sources)
