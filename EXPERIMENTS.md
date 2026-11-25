# Experiment record

The project asked two different questions. Could a feature distinguish closely related generators under controlled conditions? Could that feature improve the full 18-class system on unseen text? A positive answer to the first question sometimes produced a negative answer to the second.

## Context as a fingerprint

GLM-5.3 and GLM-5.3-Flash were difficult to distinguish. One hypothesis was that their generated text might differ in how much preceding context helps a fixed, smaller model predict its next words. The scorer used Qwen2.5-0.5B and held each four-token target span fixed while varying the preceding window among 8, 16 and 32 tokens. Paired windows remove changes in target difficulty from the comparison. Up to 12 anchors per passage kept CPU cost bounded. [Scorer](attribution/context_gain.py)

On 240 GLM passages in the prompt-grouped development folds, the gain features produced **0.494 pairwise ROC AUC**. Length controls alone reached 0.571. Adding the gains to a classifier with TF-IDF, style and length changed AUC from 0.552 to 0.540. Within the 120 matched prompt pairs, the gain-only classifier ordered the pair correctly 50.8% of the time. The experiment stopped there. The measurement tested a signal in observed text under one reference scorer. It could not identify the generator’s internal attention mechanism.

## How many answers to the same prompt will inference see?

Frozen representations from four depths of Qwen2.5-0.5B mix topic and style. E4 estimated topic by clustering inference passages with word TF-IDF and average-link cosine distance, then subtracting a cluster mean from standardized hidden features. Known prompt IDs were available during training. The final classifier fell back to its uncentered probe for inferred clusters smaller than three. [Inference code](kernels/predict_centered.py)

The first variant lost **9.3 points** with one answer per prompt. Its selected centering strength could collapse a singleton to zero. The revised control subsampled 1, 2, 3, 6 or 18 answers per prompt and repeated the comparison across five seeds. The final gains were **0.78, 0.20, 2.83, 7.16 and 9.21 points**, respectively. Other prompts can still be falsely grouped even when each true prompt contributes one passage. The leaderboard gain of 2.6 points illustrates the deployment risk without establishing how many siblings were in the hidden test set. [Full ablation](results/e4_sibling_control.md)

## Two-stage LoRA under prompt-grouped evaluation

Public adapters had already trained on official rows, which made their development scores optimistic even if a *new head* held out those rows. E8 compared two training recipes on the same five prompt-grouped folds. The two-stage recipe first trained on 11,928 public passages from 732 in-style prompts, then fine-tuned on each official training fold. Regenerations of official prompts were excluded. The control trained on official data alone.

Both used the first 24 layers of Qwen3-8B-Base, rank-16 LoRA and eight pooled residual-stream depths. Stage one used a class-weighted loss. The training batches grouped answers by prompt, allowing an auxiliary centered head. Only the plain head contributed predictions at inference. The two-stage schedule used one pool epoch followed by two official epochs at a lower learning rate. The control used three official epochs. This comparison measures the complete training recipe. Isolating the effect of extra data would require a schedule-matched ablation. [Training worker](kernels/lora_worker.py)

| Balanced accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official-only | .475 | .461 | .391 | .468 | .470 | .4528 |
| Two-stage | .528 | .493 | .479 | .521 | .535 | .5111 |

The +5.83-point gain passed the advance rule of at least two points and positive results on at least four folds. Deployment caught another error: the inference code initially attached the head’s columns to the wrong class names. Correcting label order raised the standalone public score from .140 to .511. The corrected 50/50 blend scored .581, below the existing .588 reference. The training result stood even though the blend did not advance. [Recorded fold metrics](results/e8_folds.json)

## Submission history

Selected historical comparisons, all measured on the **public leaderboard**. The project has no private-leaderboard result reported here.

| Run | Candidate | Public balanced accuracy | Decision |
| --- | --- | ---: | --- |
| Baseline | Calibrated character/word SVM and style logistic regression | .377 | Starting point |
| E3 / E4 | Frozen hidden-state probe, then gated prompt centering | .404 / .430 | Adopted both |
| E5 / E6 | Public Qwen3-8B adapter, then six-adapter ensemble | .582 / .588 | E6 became the reference |
| E8 | Locally trained two-stage adapter / blend with E6 | .511 / .581 | Retained E6 |
| E11 / E12 | ATLAS likelihood / pool SVC with test standardization | .588 / .581 | Retained E6 |
| E13 / E14 | Qwen3-14B member / additional two-stage 14B member | .630 / .631 | E14 missed its .635 bar |
| E15 / E16 | Qwen2.5-14B member, then DeBERTa-v3-large | .639 / .644 | Adopted |
| E17 | Native Qwen3-Embedding-8B expert blended into E16 | **.663** | Adopted; submission 58185113 |
| E18 | Gave WeMM-Embedding-4B 10% of the full blend | .661 | Below the .673 adoption bar |
| E19 | Conditional GLM likelihood member | — | Stopped after replication check |

**Why E17 helped.** The added expert uses last-token features from eight depths and covers long passages with weighted windows. Its five calibrated fold distributions are averaged in probability space after aligning class names. The final blend combines the E16 and native experts in log space with fixed weights of 0.6 and 0.4. This gave the ensemble a different readout. The comparison establishes a gain for the whole addition and does not isolate which design choice caused it. [Native worker](kernels/native_embedding.py)

The 4,000-row E17 dry run took 3.38 hours and peaked at 13.07 GB of allocated GPU memory. For each labeled row, comparing the native expert’s prediction with the upstream reference **for its held-out fold** gave 3,960/3,994 top-1 matches. Its held-out balanced accuracy was .6483 on 2,081 unique rows. Several older members saw official validation rows during their original training, so the full stack has no clean out-of-fold estimate. The .663 figure comes from Kaggle.

**Why E18 stopped.** The two embedding experts agreed on only 71.8% of out-of-fold predictions. Diversification looked promising. Allocating a quarter of the native share to WeMM lifted the native-only development blend from .6516 to .6609. Those same rows informed the choice of weight. The complete system scored .661 publicly against E17’s .663. The experiment did not clear the threshold for replacing E17.

## When likelihood replication and top-1 agreement diverge

E19 spent additional inference compute on passages where E17 assigned at least 10% combined probability to GLM-5.3 and Flash. It scored up to 160 tokens under both candidate generators. A conditional expert had raised the held-out native component’s balanced accuracy by 4.3 points in exploratory analysis. The routing threshold and weight were chosen using those rows, and the deployed ensemble was stronger than that component.

The first full check crashed while reading row IDs. A separate 64-passage test then found a more subtle failure. One passage’s middle tokens received implausibly low likelihood. Its presence pulled GLM-5.3’s mean log-probability correlation with the reference down to .827. The same batch reproduced the error bit for bit. Reversing row order or scoring one passage at a time removed it. This isolated a batch-position dependency that a shape check or a finite-value check would have missed.

| Final check on 398 unique official rows | Observed | Required |
| --- | ---: | ---: |
| Flash mean log-probability correlation | .997 | ≥ .95 |
| GLM-5.3 mean log-probability correlation | .995 | ≥ .95 |
| 18-way member top-1 agreement with reference | **.799** | **≥ .85** |

The second full dry run used one passage per batch and took 5.2 hours. Top-1 agreement rose to .947 on rows where the reference margin exceeded .25, and to 1.000 above .5. On the 245 GLM-labeled rows, the binary decision reached .898 accuracy versus the reference’s .894. Close 18-way races made argmax sensitive to small score changes. This explains the failed gate retrospectively. E19 was closed after the second dry run.
