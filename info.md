# Compositionality Shared Task

## Overview

Systems must produce numerical compositionality scores for target expressions in example sentences.

### Subtasks

- **Subtask A (Noun Compounds):** Rank sentences by modifier compositionality and head compositionality separately.
- **Subtask B (Particle Verbs):** Rank sentences by particle verb compositionality (expression-level).

Both subtasks support English and German, spanning historical and present-day data (temporal origin hidden during evaluation).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Compositionality Correlation** | Spearman’s rho between system and gold standard ratings. |
| **Context Variance Correlation** | Spearman’s rho between system and human rating standard deviations per target. |
| **RMSE** | Root mean squared error between system and gold standard ratings. |

Results reported per subtask/language and as mean across all settings.

---

## Dataset

| Language | Noun Compounds | Particle Verbs |
|---|---|---|
| English | 560 targets | 200 targets |
| German | 579 targets | 200 targets |

- ~10 sentences per target (5 per time period; mean ~9).
- 80/20 train/test split by target (binned by compositionality range).
- Trial data: 2 English targets × 5 examples per subtask.


## Data Format

Tab-separated files named `[language]-[task]-[split].tsv` (e.g., `en-nn-train.tsv`).

**Columns:**

| Column | Description |
|---|---|
| `ContextID` | Unique instance ID |
| `Compound` / `ParticleVerb` | Target expression |
| `Mod` / `Base` | Modifier (noun compound) / Base verb (particle verb) |
| `Head` / `Particle` | Head (noun compound) / Particle (particle verb) |
| `ModAvg`, `HeadAvg` / `Avg` | Human judgment means |
| `ModStd`, `HeadStd` / `Std` | Human judgment standard deviations |
| `Context` | Example sentence |

`*Avg` and `*Std` columns excluded from the test split.

---

## Submission Format

Single zip archive containing `[language]-[task]-pred.tsv` files.

- **No header.**
- Each line: `ContextID`, followed by tab-separated predictions (modifier + head for noun compounds; single score for particle verbs).




Noun compound example	Mod	Head
she propelled herself into the firing line in taking the stance she did	0.6	0.7
our flea market will feature some of the most creative individuals	0.0	4.7
Particle verb example	Overall
it’s gonna get very cold so pull up your leggings	4.8
I will probably wind up as a Web Haunter […] blogging for eternity	0.2

