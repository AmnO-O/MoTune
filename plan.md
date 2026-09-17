# Architecture plan: compositionality regression with calibrated uncertainty

## Decision

Use a shared mmBERT encoder with three **dedicated** Gaussian exits.  Keep the
modifier, head, and particle-verb objectives separate until they have earned a
joint-training benefit on held-out NN data.  This is the smallest architecture
that matches the labels and avoids the previous generic/fallback paths.

```text
mmBERT encoder
  block 18 ── modifier span + counterpart + sentence context ── GaussHead ── ModAvg, ModStd
  block 19 ── head span     + counterpart + sentence context ── GaussHead ── HeadAvg, HeadStd
  blocks 20–21 ── base span + particle span + sentence context ── GaussHead ── PV Avg, PV Std
```

Every exit consumes two attention-pooled spans, their relative lengths, and a
mean+CLS sentence representation from its own chosen layer.  The PV head is a
single overall-composition head: it must not be trained through the NN
modifier/head labels.

## Why this design

- Intermediate supervision is a sensible way to make a deep network’s
  representations useful at specific depths, but it should be explicit rather
  than hidden behind alternative feature paths. [Deep supervision](https://arxiv.org/abs/1505.02496)
  introduced auxiliary branches precisely for this purpose.
- Layer selection should be an ablation, not an implicit “mid-layer soup.”
  Lexical semantic information is distributed across layers, and lower layers
  tend to retain more type-level lexical information; that supports testing
  dedicated lexical exits rather than averaging arbitrary layers. [Vulić et
  al., 2020](https://aclanthology.org/2020.emnlp-main.586/)
- Predicting a mean and an input-dependent uncertainty is a heteroscedastic
  regression problem. Gaussian likelihood objectives are standard, but can be
  difficult to optimize and calibrate, so uncertainty needs its own evaluation
  rather than being treated as a free extra output. [Stirn et
  al., 2023](https://proceedings.mlr.press/v206/stirn23a.html)
- NN and PV may improve each other, but multi-task losses can conflict. Start
  with separate metrics and only add automatic task balancing if a fixed
  weighting demonstrably fails. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
  is the preferred next experiment, because it balances gradients rather than
  adding more hand-tuned loss multipliers.

## Loss policy

For a labelled role, retain one compact objective:

```text
role loss = Gaussian distribution loss + ccc_weight × (1 − CCC)
          + lambda_rank × within-compound ranking loss
```

The distribution term is never optional: it is the only direct supervision for
the Gaussian scale output.  The rank term is computed only inside a compound.
The current implementation has no compound-centre loss, no global loss
multiplier, and no unlabelled auxiliary rows that consume compute without
contributing gradients.

Before changing the loss formula, run a controlled ablation of the current
Gaussian KL direction against Gaussian NLL (equivalently, the target-to-model
KL up to constants). Select the winner by **mean rho first**, then MSE and
calibration; do not change this together with the architecture.

## Training sequence

1. Reproduce an EN-NN-only baseline with fixed split, seed, preprocessing,
   checkpoint selection, and one model.
2. Compare exits `(18, 19, 20–21)` against a last-layer-only baseline. Keep
   the dedicated layout only if it improves validation rho across at least
   three seeds.
3. Add PV while selecting checkpoints exclusively by NN validation rho. Report
   PV rho separately; never let a strong PV score conceal NN regression.
4. If NN+PV is worse than NN-only, inspect per-task gradient norms at the last
   shared layer. Only then try GradNorm or reduce PV sampling; do not add
   arbitrary new loss coefficients.
5. Train several seeds, then ensemble only models with the same architecture
   and preprocessing. Compare single models before comparing ensembles.

## Required evaluation

For every validation and trial run, save:

- rho and MSE for modifier, head, and PV overall;
- Gaussian NLL or the selected distribution loss;
- mean predicted sigma, target standard deviation, and a calibration check;
- resolved config, seed, checkpoint epoch, and data lineage.

A negative rho together with nearly constant predictions is an optimization or
checkpoint-selection failure, not evidence that a more complex head is needed.

## Deliberately excluded

- shared final prediction heads across roles;
- fake component labels for PV;
- label-free auxiliary rows without an explicit auxiliary objective;
- generic mean-pooling / context-pooling fallback modes;
- legacy single-file dataset fallbacks;
- automatic ensembling before the one-model baseline is reliable.
