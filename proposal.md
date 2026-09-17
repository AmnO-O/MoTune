# Proposal: stable multi-exit Gaussian architecture

## Objective

Predict lexical compositionality with Gaussian outputs while keeping each
prediction tied to the transformer representation that should contain the
relevant information.  The principal goal is a reproducible NN baseline; PV
is an additional task, not a source of accidental changes to the NN objective.

## Recommended model

Use one shared encoder and three independently supervised Gaussian exits.

| Exit | Transformer states | Input features | Supervision |
| --- | --- | --- | --- |
| Modifier | block 18 | modifier span, counterpart span, pooled context | `ModAvg`, `ModStd` |
| Head | block 19 | head span, counterpart span, pooled context | `HeadAvg`, `HeadStd` |
| Particle-verb (PV) | blocks 20--21 | base/verb span, particle span, pooled two-block context | overall `Avg`, `Std` |

The PV exit is intentionally an **overall** Gaussian head.  The PV data has
only overall labels, so fabricating modifier/head targets for it would add
noise.  Its input should retain the two role-specific spans (base verb and
particle) so the head can learn their interaction while predicting one overall
distribution.

Each exit has its own small MLP followed by `GaussHead(mu, sigma)`.  Do not
share final prediction layers: their targets and representation depths differ.
The encoder remains shared, so gradients from the three exits improve the
common language representation.

## Loss

For every labelled example, optimize the same well-defined objective:

```
L_role = Gaussian_KL(mu, sigma, target_mean, target_std)
       + ccc_weight * (1 - CCC(mu, target_mean))
       + lambda_rank * ranking_loss(mu, target_mean, compound_id)
```

Gaussian KL is always enabled: otherwise the predicted standard deviation has
no direct learning signal.  `ccc_weight` and `lambda_rank` remain the only
optional weighting controls.  There is no compound-centre calibration loss and
no global distribution-loss multiplier.  Apply each term only where that
role's labels exist; PV uses its overall labels and its own compound groups.

When a batch mixes NN and PV, sum the valid role losses and normalise by the
number of contributing labelled roles.  Log the three losses separately so a
PV regression cannot hide behind a good NN aggregate.

## Training policy

1. Establish a fixed NN-only baseline (`TRAIN_MIX=nn`) before enabling PV.
2. Train the three heads during the initial frozen-encoder stage, then unfreeze
   the selected LoRA modules for the second stage.  Use a short warm-up and a
   single documented learning-rate schedule.
3. Add PV only after the NN baseline is reproduced with the same split, seed,
   checkpoint-selection rule, and inference code.
4. If PV is enabled, start with a modest sampling rate and report NN and PV
   validation metrics separately.  Do not select an NN trial checkpoint using
   a mixed metric unless the deployment target is deliberately mixed.

## Evaluation policy

The trial notebook should always write predictions and calculate rho and MSE
at the end, separately for modifier, head, and PV overall predictions.  Compare
`src/` and `mm/` only when all of the following are identical:

- data split and preprocessing;
- task mixture and per-task sampling;
- checkpoint-selection metric and epoch;
- seed and ensemble/fold count;
- inference transformation and clipping;
- Gaussian loss settings.

For the current reported NN trial, the fair first comparison is an NN-only,
single-model `src/` run against an NN-only, single-model `mm/` run.  Compare
ensembles only after the single-model results agree.

## Ablation order

Run these in order; change one item at a time.

1. Existing NN-only model with the simplified loss.
2. Add the layer-18 modifier and layer-19 head exits.
3. Add the overall PV exit at layers 20--21, but evaluate it separately.
4. Tune `ccc_weight` and `lambda_rank` on validation data only.
5. Consider NN+PV joint training only if it improves the NN validation metric
   across more than one seed.

## Guardrails

- Assert that every active exit receives finite gradients in a smoke test.
- Keep role masks explicit; never route PV labels through NN component heads.
- Save the full resolved configuration beside every checkpoint.
- Record per-role rho, MSE, predicted mean, and predicted standard-deviation
  summaries.  A near-constant prediction often explains a negative rho more
  quickly than a single aggregate score.
