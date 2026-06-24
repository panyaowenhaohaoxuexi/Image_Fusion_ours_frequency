# Multi-scale DDA Update Notes

## Current Route

```text
SharedEncoder
-> DualDomainTextIntentGenerator
-> TGSFF: single-level FFT token routing with I_deg only for token scoring
-> FrequencyPyramidAdapter: maps single F_freq to F_freq_L1/L2/L3
-> TextConditionedSpatialFusion: outputs F_spa_L1/L2/L3 with I_fus
-> scale-wise DDA: D_L1/D_L2/D_L3
-> FusionDecoder(decoder_skip, D_L1, D_L2, D_L3)
-> fused image
```

## Text Intent Boundaries

* `DEGRADATION_PROMPTS` generate `I_deg`.
* `FUSION_PROMPTS` generate `I_fus`.
* `I_deg` only participates in frequency token scoring.
* `I_deg` does not modulate selected-token interaction.
* `I_deg` does not modulate low-score token bypass.
* `I_deg` does not modulate post-IFFT features.
* `I_fus` only controls spatial fusion.
* DDA receives no text intent.
* Decoder receives no text intent.

## Scale Contract

* `spatial_pyramid["l1"]` is the highest-resolution spatial feature.
* `spatial_pyramid["l2"]` is the 1/2-resolution spatial feature.
* `spatial_pyramid["l3"]` is the 1/4-resolution spatial feature.
* `freq_pyramid["l1"]`, `freq_pyramid["l2"]`, and `freq_pyramid["l3"]` are interpolated inside FrequencyPyramidAdapter to exactly match the corresponding spatial level.

## Checkpoint Policy

Old v10 checkpoints are structurally incompatible with the new multi-scale DDA
route and require retraining.
