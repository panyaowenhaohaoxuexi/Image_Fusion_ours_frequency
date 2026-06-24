# Repository Cleanup Notes

## Current Main Pipeline

Text-FSFuse now uses the current multi-scale DDA route:

```text
VIS / IR
-> SharedEncoder
-> DualDomainTextIntentGenerator generates I_deg / I_fus
-> TGSFF uses I_deg only for single-level frequency token scoring
-> FrequencyPyramidAdapter maps F_freq to F_freq_L1/L2/L3 after IFFT
-> TextConditionedSpatialFusion uses I_fus and outputs F_spa_L1/L2/L3
-> dda_l1 / dda_l2 / dda_l3 aggregate matched frequency and spatial scales
-> FusionDecoder reconstructs from decoder_skip and D_L1/D_L2/D_L3
-> fused image
```

## Route Boundaries

* `I_deg` only scores frequency tokens.
* `I_deg` does not modulate selected-token interaction.
* `I_deg` does not modulate low-score token bypass.
* `I_deg` does not modulate post-IFFT features.
* `I_fus` only controls spatial fusion.
* DDA receives no text intent.
* FusionDecoder receives no text intent.
* FusionDecoder receives no separate `fused_freq`.
* Frequency fusion remains single-level FFT token routing.
* The frequency pyramid is generated after IFFT by lightweight convolution and downsampling.

## Removed Current-Route Elements

* The old single aggregation training/testing route has been removed.
* The old decoder call that combined one domain feature with a separate frequency feature has been removed.
* The old single aggregation checkpoint key has been removed.
* Old v10 checkpoint compatibility has not been retained.

## Safety Decisions

* No loss formula or loss weight is changed.
* No epoch count, batch size, learning rate, optimizer type, scheduler setting, prompt text, or CLIP default setting is changed.
* Old v10 checkpoints are structurally incompatible with the new multi-scale DDA route and require retraining.
