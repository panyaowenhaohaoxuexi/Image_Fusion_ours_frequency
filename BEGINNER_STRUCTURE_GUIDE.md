# Text-FSFuse Current Structure Guide

## Method

Text-FSFuse uses a text-intent-guided frequency-spatial dual-domain fusion route.
The current main route is multi-scale DDA after a single-level frequency fusion.

## Main Pipeline

```python
vis_spa, vis_freq, _ = shared_encoder(data_vis)
ir_spa, ir_freq, _ = shared_encoder(data_ir)

I_deg, I_fus, intent_aux = intent_generator(
    vis_spa, ir_spa, vis_freq, ir_freq
)

fused_freq, freq_aux = frequency_fusion(
    vis_freq, ir_freq, frequency_intent=I_deg
)

spatial_out, spatial_pyramid, spa_aux = spatial_fusion(
    vis_spa, ir_spa, I_fus, return_aux=True, return_pyramid=True
)

freq_pyramid = frequency_pyramid_adapter(
    fused_freq, target_pyramid=spatial_pyramid
)

D_L1, gate_l1 = dda_l1(freq_pyramid["l1"], spatial_pyramid["l1"])
D_L2, gate_l2 = dda_l2(freq_pyramid["l2"], spatial_pyramid["l2"])
D_L3, gate_l3 = dda_l3(freq_pyramid["l3"], spatial_pyramid["l3"])

decoder_skip = 0.5 * (data_vis + data_ir)
fused_image, decoder_feature = fusion_decoder(
    decoder_skip, D_L1, D_L2, D_L3
)
```

## Core Modules

* SharedEncoder: `net/encoder/simple_encoder.py`
* DualDomainTextIntentGenerator: `net/intent.py`
* TGSFF: `net/frequency_fusion/fusion_block.py`
* FrequencyPyramidAdapter: `net/frequency_fusion/pyramid.py`
* TextConditionedSpatialFusion: `net/fusion/text_conditioned_spatial_fusion.py`
* DDA: `net/dda.py`
* FusionDecoder: `net/decoder/simple_decoder.py`

## Important Boundaries

* `DEGRADATION_PROMPTS` generate `I_deg`.
* `FUSION_PROMPTS` generate `I_fus`.
* `I_deg` only scores amplitude and phase frequency tokens.
* `I_deg` does not modulate selected-token interaction.
* `I_deg` does not modulate low-score token bypass.
* `I_deg` does not modulate post-IFFT features.
* `I_fus` only controls the spatial fusion branch.
* Frequency fusion still uses single-level FFT token routing.
* FrequencyPyramidAdapter creates `F_freq_L1/L2/L3` after IFFT by lightweight convolution and downsampling.
* TextConditionedSpatialFusion outputs `F_spa_L1/L2/L3`.
* `dda_l1`, `dda_l2`, and `dda_l3` aggregate corresponding frequency and spatial scales.
* FusionDecoder reconstructs from `decoder_skip`, `D_L1`, `D_L2`, and `D_L3`.
* DDA and FusionDecoder receive no text intent.
* Old v10 checkpoints are structurally incompatible with the new multi-scale DDA route and require retraining.
