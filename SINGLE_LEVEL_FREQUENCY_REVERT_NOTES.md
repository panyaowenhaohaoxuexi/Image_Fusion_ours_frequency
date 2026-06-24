# Single-level Frequency Branch Notes

This note records the current route after keeping the frequency branch single-level
while upgrading the domain aggregation and decoder to multi-scale features.

## Current Conclusions

* The frequency branch performs one single-level FFT token-routing pass.
* `I_deg` only participates in amplitude and phase token scoring.
* High-score frequency tokens use strong cross-modal selected-token interaction.
* Low-score frequency tokens use lightweight modality-adaptive preservation.
* `I_deg` does not modulate selected-token interaction, bypass tokens, or post-IFFT features.
* FrequencyPyramidAdapter maps the single `F_freq` into `F_freq_L1/L2/L3` after IFFT.
* The spatial branch uses `I_fus` and outputs `F_spa_L1/L2/L3`.
* `dda_l1`, `dda_l2`, and `dda_l3` aggregate matched frequency/spatial scales.
* FusionDecoder receives `decoder_skip`, `D_L1`, `D_L2`, and `D_L3`.
* DDA and FusionDecoder receive no text intent.

## Current Call Chain

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

## Checkpoint Policy

Old v10 checkpoints are structurally incompatible with the new multi-scale DDA
route and require retraining. The old single aggregation checkpoint key has been
removed from the current train/test checkpoint path.
