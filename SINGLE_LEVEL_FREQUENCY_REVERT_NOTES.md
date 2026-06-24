# Single-level Frequency Branch Notes

This note records the current v10 route after reverting the frequency branch to a single-level design.

## Current Conclusions

* frequency branch: single-level
* spatial branch: three-level
* frequency branch uses `I_deg`
* spatial branch uses `I_fus`
* DDA aggregates `F_freq` and `F_spa`
* decoder receives image skip, DDA feature, and `fused_freq`
* decoder receives no text intent

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

fused_spa, spa_aux = spatial_fusion(
    vis_spa, ir_spa, I_fus, return_aux=True
)

dual_feature, dda_gate = dda(fused_freq, fused_spa)

decoder_skip = 0.5 * (data_vis + data_ir)
fused_image, _ = fusion_decoder(
    decoder_skip, dual_feature, fused_freq
)
```

## Boundaries

* The frequency branch does not return multi-level frequency outputs.
* The spatial branch does not receive `F_freq`.
* DDA does not receive text intent.
* FusionDecoder does not receive text intent.
