# Text-FSFuse Current Structure Guide

## Method

Text-FSFuse is a text-intent-guided frequency-spatial dual-domain selective fusion framework for infrared and visible image fusion.

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

fused_spa, spa_aux = spatial_fusion(
    vis_spa, ir_spa, I_fus, return_aux=True
)

dual_feature, dda_gate = dda(fused_freq, fused_spa)

decoder_skip = 0.5 * (data_vis + data_ir)
fused_image, _ = fusion_decoder(
    decoder_skip, dual_feature, fused_freq
)
```

## Core Modules

* SharedEncoder: `net/encoder/simple_encoder.py`
* DualDomainTextIntentGenerator: `net/intent.py`
* TGSFF: `net/frequency_fusion/fusion_block.py`
* TextConditionedSpatialFusion: `net/fusion/text_conditioned_spatial_fusion.py`
* DDA: `net/dda.py`
* FusionDecoder: `net/decoder/simple_decoder.py`

## Important Boundaries

* DDA does not receive text intent.
* FusionDecoder does not receive text intent.
* Inference does not require user-provided text.
* Fixed prompt banks still participate in the forward process through `DualDomainTextIntentGenerator`.
* `I_deg` is used only by the frequency branch.
* `I_fus` is used only by the spatial branch.
