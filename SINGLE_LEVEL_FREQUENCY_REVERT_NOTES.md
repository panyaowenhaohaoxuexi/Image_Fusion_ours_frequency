# Single-level frequency branch revert notes

本版本按当前要求将频率域从三层多尺度改回单层。

## 当前结构

Encoder 返回：

```text
spatial_feats = [spa_l1, spa_l2, spa_l3]
freq_feat = freq_l1
shared_feat
```

空间域仍保持三层 Text-conditioned CSAF / PSAF：

```text
[spa_l1, spa_l2, spa_l3] + spatial_intent -> F_spa
```

频率域改回单层：

```text
freq_feat -> FFT -> amplitude / phase token scoring -> Top-K routing
-> selected strong interaction + unselected lightweight preservation
-> amplitude / phase reconstruction -> iFFT -> F_freq
```

## 重要边界

1. 不再返回 `fused_freq_levels`。
2. 不再使用 `frequency_intent_l1/l2/l3`。
3. 不再做 top-down frequency aggregation。
4. `F_freq` 不输入空间分支。
5. 频率域和空间域仍然由同一个 Text Intent Embedding 控制，但频率域为单层，空间域为三层。

## 调用链

```python
vis_spa, vis_freq, _ = encoder(data_vis)
ir_spa, ir_freq, _ = encoder(data_ir)
fused_freq, freq_aux = frequency_fusion(vis_freq, ir_freq)
spatial_intent = freq_aux['spatial_intent']
fused_spa = spatial_fusion(vis_spa, ir_spa, spatial_intent)
fused_image, _ = decoder(decoder_skip, fused_spa, fused_freq, text_intent=spatial_intent)
```
