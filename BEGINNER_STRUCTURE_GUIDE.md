# 当前主干结构速览

当前工程已调整为“高层文本意图引导的频率—空间双域选择性融合框架”。

1. `SharedEncoder`
   - 文件：`net/encoder/simple_encoder.py`
   - 功能：从可见光/红外图像中分别提取空间域特征和频率候选特征。
   - 输出：`spa_feat, freq_feat, shared_feat`。

2. `HighLevelGuidedFrequencyFusion`
   - 文件：`net/frequency_fusion/fusion_block.py`
   - 功能：`FFT -> amplitude/phase tokens -> Text Intent 条件评分 -> Top-K routing -> selected 强交互 -> unselected 轻量保真旁路 -> 频谱重组 -> iFFT`。
   - 输出：`F_freq` 与 `freq_aux`。
   - `freq_aux` 中包含：`intent / frequency_intent / spatial_intent / prompt_weight / amp_score / phase_score / amp_mask / phase_mask`。

3. `TextConditionedSpatialFusion`
   - 文件：`net/fusion/text_conditioned_spatial_fusion.py`
   - 功能：空间域并行融合分支。
   - 流程：`vis_spa / ir_spa -> cross-modal spatial interaction -> Text Intent-conditioned CSAF / PSAF -> F_spa`。
   - 条件输入：复用频率分支中同一个 IntentRouter 得到的 `spatial_intent`。

4. `FusionDecoder`
   - 文件：`net/decoder/simple_decoder.py`
   - 功能：接收 `F_spa` 与 `F_freq`，结合 image-level skip 重建融合图像。

当前 forward 主链路：

```python
vis_spa, vis_freq, _ = shared_encoder(data_vis)
ir_spa,  ir_freq,  _ = shared_encoder(data_ir)

fused_freq, freq_aux = frequency_fusion(vis_freq, ir_freq)
spatial_intent = freq_aux['spatial_intent']
fused_spa = spatial_fusion(vis_spa, ir_spa, spatial_intent, fused_freq)

decoder_skip = 0.5 * (data_vis + data_ir)
fused_image, _ = fusion_decoder(decoder_skip, fused_spa, fused_freq)
```

其中，`frequency_intent` 只服务于频率域 amplitude / phase token scorer；`spatial_intent` 只服务于空间域文本条件自适应融合。二者来自同一个 Text Intent Embedding。
