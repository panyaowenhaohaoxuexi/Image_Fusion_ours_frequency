# Three-level SPGFusion/Text-IF migration notes

本版本将空间分支从单尺度改成三层多尺度结构，目标是更充分迁移 SPGFusion / Text-IF 的有效模块，同时保留当前论文主线。

## 主链路

```text
visible / infrared
→ SharedEncoder
→ spatial_feats=[L1,L2,L3], freq_feat
→ HighLevelGuidedFrequencyFusion(freq_feat)
→ F_freq + spatial_intent
→ TextConditionedSpatialFusion(spatial_feats, spatial_intent, F_freq)
→ F_spa
→ FusionDecoder(F_spa, F_freq, spatial_intent)
→ fused image
```

## 已增强的结构

1. `SharedEncoder` 现在输出三层空间特征：
   - L1: H×W，用于浅层空间融合与频率分支输入；
   - L2: H/2×W/2，用于中层 CSAF；
   - L3: H/4×W/4，用于深层 PSAF。

2. `TextConditionedSpatialFusion` 现在采用三层 SPGFusion-style 结构：
   - L1: Text-conditioned CSAF pair；
   - L2: Text-conditioned CSAF pair + spatial self-attention；
   - L3: Text-conditioned PSAF，即 Cross_attention → CSAF → Fusion_Embed → Attention_spatial；
   - L3→L2→L1: Text-IF-style FeatureWiseAffine top-down aggregation。

3. `FusionDecoder` 也升级为三层 Text-IF-style decoder：
   - 内部构造 L1/L2/L3；
   - 每一级都有 FeatureWiseAffine；
   - 最终输出仍叠加 image-level skip，以稳定亮度和基础结构。

## 保留的方案边界

- 不使用 DINO；
- 不使用 CLIP image encoder；
- 不加载 Text-IF 整网权重；
- 不走 SPGFusion 的视觉语义先验生成路线；
- CLIP 仍只编码固定 prompt bank；
- Text Intent 仍来自 IntentRouter；
- 频率域 Text-guided amplitude/phase token routing 保持不变。

## 训练建议

该版本结构与旧 checkpoint 不完全兼容。建议重新训练。若要继承旧模型，只建议 `strict=False` 加载能匹配的 frequency_fusion 参数，然后重新训练 encoder、spatial_fusion 和 decoder。
