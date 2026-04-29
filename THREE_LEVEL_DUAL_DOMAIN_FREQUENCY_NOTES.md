# Three-level Dual-domain Text-intent Fusion Notes

本版本在 v8 三层空间 SPGFusion/Text-IF 分支基础上，将频率域也升级为三层多尺度结构，使空间域与频率域保持同构。

## 1. Encoder 输出

`SharedEncoder.forward()` 现在返回：

```text
spatial_feats   = [spa_l1, spa_l2, spa_l3]
frequency_feats = [freq_l1, freq_l2, freq_l3]
shared_feat
```

其中：

```text
L1: H × W，高分辨率细节/纹理/局部亮度
L2: H/2 × W/2，中尺度结构/目标边界/区域纹理
L3: H/4 × W/4，全局结构/显著目标/频谱能量分布
```

## 2. 多尺度频率域

`HighLevelGuidedFrequencyFusion` 现在支持输入三层频率候选特征：

```text
vis_freq = [vis_freq_l1, vis_freq_l2, vis_freq_l3]
ir_freq  = [ir_freq_l1,  ir_freq_l2,  ir_freq_l3]
```

每一层独立执行：

```text
FFT
→ amplitude / phase decomposition
→ Token Feature + Cross-modal Difference + Text Intent scoring
→ Top-K routing
→ selected tokens strong interaction
→ unselected tokens lightweight preservation
→ amplitude / phase reconstruction
→ iFFT
```

然后进行 top-down frequency aggregation：

```text
F_freq_l3
→ upsample + F_freq_l2
→ upsample + F_freq_l1
→ F_freq
```

## 3. Text Intent 一致性

同一个 Text Intent Embedding 被分为：

```text
Frequency-Intent Head L1
Frequency-Intent Head L2
Frequency-Intent Head L3
Spatial-Intent Head
```

因此，文本意图在频率域控制多尺度 amplitude/phase token routing，在空间域控制三层 CSAF/PSAF 条件调制。

## 4. 训练与测试调用

训练和测试中现在使用：

```python
vis_spa, vis_freq, _ = encoder(data_vis)
ir_spa, ir_freq, _ = encoder(data_ir)

fused_freq, freq_aux = frequency_fusion(vis_freq, ir_freq)
freq_context = freq_aux.get('fused_freq_levels', fused_freq)

fused_spa = spatial_fusion(vis_spa, ir_spa, freq_aux['spatial_intent'], freq_context)
fused_image, _ = decoder(decoder_skip, fused_spa, fused_freq, text_intent=freq_aux['spatial_intent'])
```

## 5. Loss 兼容

`TokenRoutingRankingLoss` 已兼容多尺度 score。若 `amp_score / phase_score` 是 list，会逐层计算 ranking loss 后求平均。

## 6. 验证

已完成：

```text
Python 静态语法检查：通过
小尺寸随机张量 forward/backward：通过
```

测试配置：

```text
B=1, C=16, H=W=32
三层 frequency_feats: 32×32 / 16×16 / 8×8
三层 spatial_feats:   32×32 / 16×16 / 8×8
```

梯度检查中，L1/L2/L3 amplitude scorer 均获得梯度，说明多尺度频率 token routing 已接入训练链路。
