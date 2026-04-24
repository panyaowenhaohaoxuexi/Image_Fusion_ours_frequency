# 当前主干结构速览

当前工程已经切换到干净重写版主干，并加入频率参照引导的空间残差补偿：

1. `SharedEncoder`
   - 文件：`net/encoder/simple_encoder.py`
   - 功能：提取共享特征，并拆出 `base_feat / freq_feat` 两路特征。
   - 输出：`base_feat, freq_feat, shared_feat`。

2. `BaseFusion`
   - 文件：`net/fusion/base_fusion.py`
   - 功能：对 `vis_base / ir_base` 做初始基础融合，得到 `F_base_init`。
   - 作用：建立稳定的空间基础底座，保留可见光背景亮度与红外目标主体。

3. `HighLevelGuidedFrequencyFusion`
   - 文件：`net/frequency_fusion/fusion_block.py`
   - 功能：`FFT -> amplitude/phase token 化 -> intent 条件评分 -> Top-K 选择 -> selective interaction -> unselected token bypass -> 频谱重组 -> iFFT`。
   - 输出：`F_freq`，作为高层先验引导后的频率融合特征。

4. `SpatialResidualCompensation`
   - 文件：`net/fusion/spatial_compensation.py`
   - 功能：将 `F_freq` 投影为 `F_ref`，并计算 `vis_base - F_ref` 与 `ir_base - F_ref` 的源模态残差，通过门控筛选后补回 `F_base_init`。
   - 输出：补偿后的 `F_base`。
   - 作用：补回频率分支可能遗漏的可见光亮度/背景/纹理自然性，以及红外目标主体/显著响应/目标-背景对比。

5. `FusionDecoder`
   - 文件：`net/decoder/simple_decoder.py`
   - 功能：接收 `[F_base, F_freq]` 两路特征，并结合图像级 skip 重建融合图像。

当前 forward 主链路：

```python
vis_base, vis_freq, _ = shared_encoder(data_vis)
ir_base, ir_freq, _ = shared_encoder(data_ir)

fused_base_init = base_fusion(vis_base, ir_base)
fused_freq = frequency_fusion(vis_freq, ir_freq)
fused_base = spatial_compensation(fused_base_init, fused_freq, vis_base, ir_base)

decoder_skip = 0.5 * (data_vis + data_ir)
fused_image, _ = fusion_decoder(decoder_skip, fused_base, fused_freq)
```
