# 当前主干结构速览

当前工程已经切换到干净重写版主干：

1. `SharedEncoder`
   - 文件：`net/encoder/simple_encoder.py`
   - 功能：提取共享特征，并拆出 base / freq 两路特征。

2. `BaseFusion`
   - 文件：`net/fusion/base_fusion.py`
   - 功能：融合低频共享信息，保持结构稳定。

3. `HighLevelGuidedFrequencyFusion`
   - 文件：`net/frequency_fusion/fusion_block.py`
   - 功能：FFT -> 相位/幅度 token 化 -> intent 条件评分 -> Top-K 选择 -> selective interaction -> 轻量保留路径 -> 频谱重组 -> iFFT。

4. `FusionDecoder`
   - 文件：`net/decoder/simple_decoder.py`
   - 功能：接收 `[F_base, F_freq]` 两路特征，重建融合图像。
