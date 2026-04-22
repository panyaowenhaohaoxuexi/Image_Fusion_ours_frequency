# 初学者版网络结构说明

你现在真正需要看的文件只有这些：

1. `train.py`：训练入口。
2. `test.py`：推理入口。
3. `net/frequency_fusion/prompt.py`：高层先验。当前没有接真实 CLIP。
4. `net/frequency_fusion/fft_utils.py`：FFT、幅度/相位重建、patchify。
5. `net/frequency_fusion/scoring.py`：token 评分。
6. `net/frequency_fusion/selection.py`：Top-K 选择。
7. `net/frequency_fusion/interaction.py`：Top-K token 跨模态交互。
8. `net/frequency_fusion/fusion_block.py`：总融合模块。

当前阶段先把网络结构搭起来、跑通，再考虑把 `prompt.py` 换成真实 CLIP embedding。
