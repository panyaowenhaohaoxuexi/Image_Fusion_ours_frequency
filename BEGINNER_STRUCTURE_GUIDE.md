# 初学者版结构说明

你当前真正需要看的文件有：

1. `train.py`：单阶段训练入口
2. `test.py`：推理入口
3. `net/frequency_fusion/prompt.py`：高层先验与 intent router（当前还没有接真实 CLIP）
4. `net/frequency_fusion/fft_utils.py`：FFT、幅度/相位分解、patchify / unpatchify
5. `net/frequency_fusion/scoring.py`：token 评分
6. `net/frequency_fusion/selection.py`：Top-K 选择
7. `net/frequency_fusion/interaction.py`：选中 token 的跨模态交互
8. `net/frequency_fusion/fusion_block.py`：总融合模块

## 当前网络主线

输入红外/可见光 -> 编码器 -> 基础分支融合 + 高频/频率分支融合 -> 解码器 -> 融合图像

## 当前训练主线

当前不再使用 TSFI-Fusion 的两阶段训练。

现在是 **单阶段端到端训练**：

- 从第 1 个 epoch 开始，所有模块一起训练；
- 不再单独训练自编码重建阶段；
- 你的创新点只聚焦在“高层先验引导 + 频率 token 选择 + Top-K 交互 + 频谱重组”。
