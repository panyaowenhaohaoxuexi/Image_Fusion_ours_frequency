# 本次代码重构说明

本次改动目标：
1. 不再把频率融合模块全部写在一个文件里；
2. 先把你的网络结构构造出来；
3. 先不接真实 CLIP，只保留高层先验接口；
4. 增加中文注释，便于后续继续改。

新增目录：`net/frequency_fusion/`

核心类：`HighLevelGuidedFrequencyFusion`
这就是现在在 `train.py` / `test.py` 中真正被调用的模块。
