# 给初学者看的当前网络结构

当前网络只有 4 个核心模块：

1. `Restormer_Encoder`
   - 现在实际是 `SimpleSharedEncoder`
   - 提取 `base_feat` 和 `freq_feat`

2. `BaseFeatureExtraction`
   - 现在实际是 `SimpleBaseFusion`
   - 融合基础共享信息

3. `HighLevelGuidedFrequencyFusion`
   - 这是你的核心创新模块
   - 做 FFT、幅度相位分解、token 选择与交互

4. `Restormer_Decoder`
   - 现在实际是 `SimpleDecoder`
   - 把两路融合特征重建成最终图像
