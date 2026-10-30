# 数据来源与科学边界

最终数据版本为 cwru_resampled_20261023。正常原始 97–100 文件为 48 kHz；先抗混叠降采样为 12 kHz，再切 1024 点、512 点步长的帧。故障文件原始为 12 kHz。依据为 Smith and Randall, 2015, DOI 10.1016/j.ymssp.2015.04.021, Appendix A Tables A1/A2；作者全文 https://www.researchgate.net/publication/276248554 。

28 个文件的实际 MAT 变量、SHA256、采样率、长度和划分见 raw_manifest.json。99.mat 明确选 X099_DE_time/X099_FE_time，避免旧加载器遍历顺序问题。训练用 load=0–2，load=3 源文件整体留出。StandardScaler 和量化比例仅由训练集拟合。

正常文件没有 BA；第三路由降采样后的 DE 移位派生。故障文件有 DE/FE/BA，但属于电火花人工加工缺陷，不能证明自然早期退化检测。不平衡为固定种子合成，伪负载标签只是拆分索引。

完整测试 2132 帧，按连续源文件边界得到 262 组，每组 8 帧，尾部共 36 帧不跨文件拼组。演示集 240 帧、30 组，逐帧与测试集 test_indices 匹配。演示分段使用单独的连续块 group_id，original_source 保留实际源编号。演示不是新增独立验证。

同一文件的相邻帧重叠 50%；组索引不同不等于时间样本或机器独立。当前两通道 DE/FE 对照和三通道均为 261/262，不能声称第三路提升准确率。旧 351/351 属于未重采样正常数据的历史版本，不用于新版结论。

原始来源 https://engineering.case.edu/bearingdatacenter/download-data-file 。data_audit.json 验证本机文件关系，不证明跨设备泛化或现场 ADC 输入。
