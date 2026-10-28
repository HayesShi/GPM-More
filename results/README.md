# 实际结果

Computers官方high首列，seed0。下表均为验证准确率。

|方法/配置|峰值LR|准确率%|
|---|---:|---:|
|GPM参考|6.25e-5|90.1455|
|raw_sum：SSL初始化|6.25e-5|89.3818|
|raw_sum：SSL初始化|3e-4|90.5091|
|raw_sum：fresh|6.25e-5|90.0364|
|**raw_sum：fresh，当前推荐**|**3e-4**|**90.6182**|

最高值比原raw_sum高1.2364个百分点，比GPM参考高0.4727个百分点。所有raw_sum配置都使用完整反馈；这轮提升来自初始化和学习率选择。GPM没有接受同样的新增搜索，不能据此宣布等预算测试优势。

完整数值：[comparison.csv](20261028/computers_tuning/comparison.csv)。三个新增配置均按早停完成，总用时约101.71分钟；新配置未评测测试集。

## 历史记录

保留数据，便于核查；不作为另一套推荐方案。

- [五数据集80次试验](20261028/five_dataset_tuning/all_trials.csv)：后期停止反馈没有普遍提升，部分最优模型出现在切换之前。分类数据采用旧low划分，不能与上表混用。
- [Sports探索](20261028/sports_pilot/comparison.csv)：新增双端分支没有超过独立端点版；[图](20261028/sports_pilot/core_evidence.png)。
- 本目录其余CSV是早期机制实验。它们不是GPM/G²PM标准主表成绩。
