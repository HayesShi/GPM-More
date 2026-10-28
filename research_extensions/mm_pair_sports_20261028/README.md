# Sports探索代码

这是已完成的seed0链路预测试验。新增双端分支验证MRR30.4607%，低于独立端点版36.1605%；不属于当前推荐模型。

`prepare_data.py`下载固定官方版本，`run_campaign.py`执行四模型有限计划，`summarize.py`核对保存结果；其余脚本用于结构基线、采样机制、协议审计和绘图。

需要DGL1.1.2兼容环境及`python scripts/fetch_experiment_sources.py mm-graph`取得上游代码。计划模板为`configs/sports_pilot.example.json`，原始运行目录为`outputs/mm_pair_sports_20261028`。

可不下载数据，直接重画已发布图：

```bash
python research_extensions/mm_pair_sports_20261028/plot_evidence.py --published
```

原发布候选存在训练正边重叠，诊断切片并未修复这一点；数值保留在results/20261028/sports_pilot。不要将其与论文测试数合并声称优势。
