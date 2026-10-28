# 数据

当前推荐入口只针对Computers，使用GPM官方high划分的首列：8251训练、2750验证、2751测试节点。

输入是NPZ，包含：

|键|内容|
|---|---|
|x|13752×767原始节点特征|
|edge_index|2×E整数边索引|
|y|13752个类别标签|

不要提前归一化x。脚本沿用官方NormalizeFeatures、去自环、双向化及LapPE8，再按官方seed42流程产生划分。训练只解码训练和验证标签。

首次运行会生成128条长度8的SRW缓存及93维结构描述。训练每节点抽16条，验证用全部128条。相同调参队列共享缓存。

如果已有今天使用的完整缓存，可添加：

```bash
python experiments/computers/run.py run --graph data/computers/task.npz --prepared YOUR_CACHE --output runs/computers
```

缓存需要包含shared_graph.npz、train_val.npz、patterns.pt、paths.npy及完成记录。复用原缓存能避免重新计算LapPE造成的数值轨迹差异。

通用`gpm-more prepare/train`仍支持其他图，但不会自动复现它们的论文协议。比较前必须核对各自划分和预处理。数据与权重不随Git仓库发布。
