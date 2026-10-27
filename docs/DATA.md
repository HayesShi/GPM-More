# 从公开 GraphLand 数据到训练输入

仓库提供两个真实数据集的可选获取入口：`artnet-views`（回归）和`tolokers-2`（二分类）。原始数据来自[GraphLand官方Zenodo发布](https://zenodo.org/records/16895532)，特征与目标预处理调用[官方Dataset](https://github.com/yandex-research/graphland/blob/7246fe3f6b53cffaa76e3dc22726a6db3984bbe3/dataset.py)，固定源码版本`7246fe3f6b53cffaa76e3dc22726a6db3984bbe3`。

## 安装与获取

在仓库根目录执行：

```bash
pip install -e ".[graphland]"
python scripts/fetch_graphland.py --dataset artnet-views --root data/graphland --output data/artnet_graph.npz
gpm-more prepare --graph data/artnet_graph.npz --name artnet-views --task regression --output data/artnet --seed 0
```

获取脚本只下载所选数据集的ZIP，以及固定版本的`dataset.py`和其MIT许可证。ZIP用Zenodo发布的MD5核验；只提取`info.yaml`、`features.csv`、`targets.csv`、`edgelist.csv`和`split_masks_RL.csv`五个文件。ZIP成员路径不直接用于生成本地提取目标。输出NPZ及元数据必须使用新文件名，脚本拒绝覆盖已有输出。

缓存布局为：

```text
data/graphland/
  upstream/<revision>/dataset.py
  upstream/<revision>/LICENSE
  downloads/artnet-views.zip
  data/artnet-views/{info.yaml,features.csv,targets.csv,edgelist.csv,split_masks_RL.csv}
```

要使用Tolokers：

```bash
python scripts/fetch_graphland.py --dataset tolokers-2 --root data/graphland --output data/tolokers_graph.npz
gpm-more prepare --graph data/tolokers_graph.npz --name tolokers-2 --task binary_classification --output data/tolokers --seed 0
```

首次下载可能较大。已经存在的完整ZIP和五个原始文件会被复用。可通过`--raw-root PATH`复用`PATH/data/<dataset>/`，通过`--offline`要求只使用本地缓存。脚本的`--help`不加载图、不初始化CUDA，也不开始下载。

## 预处理与标签范围

官方Dataset在CPU上使用`split="RL"`、`use_pyg=True`、`to_undirected=True`、`add_self_loops=False`。特征转换遵循官方默认设置：在传导式任务中可以读取全图特征；回归目标StandardScaler仅在有标签训练节点拟合。导出的边额外去除自环、合并重复并按起终点排序。

导出NPZ恰好含六个数组：

| 数组 | 内容 |
|---|---|
| `x` | 已完成官方转换的全节点特征 |
| `edge_index` | 无向、无自环、去重的`[2,E]`图边 |
| `train_ids`、`val_ids` | 官方RL划分中有标签的训练/验证节点 |
| `train_y`、`val_y` | 对应节点的目标；ArtNet已经完成训练集拟合的标准化 |

**获取阶段会由官方loader读取完整的原始targets.csv，包括测试行；输出中只保存训练与验证标签，且不计算测试指标。** 这是官方原始预处理的边界。之后的`gpm-more prepare`和训练数据只保留导出的训练/验证标签；不能把后续训练数据的访问限制倒推成“上游原始CSV从未读过测试标签”。

ArtNet导出目标已经标准化，因此上述`prepare`命令不添加`--standardize-targets`。旁边的`.metadata.json`记录官方版本、发布地址、任务、转换参数、数组范围和软件版本。获取脚本没有调用官方测试评估函数。

## Walk缓存与已有数字

`gpm-more prepare`会从导出的图生成新的16条普通随机游走和8条非回溯游走缓存，路径长度为9。种子控制这次缓存生成；它不意味着生成了原工作区历史实验的相同walk序列。也可通过`prepare --bank`复用符合格式的既有缓存，详情见[使用说明](USAGE.md)。

[实验证据](EVIDENCE.md)中的历史数字来自当时的缓存、预训练权重和中途优化器/RNG状态。当前公开获取入口能运行同一研究方法的新实验，不能据此声称重新下载数据、随机初始化模型就会直接得到历史R²。公开包未附带那些历史训练资产。

## 来源和许可

GraphLand源码由Oleg Platonov、Gleb Bazhenov和Liudmila Prokhorenkova发布，源码许可为MIT；下载脚本把该版本许可证与源码一并放入本地缓存。数据集本身的来源、许可和使用条件以[Zenodo发布记录](https://zenodo.org/records/16895532)及GraphLand项目说明为准；源码MIT许可不替代各数据集的许可条件。本仓库没有重新分发原始GraphLand数据。
