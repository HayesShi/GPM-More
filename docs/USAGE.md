# 使用与复现

## 最小数据接口

`gpm-more prepare` 接收已经完成特征预处理的图NPZ。它支持两种标签布局：

| 数组 | 形状 | 含义 |
|---|---|---|
| `x` | `[N,F]` | 有限的float特征；转为float32 |
| `edge_index` | `[2,E]` | 整数节点索引；采样时去自环、合并重复并转无向 |
| `train_ids`,`val_ids` | `[N_train]`,`[N_val]` | 不重合的标签根节点 |
| `train_y`,`val_y` | 对应标签行 | 回归、0/1二分类或从0开始的整数类别 |

另一种布局为`train_mask`,`val_mask`与全节点`y`。准备器只解析所允许的训练/验证标签行，不实例化其余标签值；压缩文件的跳转可能仍需解压跳过字节。两种布局都不需要测试标签。

```bash
gpm-more prepare --graph graph.npz --task regression --name my-graph --output data/my-graph --seed 0
```

原始回归目标需要标准化时可加`--standardize-targets`，均值与尺度只用训练标签拟合。GraphLand官方预处理输出已标准化，应保持该选项关闭。

准备后的目录包含：

```text
data.npz                   特征、训练/验证根节点及对应标签
context_paths.npy           [N,16,9] SRW
context_relations.npy       [N,16,93] 固定结构描述符
extra_paths.npy             [N,8,9] NBRW
extra_relations.npy         [N,8,93]
metadata.json               任务、标签范围与观测协议
```

已有同布局缓存可以通过`--bank <folder>`复制复用。路径长8次转移，位置数9。关系输入包含36个位置对等值位、36个诱导邻接位及21类带根四节点graphlet指示。孤立节点停留原地；非回溯游走在叶节点允许强制返回。

## 训练配置

```bash
gpm-more train --data data/my-graph --config configs/artnet_feedback.json --output runs/my-feedback --device cuda:0
```

默认配置为256维完整活动骨干，批次128、微批次32，训练预算从4/8/12/16采样；16条路径批次可选择SRW或NBRW扩展。验证固定12条SRW。AdamW、100步warmup、总长1000步余弦调度，400步之后额外学习率×0.1，401步起停止关系反馈。全参数学习率倍率也作用于AdamW权重衰减。

反馈策略包括`full`、`stop`、`constant`与`late_stop`。`constant`的alpha必须在[0,1]，`late_stop`在`stop_after`之后返回0。这些配置用于可检验对照，不表示已证明每个任务都应在400步停止反馈。

`common_full`裁剪先用完整梯度计算共同系数，再用相同采样/dropout重新计算所需反馈梯度。当alpha=1时只需一次反向。该策略保留历史控制实验的裁剪口径；额外计算会反映在日志中。默认模型的关系算子没有可训练参数，控制施加在其上下文输入，因此普通路径继续完整学习。

## 初始化与续训

```bash
# 导入活动模型权重，不导入历史优化器轨迹
gpm-more train --data data/my-graph --config configs/artnet_feedback.json --output runs/from-weights --init-weights model.pt --device cuda:0

# 使用本训练器保存的完整状态续训
gpm-more train --data data/my-graph --config configs/artnet_feedback.json --output runs/my-feedback --resume runs/my-feedback/latest.pt --device cuda:0
```

历史raw_sum权重按活动键严格导入，缺少活动权重或出现未知键会报错；未参与当前前向的VQ、旧phase模块、模板层与SSL预测头可以被明确忽略。权重转换不等于Adam/RNG转换。原有历史优化器状态不能与重新构建的参数清单直接混用。

本训练器的完整checkpoint保存模型、优化器、调度器及NumPy/Torch随机状态。续训要求原配置的总调度长度和模型/数据描述相符。`best.pt`来自预定验证点，`latest.pt`是最近完整边界；日志同时保留当前与最佳验证结果。

## 如何理解发布的结果

仓库内CSV是已经完成的历史机制实验聚合值，新训练命令不会自动复现这些数值。历史实验使用结构预训练权重、固定观测缓存，以及保存的第400或600步参数/Adam状态。新采样、随机初始化或仅导入权重都会改变实验条件。

本次代码整理验证了活动前向、反馈梯度语义和完整续训，不新增标准基准结果。具体记录见[代码验证](VALIDATION.md)与[证据来源](EVIDENCE.md)。
