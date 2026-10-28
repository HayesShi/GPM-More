# 训练与调参

当前保留一套推荐方案：Computers官方high首列，fresh初始化，完整关系反馈。

## 默认配置

|参数|值|
|---|---|
|模型|完整raw_sum256|
|base LR|0.048|
|实际峰值LR|3e-4|
|优化器|AdamW，weight decay0.01|
|调度|inverse-sqrt，warmup100|
|训练/验证walk数|16/128|
|训练/验证batch|256/8|
|训练上限|1000epochs|
|验证与早停|每10epoch验证，patience10|
|其他|clip1，label smoothing0.05，seed0|

任务头使用seed70000000，新增PE8投影列初始化为零。保留原数值后端，不宣称逐比特配对。

## 运行

```bash
python experiments/computers/run.py run --graph data/computers/task.npz --output runs/computers
```

默认只训练base LR=0.048。若需要比较今天用过的两个学习率：

```bash
python experiments/computers/run.py run --graph data/computers/task.npz --output runs/computers-lr --learning-rates 0.01 0.048
```

对应峰值分别为6.25e-5与3e-4。每个候选共享数据、缓存和初始化。只按验证准确率选模型，同分保留先声明的配置；不读取测试指标。默认总预算3小时，恢复不会重置截止时间。

## 停止与恢复

在输出目录建立`STOP_REQUESTED`，程序会在完整验证checkpoint后停下。预算不足时也会停下，不完整运行不会标成完成。

需要恢复时，由操作者移走停止标记，原样重跑命令。模型、Adam状态、调度器、RNG和早停状态从latest.pt恢复；已完成候选直接跳过。不要更换原计划的参数或删除budget.json。

状态查询不启动训练，也不改写结果：

```bash
python experiments/computers/run.py status --output runs/computers
```

输出包括plan.json、budget.json、summary.json，以及每个trial的config、history、best/latest权重和验证预测。代码在第一次运行时保存快照。

现有通用autotune/dispatch代码保留，历史结果见results；主文档不再同时推荐第二套协议。
