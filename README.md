# GPM-More

GPM-More 在 GPM 的 Pattern 编码器上加入跨 walk 关系 Token，用来保留不同路径之间的对应信息。

当前推荐配置是 **fresh 初始化＋完整关系反馈＋峰值学习率3e-4**。在 Computers 官方 high 划分首列上，seed0 最佳验证准确率为 **90.6182%**，本地 GPM 参考为 **90.1455%**。这是一组验证调参结果；GPM没有使用相同搜索预算，尚不能视为测试或多种子优势。

## 开始运行

```bash
python -m pip install -e ".[experiments,plots,dev]"
python scripts/fetch_experiment_sources.py gpm
python experiments/computers/run.py run --graph data/computers/task.npz --output runs/computers
```

需要准备 Computers 原始NPZ；参考环境为 Python3.10、PyTorch2.5.1+cu121、PyG2.6和8GB NVIDIA GPU。PyG采样扩展需要与Torch/CUDA版本匹配。

默认只跑当前推荐配置，不需要预训练权重。程序保存最佳验证模型和可恢复checkpoint，已完成的训练会跳过。

## 看哪里

- [方法](docs/METHOD.md)：关系Token怎样计算，反馈接口怎样工作。
- [数据](docs/DATA.md)：NPZ格式和官方划分。
- [训练与调参](docs/TUNING_STRATEGY.md)：默认参数、学习率搜索、停止与恢复。
- [实际结果](results/README.md)：当前数值及历史实验索引。
- [代码检查](docs/VALIDATION.md)：本次发布验证了什么。

`src/gpm_more` 是模型和通用训练代码；`experiments/computers` 是当前推荐的官方协议入口。
`research_extensions` 保留 Sports 探索源码，不属于推荐模型。

原始骨干来自 [GPM / ICML2025](https://github.com/Zehong-Wang/GPM)。许可与改编范围见 [NOTICE.md](NOTICE.md)。
