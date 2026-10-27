# 与已有工作的关系

本项目研究单个预测目标内部，不同计算路径如何反馈到同一个共享图编码器。下面列出理解这一定位所需的近邻概念；这不是完整的新颖性检索。

## GPM 是模型基础

[Beyond Message Passing: Neural Graph Pattern Machine](https://arxiv.org/abs/2501.18739) 是 ICML 2025 的 GPM 工作，[官方代码](https://github.com/Zehong-Wang/GPM)提供随机游走 Pattern 编码与聚合的基础实现。本仓库在此基础上整理跨 walk 相位关系收缩和反馈控制；使用上游活跃计算代码时保留相应 MIT 许可与归属。

GPM 的 Pattern 表示是本项目起点。这里的额外关系 Token 以池化前上下文与原图节点身份为输入，显式计算不同 walk 的对应关系。这个设计是否带来额外表达与任务价值，应由结构区分与任务实验说明，不能由模型名称推导。

## Stop-gradient 与自监督学习

SimSiam 的 *Exploring Simple Siamese Representation Learning* 使用 stop-gradient 作为其自监督学习机制的一部分。它说明控制反向路径已有明确先例。GPM-More 不把 stop-gradient 算子本身作为新贡献；这里关注派生关系路径在单一监督预测任务中，既使用又修改共享表示的作用。

## 多任务梯度协调

PCGrad 的 *Gradient Surgery for Multi-Task Learning* 与 GradNorm 的 *Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks* 分别代表梯度方向协调与任务权重调整的已有思路。

本项目的普通路径和关系路径可以服务同一个损失、同一组标签，路径分解不要求存在两个任务损失。这个设定差异有助于明确问题，但本身不构成对所有梯度路由工作的排他性新颖性证明。局部梯度夹角也没有被当作泛化收益的充分条件。

## 候选更新选择

用额外训练观测在两个实际更新之间选择，与随机线搜索、更新接受规则、梯度门控及双层加权具有概念联系。本项目测试过精确的 full/stop 端点选择，并记录了不足以继续的方法收益。该负结果限定了一种具体实现，不否定这些已有方法的其他用途。

## 当前贡献定位

值得继续验证的是：同一任务内，派生关系的预测价值与其塑造共享表示的作用可以被分别干预；这种反馈耦合在基于 Pattern 的图模型中表现出什么规律，以及什么控制方式能取得可重复的额外价值。

历史机制实验和代码接口已经围绕这个问题组织。新的通用优化原理、完整文献排他性结论和标准基准优势仍需各自的证据。
