# 方法

GPM先编码随机游走得到的Pattern，再用Transformer聚合。GPM-More保留这个骨干，增加一个跨walk关系Token。

关系算子读取池化前的节点上下文，用原图节点身份对齐不同walk中的访问，再通过相位双线性配对汇总关系。它补充的是路径之间的对应信息；原有Pattern内容仍走原来的编码与聚合流程。

关系Token和普通Pattern Token一起进入外层Transformer。当前推荐配置使用完整关系反馈，模型宽度256。

设第p条walk在位置t访问节点v，得到上下文h。第d个关系通道计算为：

$$r_d=\frac{1}{64K(K-1)}\sum_{p\ne q}\sum_{t,s=1}^{8}
\mathbf{1}[v_{p,t}=v_{q,s}]h_{p,t,d}h_{q,s,d}\cos\!\bigl(\omega_d(t+s)\bigr).$$

只有不同walk访问同一节点的项参与计算，起始位置不计入；相位使用位置和t+s。实现先按节点汇总，再减去同walk项，避免显式枚举所有路径对。这个算子复用已有访问，不额外查询图边。

## 反馈接口

代码允许单独控制关系路径对共享编码器的反向作用：

$$\widetilde h_R=\operatorname{sg}(h)+\alpha[h-\operatorname{sg}(h)].$$

前向值始终为h，原始反向梯度变为g_P＋αg_R。α=1是完整反馈，α=0停止关系路径回传，关系算子之后的参数仍正常训练。AdamW的实际更新还取决于历史动量。

这个接口用于机制对照。今天验证最好的Computers配置用α=1；后期停止反馈没有在已跑任务上普遍提高最佳验证成绩，因此不是默认方案。Stop-gradient本身不作为新发明。

## 代码入口

- `src/gpm_more/backbone.py`：Pattern骨干。
- `src/gpm_more/operators.py`：跨walk相位关系算子。
- `src/gpm_more/model.py`：关系Token与预测。
- `src/gpm_more/feedback.py`：反馈强度和调度。

方法仍需用公平基线和多种子结果检验。现有实验不支持“关系梯度普遍有害”的结论。
