"""Write measured findings after the frozen four-model pilot completes."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mm_pair_sports_20261028"
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"


def main():
    status = json.loads((OUT / "status.json").read_text())
    assert status["complete"] and status["phase"] == "completed"
    summary = json.loads((OUT / "verified_summary.json").read_text())
    assert summary["completed_trained_models"] == 4 and not summary["verification_errors"]
    exposed = np.load(DATA / "valid_reverse_in_train.npy")
    loops = np.load(DATA / "valid_self_loop.npy")
    clean = ~exposed & ~loops
    on = np.load(OUT / "runs/pair_joint/valid_rr_final.npy")
    off = np.load(OUT / "runs/pair_joint/no_joint_final_rr.npy")
    intervention = json.loads((OUT / "runs/pair_joint/fixed_weight_no_joint.json").read_text())
    deltas = {}
    for name, mask in [("full", np.ones(len(on), dtype=bool)), ("clean", clean), ("reverse_exposed", exposed)]:
        assert abs(float(off[mask].mean()) - intervention["validation"][name]["mrr"]) < 1e-7
        deltas[name] = {"n": int(mask.sum()), "on": float(100*on[mask].mean()),
                        "off": float(100*off[mask].mean()), "on_minus_off_pp": float(100*(on[mask]-off[mask]).mean())}
    tie = json.loads((OUT / "structural_tie_probe/results.json").read_text())
    budget = json.loads((OUT / "budget.json").read_text())
    elapsed = status["updated_unix"] - budget["started_unix"]
    assert status["updated_unix"] < budget["deadline_unix"]
    evidence = {"four_models_complete": True, "campaign_wall_seconds": elapsed,
                "within_original_90_minute_deadline": True, "fixed_weight_subgroups": deltas,
                "test_performance_evaluated": False, "formal_main_table_modified": False}
    (OUT / "completion_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    t = tie["results"]["clean_valid"]
    text = f"""# Sports试验：结果、机制及方法去留

本轮四个学习模型均完成，seed0；队列用时{elapsed/60:.2f}分钟，未延长原90分钟预算。训练进程已结束。全部模型在本轮预算内的最佳checkpoint恰好均为最终checkpoint。

## 1. 结果改变了什么判断

raw_sum独立端点版验证MRR为36.1605%，比本地SAGE的29.7805%高6.3800个百分点。但新增双端路径分数后只有30.4607%，比匹配独立版低5.6998个百分点。精确RA无需训练达到42.3137%，又比独立版高6.1532个百分点。因此不能把这版联合模型作为已经验证的性能创新，也不能仅以超过本地SAGE为依据忽略更强、更便宜的结构基线。

本轮是从头训练的融合CLIP特征链路预测开发试验。它没有验证图像、文本各自的贡献，没有测试独立模态分支、交换损失或关系梯度协调规则；不能将结果解释为多模态梯度冲突的证据。

## 2. 最清楚的干预发现：整体微小收益掩盖相反的群体作用

固定最终联合模型的所有权重与编码缓存，仅关闭新增双端分数：

|验证群体|查询数|开启MRR(%)|关闭MRR(%)|开启减关闭(pp)|
|---|---:|---:|---:|---:|
|完整验证|{deltas['full']['n']}|{deltas['full']['on']:.4f}|{deltas['full']['off']:.4f}|{deltas['full']['on_minus_off_pp']:+.4f}|
|无直接反向曝光且非自环|{deltas['clean']['n']}|{deltas['clean']['on']:.4f}|{deltas['clean']['off']:.4f}|{deltas['clean']['on_minus_off_pp']:+.4f}|
|反向边已在训练图|{deltas['reverse_exposed']['n']}|{deltas['reverse_exposed']['on']:.4f}|{deltas['reverse_exposed']['off']:.4f}|{deltas['reverse_exposed']['on_minus_off_pp']:+.4f}|

前向关系分数在反向曝光群体上的收益足以抵消另一群体的损失，使完整验证仍显示微小净收益。分组不是随机处理，尚未移除反向边做图干预，所以不能据此证明“反向边曝光本身”是原因。能够确定的是：对同一已训练模型，分数开关的即时效果明显异质，单个总体均值没有表达这一点。

关闭后诊断切片31.1143%仍低于独立训练的36.1285%。这个差距与联合训练后的表示/预测头协同变化相容，但还不能单独归因于关系梯度。前向开关、训练开关、梯度开关是不同干预。

## 3. 是否至少解决了结构相同候选的区分问题

在同一查询内固定正负候选的精确RA完全相等且大于零，比较两模型FINAL预测。规则在读取joint FINAL前保存，没有选择有利容差。

诊断切片包含{t['eligible_queries']}个查询、{t['eligible_candidate_position_pairs']}个正负比较对。独立版配对正确率为{100*t['independent']['pair_weighted_correctness']:.2f}%，联合版为{100*t['joint']['pair_weighted_correctness']:.2f}%；逐查询平均分别为{100*t['independent']['query_averaged_correctness']:.2f}%与{100*t['joint']['query_averaged_correctness']:.2f}%。

因此当前联合版也没有在“RA无法区分的候选”上展示增量。这里只控制一个结构统计，未固定全部拓扑、内容与度数，不是因果识别。精确浮点相等可能漏掉数学上相等但累加误差不同的值。原发布负候选问题也仍然存在。

## 4. 数学和同缓存干预告诉我们什么

常量内容、首步碰撞的解析特例满足 E[C]=CN/(d_u d_v)。它对候选端点度数作惩罚；RA则以共同邻居度数的倒数计权，二者不同。保持同一缓存，度数修正使K=16首步统计的诊断MRR从21.0378%变为29.0998%。这隔离了计权方式的影响，仍未达到精确RA的42.3391%。

首步漏碰24.2738%的真实共同邻居正例。八步的任意共享节点漏碰率已降至0.1090%，其原始碰撞MRR仍只有24.3045%。因此不能把完整八步模型的落后全部解释为“没有采到交集”；不同深度相遇的结构含义及计权同样重要。标量特例也不等同于带可学习上下文核的完整模型。

## 5. 一张核心图如何阅读

![核心证据](core_evidence.png)

- A固定候选、缓存与K，仅改计权，显示观测分布与目标结构统计并不相同。
- B显示首步证据覆盖限制；它不代表整个八步缓存缺少任何相遇。
- C显示相同处理正例数下的真实训练轨迹，联合版持续落后，精确RA更强。不同模型的算力和优化步骤并不相等。
- D固定最终联合模型的权重，展示同一分支在不同群体上的相反作用；蓝虚线为另行匹配训练的独立端点参考，不能解释为同权重干预。

## 6. 留下什么方法方向

停止将当前原始碰撞联合分支作为主模型扩展。一个有明确数学对象、值得后续检验的候选是“精确结构证据＋双端图文语境残差”：

\\[
\\widehat T_\\theta(u,v)=\\mathrm{{RA}}(u,v)+\\frac{{d_ud_v}}{{K^2}}
\\sum_{{p,q:\,W_{{u,p,1}}=W_{{v,q,1}}=w}}
\\frac{{\\kappa_\\theta(H_{{u,p,1}},H_{{v,q,1}})-1}}{{d_w}}.
\\]

当语境核恒为1，任意有限缓存下都精确恢复RA，不再用随机碰撞重新估计已知且廉价的基础结构。只有语境残差由路径估计。推导及条件见next_mechanism.md。

重要性估计、control variates、共同邻居神经聚合都已有先例，不能包装成原创理论。真正可能成立的新问题是：在结构证据给定时，同一共同邻居在两端图文路径语境中的角色，是否提供静态共同邻居特征没有的关系信息？只有超越相同RA输入的静态witness模型、NCN等强对照，并经固定witness的语境置换和模态消融支持，才值得发展成ACM MM方法。此候选尚未在本轮训练，当前结果没有证明它会涨点。

这也意味着不应机械模仿范文的“新空间＋新损失”外形。可借鉴的是每个数学构造都服务于同一个可检验问题。本轮没有支持新流形或新损失的证据。

## 7. 数据和比较边界

官方Sports共购任务不提供互补/替代标签，不能改称互补关系识别。发布训练/验证有向边不重叠，但官方双向化暴露8264个验证正例的反向边。另有185714个验证负候选位置对应原始训练正边，涉及81.75%的验证查询。后者不是双向化才产生，也没有被诊断切片修复。原split和候选均保持不变。

论文Table5的SAGE、GCN、MMGCN、BUDDY等测试结果在REPORT.md独立列出；本地只有MLP/SAGE实际运行，不能把文献测试数与此处验证数合并判断胜负。没有读测试指标选择方法，没有改正式主表，没有启动Cloth、多种子或额外搜索。

## 8. 复现和产物

完整本地比较见REPORT.md、comparison.csv、verified_summary.json；逐查询预测/排名、权重与轨迹在runs/；结构统计、同缓存探针、结构相等候选分析分别在structural_baselines/、sampling_mechanism/、structural_tie_probe/。核心图提供PNG/SVG/PDF。

完成后的分析复现依次运行summarize.py、structural_tie_probe.py、plot_evidence.py、finalize_analysis.py。训练快照在code_snapshot，报告脚本另存analysis_snapshot，历史预算与原始日志均保留。run_campaign.py run只恢复同一有限计划，不会把已完成试验重新训练。
"""
    (OUT / "FINDINGS.md").write_text(text, encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
