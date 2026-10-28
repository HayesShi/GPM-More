"""Regenerate validation-only pilot report from completed result files."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mm_pair_sports_20261028"
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"


def main():
    exposed = np.load(DATA / "valid_reverse_in_train.npy")
    self_loop = np.load(DATA / "valid_self_loop.npy")
    clean = ~exposed & ~self_loop
    rows, errors, resources = [], [], []
    for method, label in [("mlp", "MLP / official architecture"), ("sage", "SAGE / official architecture"),
                          ("pair_independent", "raw_sum256-LP / independent endpoints"),
                          ("pair_joint", "raw_sum256-LP / joint path witnesses")]:
        run = OUT / "runs" / method
        if not (run / "metrics.json").exists():
            continue
        metadata = json.loads((run / "metrics.json").read_text())
        if not metadata.get("complete"):
            continue
        rr = np.load(run / "valid_rr.npy")
        final_rr = np.load(run / "valid_rr_final.npy")
        assert rr.shape == final_rr.shape == (45557,)
        assert np.isfinite(rr).all() and (rr > 0).all() and (rr <= 1).all()
        assert np.isfinite(final_rr).all() and (final_rr > 0).all() and (final_rr <= 1).all()
        stated = metadata["best_valid"]["mrr"] if method in ("mlp", "sage") else metadata["best"]["full_mrr"]
        delta = abs(float(rr.mean()) - stated)
        if delta > 1e-7:
            errors.append({"method": method, "error": delta})
        final_stated = metadata["final_valid"]["mrr"] if method in ("mlp", "sage") else metadata["final_validation"]["full"]["mrr"]
        if abs(float(final_rr.mean()) - final_stated) > 1e-7:
            errors.append({"method": method, "kind": "final_mrr", "error": abs(float(final_rr.mean()) - final_stated)})
        resources.append({"method": method, "seconds": metadata.get("elapsed_seconds", metadata.get("wall_seconds_this_execution")),
                          "train_seconds": metadata.get("train_seconds"), "peak_allocated_GiB": metadata.get("peak_allocated_GiB")})
        rows.append({"method": label, "kind": "trained_seed0", "full_val_mrr_percent": 100*rr.mean(),
                     "clean_val_mrr_percent": 100*rr[clean].mean(),
                     "reverse_exposed_mrr_percent": 100*rr[exposed].mean(),
                     "final_full_val_mrr_percent": 100*final_rr.mean(),
                     "final_clean_val_mrr_percent": 100*final_rr[clean].mean(),
                     "best_step": metadata["best_valid"]["step"] if method in ("mlp", "sage") else metadata["best"]["step"],
                     "trained_positive_examples": metadata["step"] * (512 if method in ("mlp", "sage") else 128)})
    for method, label in [("edge", "Observed edge indicator"), ("cn", "Common neighbors (CN)"), ("ra", "Resource Allocation (RA)")]:
        rr = np.load(OUT / "structural_baselines" / f"{method}_rr.npy")
        rows.append({"method": label, "kind": "no_training", "full_val_mrr_percent": 100*rr.mean(),
                     "clean_val_mrr_percent": 100*rr[clean].mean(),
                     "reverse_exposed_mrr_percent": 100*rr[exposed].mean(),
                     "final_full_val_mrr_percent": 100*rr.mean(), "final_clean_val_mrr_percent": 100*rr[clean].mean(),
                     "best_step": None, "trained_positive_examples": 0})
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    rows = [{k: v.item() if isinstance(v, np.generic) else v for k, v in r.items()} for r in rows]
    intervention = None
    intervention_path = OUT / "runs/pair_joint/no_joint_final_rr.npy"
    if intervention_path.exists():
        off_rr = np.load(intervention_path)
        on_rr = np.load(OUT / "runs/pair_joint/valid_rr_final.npy")
        intervention = {"kind": "same final joint-trained checkpoint, disable only added endpoint score",
                        "on_full_mrr_percent": float(100 * on_rr.mean()),
                        "off_full_mrr_percent": float(100 * off_rr.mean()),
                        "on_clean_mrr_percent": float(100 * on_rr[clean].mean()),
                        "off_clean_mrr_percent": float(100 * off_rr[clean].mean())}
    summary = {"rows": rows, "verification_errors": errors, "test_evaluated": False,
               "resources": resources, "fixed_weight_intervention": intervention,
               "completed_trained_models": sum(r["kind"] == "trained_seed0" for r in rows)}
    (OUT / "verified_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Amazon-Sports：双端 Graph Pattern 的首轮真实试验", "",
             "本报告由保存的每条验证查询 reciprocal rank 重新生成。开发试验，seed0；不修改论文正式主表，不报告测试性能。", "",
             "## 协议与比较", "",
             "采用 MM-GRAPH 官方 Sports 发布划分、CLIP 1024维特征和完整45557个验证正例，每正例150个固定负例。",
             "所有学习方法最多观察512000条训练正例，统一Uniform(1)负采样；MLP/SAGE batch512、1000步，两个raw_sum臂batch128、4000步。该预算约为官方代码一个capped epoch，不是论文30epochs/HPO复现。",
             "MLP/SAGE保留官方三层256维模型及预测头；Windows DGL无CUDA图后端，SAGE采用官方CPU采样和同算子的Torch CUDA聚合。",
             "raw_sum两臂共享完整256维骨干、真实93维结构描述、单根closure与LP头。联合臂增加16秩、按深度投影后在同一真实节点上收缩的双端得分；其余初始化、目标边、负例、walk种子与优化配置匹配。",
             "本轮raw_sum两臂从同一随机初始化训练，没有加载此前SSL预训练权重；独立端点臂仍保留原有单根关系Token，仅关闭新增双端分数。因此这不是此前节点任务微调的直接延续。MLP/SAGE与raw_sum只匹配处理正例数，优化器、批量、步骤、参数量及算力不同。",
             "训练目标及反向边在当前batch采样和描述探针中同步排除。评估固定16条长度8的SRW缓存，所有候选复用节点路径编码。", "",
             "## 必须保留的数据发现", "",
             "官方有向split互不重叠，但官方训练代码双向化后，8264条验证正例已有反向训练边，另有3条验证自环。我们保留官方协议，同时报告37290条无直接反向曝光且非自环的诊断切片。它不替代官方split，也不等于重新构造的全图严格无泄漏基准。",
             "另有185714个验证负候选位置（全部6833550个位置的2.71768%）在原始有向训练集中已经是同方向正边，涉及37244/45557个查询；训练图双向化后的命中负候选共236808个。该问题在上述clean诊断切片中也未被修复。候选保持原样用于同协议对照；不能以这里的MRR单独证明真实未观察关系泛化。完整证据见negative_overlap_audit.json。",
             "checkpoint按完整验证MRR选择；clean列评估同一checkpoint，并非另选clean最优。", "",
             "## 已完成本地验证结果", "",
             "MRR(%)。无训练统计与学习方法分清；单种子不提供虚假的seed标准差。", "",
             "|方法|完整验证|clean诊断切片|最终完整验证|最佳step|", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'|{r["method"]}|{r["full_val_mrr_percent"]:.4f}|{r["clean_val_mrr_percent"]:.4f}|{r["final_full_val_mrr_percent"]:.4f}|{r["best_step"] if r["best_step"] is not None else "—"}|')
    if intervention:
        lines += ["", "## 固定权重干预", "",
                  "使用联合训练的最终checkpoint，仅关闭新增双端分数，保留所有参数、单根关系Token及缓存。它测量这个已训练模型对分支的即时依赖，不能单独说明关系梯度造成了什么。", "",
                  "|最终联合模型|完整验证MRR(%)|clean诊断MRR(%)|", "|---|---:|---:|",
                  f'|双端分数开启|{intervention["on_full_mrr_percent"]:.4f}|{intervention["on_clean_mrr_percent"]:.4f}|',
                  f'|同一权重关闭双端分数|{intervention["off_full_mrr_percent"]:.4f}|{intervention["off_clean_mrr_percent"]:.4f}|']
    lines += ["", "## 资源与运行记录", "", "RTX4060 Laptop 8GB；CPU采样/诊断使用2线程。以下为各方法实际墙钟时间；无训练统计评分CN/RA合计约2.56秒，二者不包含数据下载时间。", "",
              "|方法|墙钟秒|训练秒|Torch峰值已分配GiB|", "|---|---:|---:|---:|"]
    for r in resources:
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        lines.append(f'|{r["method"]}|{fmt(r["seconds"])}|{fmt(r["train_seconds"])}|{fmt(r["peak_allocated_GiB"])}|')
    lines += ["", "MLP/SAGE原始metrics.json中的positive_targets_seen字段来自节流进度写入，完成时可能仍是旧值；最终步数与history.jsonl均为1000，每步512个完整目标，因此实际处理512000个正例。保留原始文件，未因日志字段问题重跑训练。",
              "已完成的前向/梯度接入检查见baseline_provenance.md与preflight.json。此前数据准备的反向曝光断言中止及报告生成的NumPy JSON序列化错误均已记录；没有据此改split或重启完整训练。"]
    lines += ["", "## 已发表结果：独立的测试参考", "",
              "以下为MM-GRAPH Table5的已发表测试MRR(%)；本地上表是验证结果，不能据两表大小宣布超过论文。", "",
              "|方法／特征|Sports test MRR|Cloth test MRR|", "|---|---:|---:|",
              "|MLP / CLIP|28.22 ± 0.09|21.10 ± 0.04|",
              "|GCN / CLIP|31.38 ± 0.08|22.28 ± 0.05|",
              "|MMGCN / CLIP|31.96 ± 0.10|22.20 ± 0.05|",
              "|BUDDY / CLIP|31.55 ± 0.13|23.44 ± 0.26|",
              "|SAGE / CLIP|33.83 ± 0.08|24.58 ± 0.18|", "",
              "来源：https://arxiv.org/html/2406.16321v2#S4.T5 。以已缓存原表为准。", "",
              "## 复现入口", "",
              "在项目根目录，使用具备Torch CUDA、DGL CPU、Numba、SciPy的环境：", "",
              "```powershell",
              "python research_extensions/mm_pair_sports_20261028/prepare_data.py",
              "python research_extensions/mm_pair_sports_20261028/structural_baselines.py",
              "python research_extensions/mm_pair_sports_20261028/run_campaign.py prepare",
              "python research_extensions/mm_pair_sports_20261028/run_campaign.py run",
              "python research_extensions/mm_pair_sports_20261028/summarize.py", "```", "",
              "训练遵守plan.json及首次启动写入budget.json的90分钟绝对预算、共享GPU锁和STOP_REQUESTED。不得重置预算以重跑完成试验。",
              "code_snapshot保留本轮源码，data_audit与baseline_provenance记录来源/接入差异；各runs子目录保留配置、检查点、轨迹和逐查询RR。",
              "交换配对损失、内容条件梯度规则、多种子和Cloth尚未在本轮运行，不归入现有方法收益。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise AssertionError(errors)


if __name__ == "__main__":
    main()
