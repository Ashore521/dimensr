"""Verify saved acceptance predictions, then render all runs without dropping failures."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from acceptance_benchmark import read, write, sha, fingerprint, metric, ROOT
from dimensr import DimenSRRegressor
from dimensr.expression import Node
from dimensr.real_data import audit_tree
from dimensr.units import ZERO, unit

OUT = ROOT / "results/acceptance_v1"
SPLITS = ("test", "ood", "probe")


def finite(value):
    return value is not None and bool(np.isfinite(value))


def numeric_flags(audit):
    recovery = all(audit[s]["invalid"] == 0 and finite(audit[s]["nrmse"])
                   and 0 <= audit[s]["nrmse"] < 1e-6 for s in SPLITS)
    test = audit["test"]
    accurate = test["invalid"] == 0 and finite(test["r2"]) and test["r2"] > .999
    return dict(numerical_recovery=bool(recovery), accurate_iid=bool(accurate))


def finite_audit(audit):
    return all(audit[s]["invalid"] == 0 and finite(audit[s]["nrmse"])
               and finite(audit[s]["r2"]) for s in SPLITS)


def numeric_domain_gate(rows):
    return bool(rows) and all(r["status"] == "ok" and finite_audit(r["selected_audit"])
                              for r in rows)


def verify(out, rows, protocol_sha256):
    protocol = read(out / "protocol.json")
    assert protocol_sha256 == sha(out / "protocol.json")
    assert fingerprint() == protocol["source_hashes"]
    assert len(rows) == len(protocol["jobs"])
    assert set(r["id"] for r in rows) == {j["id"] for j in protocol["jobs"]}
    jobs = {j["id"]: j for j in protocol["jobs"]}
    for relative, expected in protocol["dataset_hashes"].items():
        assert sha(out / relative) == expected
    trees = models = 0
    for row in rows:
        assert all(row.get(key) == value for key, value in jobs[row["id"]].items())
        if row["status"] != "ok":
            continue
        case = protocol["cases"][row["case"]]
        data = np.load(out / "data" / f"{row['data_key']}_holdout.npz")
        fit = np.load(out / "data" / f"{row['data_key']}_fit.npz")
        units = [ZERO]*len(case["names"]) if row["variant"] == "no_units" else [unit(u) for u in case["units"]]
        target = ZERO if row["variant"] == "no_units" else unit(case["target"])
        scores = []
        assert row["candidates"] and len(row["candidates"]) == len(row["audit"])
        derived = []
        for candidate, saved_audit in zip(row["candidates"], row["audit"]):
            tree = Node.from_dict(candidate["tree"])
            audit_tree(tree, units, target, [ZERO])
            constants = np.array(candidate["constants"])
            assert np.isfinite(constants).all() and len(constants) == tree.n_constants
            assert tree.size <= 21
            recomputed = {split: metric(tree, constants, data, split) for split in SPLITS}
            for split in SPLITS:
                assert recomputed[split] == saved_audit[split]
            flags = numeric_flags(recomputed)
            assert all(saved_audit[key] is value for key, value in flags.items())
            derived.append(flags)
            pred = tree.evaluate(fit["X_val"], constants)
            error = np.sqrt(np.mean((pred-fit["y_val"])**2))/np.std(fit["y_train"])
            scores.append(error+.01*tree.size/21 if np.isfinite(error) else np.inf)
            trees += 1
        assert np.isfinite(scores).any()
        assert row["selected"] == int(np.argmin(scores))
        assert row["selected_audit"] == row["audit"][row["selected"]]
        assert row["pareto_any_recovery"] is any(a["numerical_recovery"] for a in derived)
        assert row["pareto_any_accurate_iid"] is any(a["accurate_iid"] for a in derived)
        if row["variant"] != "physo":
            model = DimenSRRegressor.load(out / "models" / f"{row['id']}.json")
            for split in ("test", "ood", "probe"):
                assert metric(model.best_.tree, model.best_.constants, data, split) == row["native_selected_audit"][split]
            assert model.diagnostics_["evaluations"] <= 4096
            models += 1
    result = dict(verified_models=models, verified_candidate_trees=trees,
                  planned_runs=len(protocol["jobs"]), statuses=dict(Counter(r["status"] for r in rows)),
                  all_dataset_hashes_match=True, algorithm_and_runner_hashes_match=True,
                  all_selection_indices_recomputed=True, all_holdout_metrics_recomputed=True,
                  all_numeric_success_flags_recomputed=True, selected_audits_match=True,
                  summary_protocol_hash_matches=True,
                  protocol_sha256=sha(out / "protocol.json"))
    write(out / "verification.json", result)
    return result


def wilson(k, n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return f"{k}/{n} [{100*(center-half):.0f}%, {100*(center+half):.0f}%]"


def count(rows, key):
    def success(r):
        if r["status"] != "ok": return False
        if key == "recovery": return r["selected_audit"]["numerical_recovery"]
        if key == "accurate": return r["selected_audit"]["accurate_iid"]
        if key == "native":
            return numeric_flags(r["native_selected_audit"])["numerical_recovery"]
        if key == "native_accurate":
            return numeric_flags(r["native_selected_audit"])["accurate_iid"]
        if key == "native_domain":
            return any(m["invalid"] for m in r["native_selected_audit"].values())
        if key == "domain": return any(r["selected_audit"][s]["invalid"] for s in ("test", "ood", "probe"))
        return r[key]
    return sum(bool(success(r)) for r in rows)


def median(rows, split="test", metric_name="nrmse"):
    values = [r["selected_audit"][split][metric_name] for r in rows if r["status"] == "ok"]
    valid = [v for v in values if finite(v)]
    return f"{np.median(valid):.4g} ({len(valid)}/{len(rows)})" if valid else "无有效预测"


def report(out):
    payload = read(out / "summary.json")
    rows = payload["results"]
    protocol = read(out / "protocol.json")
    verified = verify(out, rows, payload["protocol_sha256"])
    amendment = read(out / "scoring_amendment.json")
    assert amendment["protocol_sha256"] == payload["protocol_sha256"]
    assert amendment["adapter_sha256"] == sha(Path(__file__).with_name("score_acceptance.py"))
    for row in rows:
        run_path = out / "runs" / f"{row['id']}.json"
        assert sha(run_path) == amendment["run_hashes"][run_path.name]
        assert all(row[key] == value for key, value in read(run_path).items())
    verified["scored_rows_match_frozen_fit_records"] = True
    verified["bounded_scoring_adapter_hash_matches"] = True
    write(out / "verification.json", verified)
    repeat = read(out / "reproducibility.json")
    assert repeat["protocol_sha256"] == payload["protocol_sha256"]
    for name, expected in repeat["evidence_hashes"].items():
        assert sha(out / "repeat" / name) == expected
    main = [r for r in rows if r["variant"] in ("full", "physo")]
    full = [r for r in main if r["variant"] == "full"]
    clean = [r for r in full if r["noise"] == 0]
    noisy = [r for r in full if r["noise"] == .01]
    core = [r for r in clean if r["case"] in ("Feynman-I.12.1", "Feynman-I.14.4")]
    evaluated = sum(r["status"] == "ok" for r in full)
    metric_failures = sum(r["status"] == "ok" and any(
        not finite(r["selected_audit"][s][key]) for s in SPLITS for key in ("nrmse", "r2")) for r in full)
    primary_metrics = dict(clean_recovery=count(clean, "recovery"), clean_n=len(clean),
                           noisy_accurate=count(noisy, "accurate"), noisy_n=len(noisy),
                           core_recovery=count(core, "recovery"), core_n=len(core),
                           selected_domain_failure_runs=count(full, "domain"),
                           evaluated_runs=evaluated, planned_runs=len(full),
                           nonfinite_metric_runs=metric_failures)
    gates = dict(core_recovery=bool(core) and count(core, "recovery")==len(core),
                 suite_recovery=bool(clean) and count(clean, "recovery")/len(clean)>=.75,
                 noisy_accuracy=bool(noisy) and count(noisy, "accurate")/len(noisy)>=.75,
                 numeric_domains=numeric_domain_gate(full),
                 execution=bool(full) and all(r["status"]=="ok" for r in full),
                 units=bool(full) and all(r["status"]=="ok" and r["exported_target_unit_violations"]==0
                           and r["diagnostics"]["unit_audit"]["violations"]==0 for r in full),
                 independent_repeat=repeat["passed"], dso_direct_comparison=False)
    write(out / "acceptance.json", dict(primary_metrics=primary_metrics, gates=gates,
                                        native_metrics=dict(clean_recovery=count(clean,"native"),
                                                            core_recovery=count(core,"native"),
                                                            noisy_accurate=count(noisy,"native_accurate"),
                                                            domain_failures=count(full,"native_domain")),
                                        scope="pilot gates only; no claim of statistical superiority"))
    text = ["# DimenSR 0.2 验收与原版 PhySO 对照", "", "## Material Passport", "",
            "- Origin Skill: academic-research-suite / experiment-agent。",
            "- Origin Mode: run + validate；Origin Date: 2026-10-17–18；Version Label: acceptance_v1。",
            f"- Verification Status: ANALYZED（全量留出集指标重算）；{'VERIFIED' if repeat['passed'] else 'UNAVAILABLE'}（单案例独立重跑）。数值成功标志和共同选择索引已核验，未声称全量重复训练。",
            "- 算法版本：0.2.0 / a555f11；本次增加验收工具与证据，未按测试结果修改算法。", "",
            "## 结论与门槛", "",
            "本轮是按照原文方法改编的有限预算先导试验，不能视为原论文完整复现或统计排名。"
            "验收门槛在训练前冻结；未通过的项目保留为后续改进依据。", "",
            "| 项目 | 预设要求 | 实测 | 判定 |", "|---|---|---|---|"]
    gate_rows = [("基础物理恢复", "6/6", f"{count(core,'recovery')}/6", "core_recovery"),
                 ("无噪声全题最终数值恢复", "≥75%", f"{count(clean,'recovery')}/{len(clean)}", "suite_recovery"),
                 ("1% 噪声最终 IID R²>0.999", "≥75%", f"{count(noisy,'accurate')}/{len(noisy)}", "noisy_accuracy"),
                 ("最终模型数值域", "0 次无定义；全部可核验", f"{count(full,'domain')} 次域失败；已评估 {evaluated}/{len(full)}；{metric_failures} 次指标非有限", "numeric_domains"),
                 ("full 拟合执行", "全部成功", str(Counter(r['status'] for r in full)), "execution"),
                 ("full 保存候选量纲", "0 违例", "已递归核验" , "units"),
                 ("预选案例独立重跑", "非耗时状态和预测完全一致", repeat["job"], "independent_repeat"),
                 ("DSO 原版实跑", "完成", "旧依赖与本机环境不兼容", "dso_direct_comparison")]
    for label, requirement, actual, gate in gate_rows:
        verdict = "通过" if gates[gate] else "未完成" if gate == "dso_direct_comparison" else "未通过"
        text.append(f"| {label} | {requirement} | {actual} | {verdict} |")
    audited = sum(r["diagnostics"]["unit_audit"]["proposals"] for r in full if r["status"]=="ok")
    physical_audited = sum(r["diagnostics"]["unit_audit"]["proposals"] for r in full
                           if r["status"]=="ok" and r["case"].startswith("Feynman"))
    text += ["", f"full 的运行时审计覆盖 {audited:,} 次提案（含缓存重复），其中 {physical_audited:,} 次来自非平凡物理单位任务。"
             "保存候选还独立核对了叶节点与输入单位声明。未完成运行不计入已审计覆盖范围。"]
    text += ["", "独立重跑排除的字段仅为 elapsed_seconds、fit_seconds、wall_seconds。"
             "比较了全部非耗时模型/运行状态（含常数、候选集和搜索历史）以及 train/val/IID/OOD/probe 预测。"
             "原始重复模型保存在 `results/acceptance_v1/repeat/`，对比证据见 `reproducibility.json`；"
             "这是一例确定性检查，不能替代整个题库的重复验证。"]
    text += ["", "本轮结论：基础乘积/平方物理公式与带噪 IID 预测达到预设门槛；"
             "全题严格恢复、边界定义域和 DSO 直接对照仍未过关，不能写成完整验收通过。"
             "数值域检查覆盖预先声明的压力探针，其中零点可以在训练范围之外；"
             "域失败说明未满足本轮外推/边界要求，不表示训练区间内每次预测都无效。", ""]
    text += ["", "## 逐题主对照", "",
             "无噪声恢复要求原始树在 IID、OOD、固定零/边界探针上均 NRMSE<1e-6。"
             "噪声准确率使用干净 IID 测试 R²>0.999。方括号为各题 3 次重复的 Wilson 95% 区间；"
             "即使 3/3，也只有约 44%–100%，不足以证明稳定胜出。", "",
             "| 题目 | 方法 | 无噪声最终恢复 | 无噪声候选任一恢复 | 1% 噪声最终准确 | 1% 噪声候选任一准确 |",
             "|---|---|---:|---:|---:|---:|"]
    for name in protocol["cases"]:
        for method in ("full", "physo"):
            rs = [r for r in main if r["case"]==name and r["variant"]==method]
            a, b = [r for r in rs if r["noise"]==0], [r for r in rs if r["noise"]==.01]
            text.append(f"| {name} | {method} | {wilson(count(a,'recovery'),len(a))} | "
                        f"{count(a,'pareto_any_recovery')}/{len(a)} | {wilson(count(b,'accurate'),len(b))} | "
                        f"{count(b,'pareto_any_accurate_iid')}/{len(b)} |")
    text += ["", "主表是两方法共同的验证选择器（NRMSE+0.01×节点数/21）。"
             "DimenSR 原生混合复杂度选择器也保存并独立评分，不得把二者混为一个输出。", "",
             "| 方法/噪声 | 最终恢复 | 候选任一恢复 | 最终 IID 准确 | 最终域错误运行 | IID NRMSE 中位数（有效/计划） | OOD NRMSE 中位数（有效/计划） |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("full", "physo"):
        for noise in (0., .01):
            rs = [r for r in main if r["variant"]==method and r["noise"]==noise]
            text.append(f"| {method}/{noise:g} | {count(rs,'recovery')}/{len(rs)} | "
                        f"{count(rs,'pareto_any_recovery')}/{len(rs)} | {count(rs,'accurate')}/{len(rs)} | "
                        f"{count(rs,'domain')} | {median(rs)} | {median(rs,'ood')} |")
    text += ["", f"DimenSR 原生选择器无噪声恢复：{count(clean,'native')}/{len(clean)}。"
             "候选集和最终选中之间的差距是优选诊断；事后根据真值挑选候选不能作为最终模型的实用精度。", "",
             "原生优选输出的补充验收（与主表相同阈值）：", "",
             "| 指标 | 原生混合复杂度优选 | 共同节点数优选 |", "|---|---:|---:|",
             f"| 基础物理无噪声恢复 | {count(core,'native')}/{len(core)} | {count(core,'recovery')}/{len(core)} |",
             f"| 全题无噪声恢复 | {count(clean,'native')}/{len(clean)} | {count(clean,'recovery')}/{len(clean)} |",
             f"| 1% 噪声 IID 准确 | {count(noisy,'native_accurate')}/{len(noisy)} | {count(noisy,'accurate')}/{len(noisy)} |",
             f"| 出现数值域错误的运行 | {count(full,'native_domain')} | {count(full,'domain')} |", "",
             "## 符号判定和数值域", "",
             "| 方法/噪声 | 最终系数舍入 8 位后符号相等 | 最终舍入 2 位后差/比为常数 | 无法判定（8 位/2 位） |", "|---|---:|---:|---:|"]
    for method in ("full", "physo"):
        for noise in (0., .01):
            rs = [r for r in main if r["variant"]==method and r["noise"]==noise]
            exact = sum(r['status']=='ok' and r['selected_audit']['symbolic_8digits']['exact'] for r in rs)
            relaxed = sum(r['status']=='ok' and r['selected_audit']['symbolic_2digits']['constant_difference_or_ratio'] for r in rs)
            unknown = [sum(r['status']=='ok' and r['selected_audit'][key]['status'] != 'ok' for r in rs)
                       for key in ('symbolic_8digits', 'symbolic_2digits')]
            text.append(f"| {method}/{noise:g} | {exact}/{len(rs)} | {relaxed}/{len(rs)} | {unknown[0]}/{unknown[1]} |")
    text += ["", "该双精度舍入判定借鉴了上游思想，但不是逐行调用其含三角常数处理的比较函数。"
             "8 位舍入后相等不是原始浮点式精确相等的证明。代数化简可能抹掉除零点，因此严格数值门槛始终使用原始树。", ""]
    text += ["SymPy 使用普通符号，没有自动加入正值等额外假设；未能化简为零也不证明两式不等。"
             "数值恢复和符号判定分开保留，复杂恒等式需要进一步人工核验。", ""]
    symbolic = read(out / "symbolic_diagnostics.json")
    text += [f"独立评分适配器将每次符号判定限制为 {symbolic['timeout_seconds']:g} 秒；"
             f"去重后 {symbolic['unique_comparisons']} 次比较的状态：{symbolic['unique_statuses']}。"
             "超时/异常保留为无法判定并留在分母，不计恢复成功。数值指标不受符号判定超时影响。", ""]
    for r in full:
        if r["status"]=="ok" and any(r["selected_audit"][s]["invalid"] for s in ("test","ood","probe")):
            c = r["candidates"][r["selected"]]
            text.append(f"- 域失败 `{r['id']}`：`{c['expression']}`；原始树见对应 run 文件，不能仅凭简化式推断域。")
    text += ["", "## 失败定位与下一轮验收重点", "",
             "- **搜索与表达能力**：无噪声 full 的任一保存候选恢复数和最终恢复数相同（12/24），"
             "本轮没有出现已保存正确候选被最终选择器漏选的恢复案例。应先改进结构发现与候选保留，"
             "仅调最终优选权重无法找回本轮候选集里不存在的正确式；也不能排除搜索中曾出现后被筛掉的结构。",
             "- **局部拟合与外推**：Nguyen-5/7/9 的部分模型在 IID 上 R² 接近 1，仍不是目标公式。"
             "例如 Nguyen-9 seed 22 无噪声 IID NRMSE≈0.00211，OOD NRMSE≈31.53。"
             "应在新开发题上改进三角/对数结构发现，并用未参与调整的新题重新验收。",
             "- **有理式结构**：I.18.4 的 full 在两种噪声下均未达 IID R²>0.999；"
             "六次选中式全部在零点探针有无定义值。优先检查共同分母、分子分母联合变异及同量纲项组合的搜索效率。",
             "- **边界合法性**：I.10.7 三个无噪声选中式的随机 IID/OOD 误差接近机器精度，"
             "但原始树引入 1/v 或 1/v²，在 v=0 处无定义；真式此时为 m。"
             "下一轮应从事先声明的物理输入域构造训练可用的边界约束和原始树审计，"
             "不能把本轮测试探针直接移入训练后继续称作盲测，也不能仅用化简后的显示式掩盖除零。",
             "- **SD/Rademacher 证据**：本轮六次消融不足以支持其独立收益；"
             "完整方法与 no_sd 均 3/6 达到带噪 IID 准确门槛，no_rad 为 4/6。"
             "这些计数只描述当前小预算结果，需扩大任务/种子并对齐实际工作量后再判断模块贡献。", "",
             "## 消融与计算成本", "",
             "在 Nguyen-7 和 I.18.4、1% 噪声、三个种子上配对；每方法共 6 次，实际工作量不同。"
             "本轮只描述变化，不根据 6 次结果宣称 SD 或 Rademacher 已有独立统计收益。", "",
             "| 方法 | 最终 IID 准确 | 最终域错误 | IID NRMSE 中位数 | OOD NRMSE 中位数 | 墙钟中位秒（含失败） |", "|---|---:|---:|---:|---:|---:|"]
    for method in ("full", "no_rnn", "no_gp", "no_units", "no_sd", "no_rad"):
        rs = [r for r in rows if r["variant"]==method and r["noise"]==.01 and r["case"] in ("Nguyen-7","Feynman-I.18.4")]
        text.append(f"| {method} | {count(rs,'accurate')}/{len(rs)} | {count(rs,'domain')} | {median(rs)} | "
                    f"{median(rs,'ood')} | {np.median([r['wall_seconds'] for r in rs]):.1f} |")
    text += ["", "以下为相同六次消融条件下的结构诊断中位数，仅使用成功完成的运行。"
             "末代不同节点数的个数只是规模覆盖指标，不等于完整种群分布或理论复杂度。", "",
             "| 方法 | 有效运行 | 末代 NSGA-II 不同节点数 | 最终节点数 | 训练 NRMSE | 验证 NRMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in ("full", "no_sd", "no_rad"):
        rs = [r for r in rows if r["variant"]==method and r["noise"]==.01
              and r["case"] in ("Nguyen-7","Feynman-I.18.4") and r["status"]=="ok"]
        diagnostics = []
        for r in rs:
            model = read(out / "models" / f"{r['id']}.json")
            history = [h for h in model["history"] if h["phase"]=="nsga2"]
            c = r["candidates"][r["selected"]]
            diagnostics.append([history[-1]["unique_sizes"] if history else np.nan,
                                c["nodes"], c["train_nrmse"], c["validation_nrmse"]])
        values = " | ".join(f"{v:.4g}" for v in np.nanmedian(diagnostics, axis=0)) if diagnostics else "无 | 无 | 无 | 无"
        text.append(f"| {method} | {len(rs)}/6 | {values} |")
    text += ["", "| 主方法 | 状态 | 实际评估数中位数 | 提案数中位数 | 墙钟中位秒（全部运行） |", "|---|---|---:|---:|---:|"]
    for method in ("full", "physo"):
        rs = [r for r in main if r["variant"]==method]
        ok = [r for r in rs if r["status"]=="ok"]
        evalkey, propkey = ("evaluations","proposals") if method=="full" else ("rewarded_evaluations","sampled_proposals")
        med = lambda key: f"{np.median([r['diagnostics'][key] for r in ok]):.0f}" if ok else "无完成记录"
        text.append(f"| {method} | {dict(Counter(r['status'] for r in rs))} | {med(evalkey)} | {med(propkey)} | {np.median([r['wall_seconds'] for r in rs]):.1f} |")
    text += ["", "DimenSR 的评估数是唯一结构，PhySO 是得到正奖励的评估，含重复；"
             "PhySO 最多 32768 次原始采样。失败/超时纳入成功率分母，无法取得完成诊断的运行不纳入评估数中位数。"
             "最多 4 个并发任务，共享主机，不能据墙钟比值宣称速度优势。常数语义、优化器和网络宽度也不同；"
             "PhySO 本轮没有总常数优化调用数，DimenSR 的调用数保存在每次诊断中，这项成本尚未对齐。", "",
             "## 三个真实数据集的补充证据", "",
             "重新核验了 42 条原版记录、15 个原版保存模型、27 个优化版模型及 732 个候选；指标和原始基线未改变。"
             "这些测试集此前已用于开发比较，只作探索性补充。", "",
             "| 方法（测试 R² 均值） | Airfoil | Concrete | Yacht |", "|---|---:|---:|---:|",
             "| DimenSR 0.1 | 0.4239 | 0.4957 | 0.9900 |", "| 0.2 预设 41 节点 | 0.6734 | 0.8137 | 0.9899 |",
             "| Ridge | 0.4925 | 0.5812 | 0.6591 |", "| Random Forest | 0.8121 | 0.8556 | 0.9945 |", "",
             "0.2 在这三个数据集上仍均低于随机森林均值；更大预算伴随前两项提升。Concrete seed 0 仍有 −0.23029 MPa 的 41 节点模型预测，"
             "量纲正确不能代替全域非负保证。21/31/41 节点全部结果见 [优化报告](OPTIMIZATION_RESULTS.zh-CN.md)。", "",
             "## 统计审查与复现边界", "",
             "11/11 项方法学审查：", "",
             "| 项目 | 处理/限制 |", "|---|---|",
             "| Simpson 汇总反转 | 逐题、逐噪声报告；汇总仅描述 |",
             "| 生态谬误 | 结论限于预选方程，不外推所有科学问题 |",
             "| 选择偏差/Berkson | 小题库人为覆盖选择，承认不代表全基准 |",
             "| Collider | 不按成功与否过滤运行；有效误差中位数标明有效分母 |",
             "| 基础率忽略 | 成功数始终除以计划运行数 |",
             "| 均值回归 | 新合成试验与旧真实数据改进分开 |",
             "| 幸存者偏差 | 失败/超时完整保留 |",
             "| 多重比较 | 不做小样本显著性排名；正式扩展需 Holm 校正 |",
             "| 分析路径选择 | 协议/源码/数据冻结；报告全部消融与题目 |",
             "| 相关代替因果 | 不把不同预算的变化归因于单一模块 |",
             "| 反向因果 | 不从预测准确推断真实物理因果机制 |", "",
             f"重算核验：`{json.dumps(verified, ensure_ascii=False)}`。", "",
             "DSO 未实跑：固定项目依赖 TensorFlow 1.14、NumPy≤1.19、Numba 0.53.1；本机 Python 3.11/3.12 无兼容 TensorFlow 安装包，"
             "也未安装 WSL/Docker。安装探测和源代码压缩包哈希保存在 `provenance.json`。这项仍是正式验收缺口，不以 DimenSR 消融结果替代。", "",
             "完整方案及逐项来源：[验收方案](ACCEPTANCE_PROTOCOL.zh-CN.md)。原始证据：[结果目录](../results/acceptance_v1)。"]
    (ROOT / "docs/ACCEPTANCE_RESULTS.zh-CN.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps(dict(verified=verified, gates=gates, primary=primary_metrics), ensure_ascii=False))


if __name__ == "__main__":
    report(OUT)
