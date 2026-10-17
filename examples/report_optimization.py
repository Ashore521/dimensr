"""Render the completed, independently verified optimization study as Chinese Markdown."""
import argparse
import json
import os
from pathlib import Path

import sympy as sp

from dimensr import DimenSRRegressor
from dimensr.real_data import DATASETS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(output, destination):
    protocol = read(output / "protocol.json")
    verification = read(output / "verification.json")
    summary = read(output / "summary.json")["summary"]
    rows = [read(p) for p in sorted((output / "runs").glob("*.json"))]
    lookup = {(r["dataset"], r["method"]): r for r in summary}
    variants = list(protocol["variants"])
    names = {"airfoil": "翼型噪声", "concrete": "混凝土强度", "yacht": "船体阻力"}
    labels = {"full": "原版 DimenSR", "ridge": "Ridge", "random_forest": "随机森林",
              **{v: f"优化版 {protocol['variants'][v]['max_nodes']} 节点" for v in variants}}
    primary = protocol["study_spec"]["primary_variant"]
    serial_count = sum(r.get("execution_workers", 1) == 1 for r in rows)
    concurrency = sorted({r.get("execution_workers", 1) for r in rows})
    dispatch_note = ("原协议、已完成结果和算法/配置/数据指纹保持不变，调度修订见 `dispatch_amendment.json`。"
                     if (output / "dispatch_amendment.json").exists() else "")

    def link(label, path):
        return f"[{label}]({os.path.relpath(path, destination.parent).replace(os.sep, '/')})"

    def score(name, method, part="test", metric="r2", dispersion=False):
        entry = lookup[name, method]
        if not entry["valid_test_runs"]:
            return "无有效结果"
        value = entry[part][metric]
        result = f"{value['mean']:.3f}"
        if dispersion and value["sample_std"] is not None:
            result += f" ± {value['sample_std']:.3f}"
        if entry["valid_test_runs"] != entry["runs"]:
            result += f" (有效 {entry['valid_test_runs']}/{entry['runs']})"
        return result

    text = ["# DimenSR 0.2 优化与真实数据复测", "",
            "本轮保留量纲约束、RNN 引导 GP、NSGA-II、SD 和 Rademacher 项，优化系数拟合、有效搜索、表达式容量与候选优选。"
            "原版结果和划分完整保留。本轮属于已查看测试集上的探索性复测，不是新的盲测，也不是逐项消融实验。", "",
            "## 主要结果", "",
            "以下为种子 0、1、2 的测试 R² 均值 ± 样本标准差。所有容量组均报告；事先指定 41 节点组为主配置，未按测试分数挑选配置或种子。", "",
            "| 方法 | 翼型噪声 | 混凝土强度 | 船体阻力 |", "| --- | ---: | ---: | ---: |"]
    for method in ["full"] + variants + ["ridge", "random_forest"]:
        text.append("| " + labels[method] + " | " + " | ".join(score(n, method, dispersion=True) for n in DATASETS) + " |")
    text += ["", "主配置相对原版的变化：", ""]
    for name in DATASETS:
        old, new = lookup[name, "full"], lookup[name, primary]
        if new["valid_test_runs"] != new["runs"]:
            text.append(f"- {names[name]}：仅 {new['valid_test_runs']}/{new['runs']} 次测试有效，不对不完整子集宣称总体提升。")
        elif new["valid_test_runs"]:
            delta = new["test"]["r2"]["mean"] - old["test"]["r2"]["mean"]
            reduction = 100 * (1 - new["test"]["rmse"]["mean"] / old["test"]["rmse"]["mean"])
            change = f"{delta:+.6f}" if abs(delta) < 0.001 else f"{delta:+.3f}"
            text.append(f"- {names[name]}：测试 R² {old['test']['r2']['mean']:.4f} → {new['test']['r2']['mean']:.4f}"
                        f"（变化 {change}）；RMSE 变化 {-reduction:+.1f}%。")
    text += ["", "上述变化包含算法更新、搜索空间改变以及扩大预算的共同影响。21 节点组保留原来的种群、轮数、代数和唯一树上限，"
             "但重试、系数求解、精调和真实评价数仍不同，不能称为严格等计算量实验。31/41 节点组设置相同的扩大搜索预算，"
             "用于观察容量影响。三个种子同时改变划分和搜索初始化，测试样本可能重叠；标准差不是置信区间。", "",
             "## 拟合、验证和测试", "",
             "| 数据集 | 方法 | 训练 R² | 验证 R² | 测试 R² | 测试 RMSE（原始单位） |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in DATASETS:
        for method in ["full"] + variants:
            text.append(f"| {names[name]} | {labels[method]} | {score(name,method,'train')} | "
                        f"{score(name,method,'validation')} | {score(name,method)} | {score(name,method,metric='rmse')} |")
    text += ["", "RMSE 单位依次为 dB、MPa、源数据阻力系数尺度。预测未裁剪；非有限预测不会按有效子集计算分数。", "",
             "## 实际计算开销与模型规模", "",
             "| 数据集 | 方法 | 节点数（seed 0/1/2） | 平均唯一树评价数 | 平均拟合秒数 |", "| --- | --- | --- | ---: | ---: |"]
    for name in DATASETS:
        for method in ["full"] + variants:
            entry = lookup[name, method]
            if entry["valid_test_runs"]:
                text.append(f"| {names[name]} | {labels[method]} | {' / '.join(map(str, entry['nodes']))} | "
                            f"{entry['mean_evaluations']:.0f} | {entry['mean_fit_seconds']:.1f} |")
    text += ["", f"原版为串行运行。本轮有 {serial_count} 个模型以单进程运行，记录的并发进程数为 {concurrency}；每个进程一个计算线程。"
             + dispatch_note +
             "时间包含系数拟合和模型保存等开销，受并发资源竞争影响，仅为实际耗时记录，不是受控加速比。"
             "`constant_residual_calls` 在新版本还包含初始化、仿射子问题和候选检查时的训练预测调用，跨版本口径较宽；完整计数保存在逐次运行文件中。", "",
             "## 实现的优化", "",
             "1. **常数拟合**：修复直接常数偏移引起的根号/对数/分母初值问题；无可行初值时做有限多尺度重启。"
             "对常数联合线性的公式直接求解，非线性公式保留有界最小二乘。GP 继承未改变部分的系数，并对少量精英追加精调。",
             "2. **搜索效率**：交换律规范化缓存、无效变异有限重试、跨轮保留多规模精英；加入符合相同语法与量纲的多项起始结构。"
             "第一阶段最多使用 75% 的唯一树预算，给 NSGA-II 留出空间。",
             "3. **策略学习**：RNN 同时学习算子/终端与合法子树量纲组合；GP 精英模仿仍与 on-policy 策略梯度分开。",
             "4. **候选优选**：预筛选保留精度、规模和 Pareto 代表；SD 继续调整 NSGA-II 同层拥挤度。大档案第一前沿用二维排序扫描提取。",
             "5. **Rademacher**：改为系数 Jacobian 所张成局部输出球上的经验估计，减少常规冗余参数化的影响。"
             "以 `sqrt(n)/rho` 归一化后参与混合复杂度，同时保存原始估计；它衡量局部独立输出方向，不是整个非线性结构类的泛化界。",
             "6. **物理约束与日志**：混凝土强度、船体阻力设置已知非负约束；只用训练输入构造 64 个凸组合检查点。"
             "记录实际新结构、RNN 候选质量、拟合失败原因、各 GP 算子效率和当前公式。", "",
             "21 节点组为 10 个神经轮次、每轮 3 代 GP、6 代 NSGA-II、4000 个唯一树上限；"
             "31/41 节点组为 24 个神经轮次、每轮 3 代 GP、12 代 NSGA-II、8000 个唯一树上限。"
             "所有组均采用两次常规系数初值、每次最多 40 次优化函数评价，另有最多 8 个精英、每次最多 160 次的精调。"
             "Rademacher 使用 64 组符号抽样。参数权重未按测试集调优。", "",
             "## 量纲与物理检查", "",
             "| 数据集 | 新版全部容量组唯一树审计数 | 量纲违规 | 主配置测试物理违规 |", "| --- | ---: | ---: | ---: |"]
    for name in DATASETS:
        subset = [r for r in rows if r["dataset"] == name and r["status"] == "ok"]
        entry = lookup[name, primary]
        text.append(f"| {names[name]} | {sum(r['unit_audit']['unique_evaluated_trees'] for r in subset)} | "
                    f"{sum(r['unit_audit']['violations'] for r in subset)} | {entry.get('test_physical_violations', '不可用')} |")
    text += ["", "树数为各次运行内部去重后求和，跨运行可以重复。翼型只检查数值域，没有声明 dB 必须非负。船体所有字段均无量纲，不能用于证明量纲模块收益。"
             "有限点检查不等于连续输入域的安全证明；量纲一致也不保证因果正确、正性或外推可靠。逐点违规索引保存在各运行文件的 `physical_checks` 中。", ""]
    for row in rows:
        if row["status"] != "ok" or not row["physical_checks"]["test"]["violations"]:
            continue
        split = read(output / "splits" / f"{row['dataset']}_{row['seed']}.json")["indices"]["test"]
        bad_indices = set(row["physical_checks"]["test"]["violating_original_rows"])
        bad = [(i, v) for i, v in zip(split, row["test_predictions"]) if i in bad_indices]
        text.append(f"- {names[row['dataset']]}、{labels[row['method']]}、seed {row['seed']}："
                    + "；".join(f"原始零基行号 {i}，预测 {v:.6f} {row['target_metric_unit']}" if v is not None
                                else f"原始零基行号 {i}，预测非有限" for i, v in bad) + "。")
    text += ["", "这些违规预测未裁剪、未删除，均保留在测试误差中。`predict` 会对违反声明范围的输入结果报错；"
             "实验直接评价原始树，以保留违规数值。需要全域非负保证的应用还应采用有依据的正性参数化或更强的域证明。", "",
             "## 候选模型与示例公式", "",
             "每个模型 JSON 都包含最终选择的表达式、原始树、常数及单位、验证有效的 Pareto 候选列表、配置与完整轮次日志。"
             "下列固定展示主配置的 seed 0，不挑选最佳种子。公式已将训练缩放还原到原始输入/输出数值单位，系数仅为展示舍入；预测请加载 JSON 原始树。", ""]
    for name in DATASETS:
        path = output / "models" / f"{name}_0_{primary}.json"
        if not path.exists():
            continue
        model = DimenSRRegressor.load(path)
        scales = read(output / "splits" / f"{name}_0.json")["preprocessing"]
        substitutions = {sp.Symbol(n): sp.Symbol(n) * factor for n, factor in
                         zip(model.feature_names_, scales["x_native_to_model"])}
        native = model.expression_.subs(substitutions, simultaneous=True) * scales["y_model_to_native"]
        text += [f"**{names[name]}**：{len(model.pareto_front_)} 个保存的 Pareto 候选；选中模型 {model.best_.tree.size} 节点。", "",
                 "```text", str(sp.N(native, 6)), "```", "",
                 "输入数值单位：" + "；".join(f"{n}: {u}" for n, u in zip(DATASETS[name]["names"], DATASETS[name]["native_units"])) + "。", ""]
    text += ["## 使用保存的实测模型", "",
             "实测模型 JSON 接收的是实验中经过 SI 换算和训练缩放的输入。新样本必须使用对应 `splits` 文件中已保存的因子，"
             "输出再乘回原始单位；不要对新样本重新估计缩放因子。以下配方仅为推理用法示例，不是新增验证数据。", "",
             "```python", "import json", "from pathlib import Path", "import numpy as np",
             "from dimensr import DimenSRRegressor", f'root = Path("{output.as_posix()}")',
             f'model = DimenSRRegressor.load(root / "models/concrete_0_{primary}.json")',
             'pre = json.loads((root / "splits/concrete_0.json").read_text(encoding="utf-8"))["preprocessing"]',
             "# cement, slag, fly_ash, water, superplasticizer, coarse, fine (kg/m³), age (day)",
             "X_native = np.array([[350., 100., 0., 180., 6., 1000., 800., 28.]])",
             'strength_MPa = model.predict(X_native * np.array(pre["x_native_to_model"])) * pre["y_model_to_native"]',
             "print(strength_MPa)", "```", "",
             "## 证据与复现", "",
             f"独立复核通过 {verification['verified_models']} 个模型的训练/验证/测试指标及测试预测，"
             f"并检查 {verification['verified_pareto_models']} 个保存的 Pareto 模型量纲；"
             f"记录的失败运行数为 {verification['failed_runs']}。旧结果文件指纹保持不变。", "",
             "- " + link("冻结协议", output / "protocol.json"),
             "- " + link("汇总指标", output / "summary.json"),
             "- " + link("独立复核", output / "verification.json"),
             "- " + link("全部逐次结果", output / "runs"),
             "- " + link("全部模型及候选", output / "models"),
             "- " + link("原始三数据集来源、许可和分组说明", Path(__file__).parents[1] / "docs" / "REAL_DATA_PROTOCOL.md"), "",
             "```bash", 'python -m pip install -e ".[dev,experiments]"',
             "python examples/compare_optimization.py run --output outputs/my_optimization --workers 4",
             "python examples/compare_optimization.py verify --output outputs/my_optimization",
             "python examples/report_optimization.py --output outputs/my_optimization --report outputs/my_optimization/report.md",
             "```", "",
             "本轮不包含 PhySO/DSO 的直接匹配预算比较，也没有把每项优化拆成单因素消融，不能独立归因给 SD、Rademacher 或 RNN。"
             "后续确认性研究应使用新的外部数据或预先规定的外层评估，并独立改变搜索种子以分离划分与算法随机性。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/optimization_v2"))
    parser.add_argument("--report", type=Path, default=Path("docs/OPTIMIZATION_RESULTS.zh-CN.md"))
    args = parser.parse_args()
    report(args.output, args.report)
