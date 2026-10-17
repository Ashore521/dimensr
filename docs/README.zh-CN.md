# DimenSR 使用说明

DimenSR 是按项目中的“量纲约束 RNN-GP”设计方案实现的符号回归研究软件。输入自变量、因变量和物理量纲，输出显式表达式及 Pareto 候选模型。当前实现可以安装、运行和扩展；合成示例的成功不代表已经证明方法的新颖性或优于现有算法。

## 安装和运行

在仓库目录建立 Python 3.10 或以上版本的虚拟环境。Windows PowerShell：

```powershell
git clone https://github.com/Ashore521/dimensr.git
cd dimensr
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m dimensr demo --case force --config examples/quick_config.json
```

以上写法不需要修改 PowerShell 脚本执行策略。Linux/macOS 将 `.venv\Scripts\python.exe` 换为 `.venv/bin/python`。也可激活环境后使用 `dimensr` 命令。

默认演示以随机生成的质量和加速度为输入，寻找力的表达式。`kinetic`、`nguyen1`、`trigonometric` 提供另外三个测试问题。

## 输入你的数据

CSV 示例：

```csv
mass,acceleration,force
1.0,2.0,2.0
2.0,3.0,6.0
```

训练至少需要 5 行，实际研究需要更多独立观测。不能包含缺失值、无穷值或非数值数据。元数据示例见 `examples/force_metadata.json`，量纲按以下顺序填写：

| 顺序 | L | M | T | I | Theta | N | J |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 基本量 | 长度 | 质量 | 时间 | 电流 | 温度 | 物质的量 | 发光强度 |

例如速度为 `[1,0,-1,0,0,0,0]`，力为 `[1,1,-2,0,0,0,0]`，设计方案中的 `C/N` 为 `[-1,-1,3,1,0,0,0]`。量纲不包含单位换算比例，输入数值应事先转换到一致单位。

```bash
dimensr fit --data measurements.csv --target force --metadata examples/force_metadata.json --output outputs/model.json
dimensr predict --model outputs/model.json --data new_measurements.csv --output outputs/predictions.csv
```

同一列在不同文件中需保持同一单位。未提供元数据时所有变量按无量纲处理，不会自动猜测量纲。带物理单位的未知系数需要显式声明 `constant_units`，例如速度系数可用 `[[1,0,-1,0,0,0,0]]`。每个常数叶节点独立拟合；当前不支持多个位置共享同一个参数约束。

## 如何选择参数

`--config` 读取一个 JSON，可填写 `SearchConfig` 的任意字段；未填写的字段使用默认值。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `population_size` | 64 | 每轮生成数和进化种群上限 |
| `epochs` | 6 | RNN 播种与 GP 搜索轮数 |
| `gp_generations` | 2 | 每轮第一阶段 GP 代数 |
| `selection_generations` | 3 | 第二阶段 NSGA-II 代数 |
| `max_nodes`, `max_depth` | 15, 6 | 树节点数和深度上限 |
| `risk_fraction` | 0.3 | 策略更新的高奖励分位比例 |
| `min_candidates` | 50 | 候选池期望最少唯一表达式数 |
| `complexity_weights` | `[1,0.5,2]` | 节点数、算子种类、局部 Rademacher 项权重 |
| `selection_complexity_weight` | 0.01 | 验证选择中复杂度的权重 |
| `max_evaluations` | 5000 | 唯一树评价次数上限 |
| `selection_budget_fraction` | 0.25 | 为第二阶段预留的唯一树预算比例 |
| `variation_attempts` | 4 | GP 遇到重复结构时的最多尝试次数 |
| `elite_fraction` | 0.25 | 跨轮保留精英的比例；第一阶段也按规模保留代表 |
| `linear_seeds` | `true` | 加入量纲合法的单项、多项初始结构 |
| `domain_restarts` | 4 | 初值均无效时的多尺度常数初始化尝试数 |
| `polish_candidates`, `polish_max_nfev` | 8, 160 | 搜索后精调的模型数与每次系数优化上限 |
| `output_bounds` | `null` | 可选输出范围，例如 `[0,null]` 要求非负 |
| `domain_probe_count` | 32 | 用训练输入凸组合构造的无标签域检查点数 |
| `random_state` | 0 | 随机种子 |

候选池不足 50 个时会警告并报告实际数量，不复制表达式凑数。0.2 版对加法和乘法的左右操作数进行规范排序并同步调整独立常数位置，用于减少重复；不会消去分母或改变定义域。不同结构仍可能代数等价。

变量较多时需要相应增大节点上限。例如带 8 个独立系数和截距的普通线性模型，在当前树表示中需要 33 个节点。可对照 `max_nodes=31/41`，同时增加搜索预算；更大的树并不保证泛化更好。

默认支持 `add, sub, mul, div, square, sin, cos`；还可启用 `neg, sqrt, exp, log`。对数和平方根使用真实数学定义，除零和溢出会使候选无效。若新数据超出模型定义域，预测会明确报错。

`output_bounds` 必须来自领域知识，并使用输入目标数值的单位。例如强度非负可以设 `[0,null]`；不要从测试标签反推上下界。它参与候选接受和验证选择，`predict` 对违规预测报错，不会裁剪输出。有限检查点不能证明整个连续输入域均安全；对于类别变量或受特殊几何约束的输入，可将 `domain_probe_count` 设为 0。

## 输出与数据划分

模型 JSON 包含最佳树、常数、SymPy 表达式、量纲、验证误差、可在验证集上求值的 Pareto 模型、每轮训练日志和完整搜索配置。它用于预测加载，不包含优化器恢复状态。

0.2 日志增加了每阶段新增评价数、RNN 独立生成的有效/唯一候选数、当轮最佳表达式。诊断还记录各 GP 操作的提议数和新结构数、线性系数求解数、域修复数、优化终止状态、精调次数、Rademacher 计算次数。`max_evaluations` 仅限制唯一树数，精调和重试仍消耗时间。`constant_residual_calls` 在 0.2 中包括初始化、线性求解和候选检查时的训练预测调用，口径比 0.1 的非线性残差调用更宽；跨版本应同时查看墙钟时间。

默认随机划分 80% 训练和 20% 验证。训练集用于 RNN 奖励、常数拟合、遗传选择和复杂度估计；搜索结束后，验证集用于选择最终表达式。测试集需额外保留。对重复测量、同一样本的多条记录或时间序列，应传入独立的验证 CSV，防止随机拆分造成泄漏。

## 研究定位

真实数据验证包括 UCI 翼型噪声、混凝土抗压强度和船体阻力三个实测任务，按工况或配方分组。历史 0.1 结果保存在 `results/real_data`，对应提交 `5492122`；不能用新算法覆盖历史目录。安装实验依赖 `python -m pip install -e ".[experiments]"` 后，可运行：

```bash
python examples/compare_optimization.py run --output outputs/my_optimization --workers 4
python examples/compare_optimization.py verify --output outputs/my_optimization
```

优化对照配置位于 `examples/optimized_real_data_configs.json`，已记录结果位于 `results/optimization_v2`，以上命令在新目录复测。使用原先三组划分和种子，共比较 21、31、41 节点的 27 个模型，全部训练完才统一计算测试指标。`--workers` 控制独立拟合进程数，每个进程使用一个 PyTorch 线程；时间对比需考虑并发。每次完成的训练均可在中断后跳过；正在训练的单个模型需要重跑。算法源码或配置改变时必须用 `--output` 指定新目录。仓库记录了一次仅调度改变的修订，复用其已有结果需显式传 `--resume-dispatch`，此选项仍拒绝算法、数据或配置改变。

仓库中的实测模型使用经过换算/缩放的输入。部署时先将原始单位的特征乘以对应 `splits/*.json` 中的 `preprocessing.x_native_to_model`，再把预测乘以 `y_model_to_native`；不要重新计算这些因子。优化报告包含完整推理示例。普通用户自己调用 `fit` 时，`predict` 的输入尺度应与传给 `fit` 的数据一致。

历史协议与结果见 [固定实验协议](REAL_DATA_PROTOCOL.md)、[原始真实数据报告](REAL_DATA_RESULTS.zh-CN.md)；新版本对照见 [优化结果报告](OPTIMIZATION_RESULTS.zh-CN.md)。因为原测试集已被查看，本轮属于探索性复测，不能作为新的盲测结论。

当前仓库把方案中的主要机制实现为可运行算法。具体数学定义和语法覆盖范围见 [ALGORITHM.md](ALGORITHM.md)。0.2 的 Rademacher 项在系数 Jacobian 张成的局部输出空间内估计，能减少冗余参数化的影响；同时保存原始估计和归一化后的复杂度分量。它不代表完整结构类的容量，也不构成符号回归的一般化误差上界。SD 仍参与 NSGA-II 同层拥挤度调整，复杂度权重未通过测试集调优。
