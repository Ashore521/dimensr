# DimenSR 使用说明

DimenSR 是按项目中的“量纲约束 RNN-GP”设计方案实现的符号回归研究软件。输入自变量、因变量和物理量纲，输出显式表达式及 Pareto 候选模型。当前实现可以安装、运行和扩展；合成示例的成功不代表已经证明方法的新颖性或优于现有算法。

## 安装和运行

在仓库目录建立 Python 3.10 或以上版本的虚拟环境。Windows PowerShell：

```powershell
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
| `random_state` | 0 | 随机种子 |

候选池不足 50 个时会警告并报告实际数量，不复制表达式凑数。不同语法写法可能代数等价，唯一性指原始树结构而非代数化简后的等价类。

默认支持 `add, sub, mul, div, square, sin, cos`；还可启用 `neg, sqrt, exp, log`。对数和平方根使用真实数学定义，除零和溢出会使候选无效。若新数据超出模型定义域，预测会明确报错。

## 输出与数据划分

模型 JSON 包含最佳树、常数、SymPy 表达式、量纲、验证误差、可在验证集上求值的 Pareto 模型、每轮训练日志和完整搜索配置。它用于预测加载，不包含优化器恢复状态。

默认随机划分 80% 训练和 20% 验证。训练集用于 RNN 奖励、常数拟合、遗传选择和复杂度估计；搜索结束后，验证集用于选择最终表达式。测试集需额外保留。对重复测量、同一样本的多条记录或时间序列，应传入独立的验证 CSV，防止随机拆分造成泄漏。

## 研究定位

当前仓库把方案中的主要机制实现为可运行算法。具体数学定义、与原方案的区别及有限语法的覆盖范围见 [ALGORITHM.md](ALGORITHM.md)。Rademacher 项是局部扰动族的经验估计，不构成一般化误差上界。真正的贡献需要通过跨任务、跨噪声、足够随机种子的匹配预算实验确认；当前实测结果见 [VALIDATION.md](VALIDATION.md)。

