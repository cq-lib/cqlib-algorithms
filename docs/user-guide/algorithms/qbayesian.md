# 量子贝叶斯推断 QBayesian

`QBayesian` 根据联合概率分布和已知证据，估计查询变量的条件分布。
调用方提供联合分布制备线路、变量位置映射及 Sampler，算法通过振幅放大和采样完成推断。

## 使用示例

以下联合概率按 `index = q + 2*e` 排列。查询目标为证据 `e=1` 条件下变量 q 的分布，
其理论值为 `P(q=1 | e=1)=0.8`。

```python
import math

from cqlib_algorithms.algorithms import QBayesian
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import ArbitraryStatePreparation

joint_probabilities = [0.60, 0.15, 0.05, 0.20]
preparation = ArbitraryStatePreparation(
    [math.sqrt(value) for value in joint_probabilities]
).build_circuit("")
inference = QBayesian(preparation, {"q": 0, "e": 1}, StatevectorSampler())
result = inference.infer(
    ["q"],
    {"e": 1},
    target_accepted=128,
    shots_per_round=128,
    amplification_powers=(0, 1),
)
print("Posterior:", dict(result.probabilities or {}))
print("Status:", result.status.value)
print("Accepted / attempted:", result.accepted, result.attempted_shots)
```

## 接口与参数

```text
QBayesian(state_preparation, variable_qubits, sampler)
infer(query, evidence=None, *, target_accepted=256, shots_per_round=256,
      amplification_powers=(0, 1, 2, 4, 8, 16, 32), max_rounds=128,
      max_grover_power=1024, max_total_shots=1_000_000,
      max_grover_queries=10_000_000)
build_circuit(evidence=None, *, grover_power=0, max_grover_power=1024)
```

| 参数 | 含义 |
| --- | --- |
| `state_preparation` | 编码联合分布的 Cqlib `Circuit`，要求参数完全绑定、静态酉且可逆 |
| `variable_qubits` | 变量名到量子比特位置的映射，位置按输入线路的量子比特插入顺序从 0 编号 |
| `sampler` | 本库 `Sampler` 组件，如 `StatevectorSampler` |
| `query` | 查询变量名称序列，结果元组按此顺序排列 |
| `evidence` | 变量名到 0/1 或 bool 的映射；多个条件共同成立，`None` 或 `{}` 表示直接查询联合分布的边缘分布 |
| `target_accepted` | 目标接受样本数 |
| `shots_per_round` | 每轮采样次数 |
| `amplification_powers` | 各轮 Grover 迭代次数，0 表示直接采样制备线路 |
| `max_rounds`、`max_grover_power` | 轮次数与单轮放大幂次上限 |
| `max_total_shots`、`max_grover_queries` | 整个采样计划的采样次数与 Grover 查询次数上限 |

变量名称与位置分别唯一，查询和证据均使用映射中已定义的变量。查询与证据可重叠。
采样前检查整个幂次序列是否符合预算。`build_circuit` 返回指定放大次数的未测量线路。

## 返回结果

`infer` 返回 `QBayesianResult`。条件概率由符合证据的样本频数归一化得到。

| 属性 | 含义 |
| --- | --- |
| `probabilities` | 查询结果元组到条件概率的映射；接受数为 0 时为 `None` |
| `counts` | 各查询结果的接受频数 |
| `accepted`、`attempted_shots` | 累计接受数与实际采样次数 |
| `status` | `TARGET_REACHED` 表示达到目标接受数，`BUDGET_EXHAUSTED` 表示已完成给定采样计划 |
| `rounds` | 各轮的放大次数、采样次数及接受数 |
| `num_grover_queries`、`num_state_preparation_queries` | Grover 与状态制备线路的查询次数 |

达到目标后返回当前轮的全部接受样本；计划结束时返回已取得的统计结果。

## 参考文献

- [Low、Yoder、Chuang：Quantum Inference on Bayesian Networks](https://arxiv.org/abs/1402.7359v1)。
- [Brassard 等：Quantum Amplitude Amplification and Estimation](https://arxiv.org/abs/quant-ph/0005055v1)。
