# 量子贝叶斯推断 QBayesian

`QBayesian` 从调用方提供的联合分布制备线路出发，通过振幅放大、末端采样和证据筛选，
估计查询变量的条件分布。实现位于 `algorithms/qbayesian.py`，从
`cqlib_algorithms.algorithms` 导入。

## 数学定义与来源

设完全绑定的静态酉线路满足

\[
|\psi\rangle=A|0\cdots0\rangle=\sum_z a_z|z\rangle,
\qquad P(z)=|a_z|^2.
\]

变量名对应线路量子比特的插入位置。证据 `e` 是若干变量二进制取值的**合取**，
`Pi_e` 投影到所有证据都成立的计算基子空间。对查询变量 `Q`，目标为

\[
P(Q=q\mid e)=
\frac{\sum_{z:Q(z)=q,\,e(z)}|a_z|^2}
     {\sum_{z:e(z)}|a_z|^2}.
\]

未命名量子比特和未查询变量均被求和消去。输入允许复振幅；调用方负责保证线路所表示
的分布符合自己的模型，组件不读取条件概率表，也不验证网络分解。

证据相位 Oracle 与 Grover 迭代采用

\[
O_e=I-2\Pi_e,\qquad
G=(2|\psi\rangle\langle\psi|-I)O_e.
\]

线路时间顺序为 `A` 后接 `k` 个完整的 `G`。Oracle 用证据位上的 X、多控 Z、X
构建，反射复用已有 `GroverOperator`，作用于全部制备量子比特。若
`P(e)=sin(theta)^2`，理想接受概率为 `sin((2k+1)*theta)^2`；非零接受子空间内
的归一化条件态保持不变，因此可以汇总不同幂次的接受频数。
这一构造采用 [Low、Yoder 与 Chuang 的量子拒绝采样方法，arXiv:1402.7359v1](https://arxiv.org/abs/1402.7359v1)
以及 [Brassard 等的振幅放大恒等式，arXiv:quant-ph/0005055v1](https://arxiv.org/abs/quant-ph/0005055v1)。
本实现使用显式有限幂次序列，不实现论文的无界停止过程或其期望复杂度保证。

## 接口与输入约定

```text
QBayesian(state_preparation, variable_qubits, sampler)
build_circuit(evidence=None, *, grover_power=0, max_grover_power=1024)
infer(query, evidence=None, *, target_accepted=256, shots_per_round=256,
      amplification_powers=(0, 1, 2, 4, 8, 16, 32), max_rounds=128,
      max_grover_power=1024, max_total_shots=1_000_000,
      max_grover_queries=10_000_000)
```

- `state_preparation` 是至少含一个量子比特、完全绑定、静态酉且可逆的 Cqlib `Circuit`。
  允许 barrier；拒绝 measurement、reset、delay、经典数据、动态控制及嵌套的非酉操作。
  Core 当前不能静态求逆的结构也会被拒绝。
- `variable_qubits: Mapping[str, int]` 必须非空，名称为非空字符串，位置唯一且在范围内。
  位置是原线路 `circuit.qubits` 的索引，不是 `Qubit` 的数字标识；例如插入顺序
  `[Qubit(7), Qubit(2)]` 的位置 0 指向 `Qubit(7)`。布尔位置不作为整数接受。
- `sampler` 必须实现本库公共 `Sampler` 抽象契约。构造时复制线路及变量映射；
  `state_preparation` 属性返回独立副本，`variable_qubits` 属性为不可变映射。
- `query` 是非空、名称唯一、全部已知的有限序列，拒绝单独的字符串。
  返回元组的第 `i` 位对应 `query[i]`。
- `evidence` 是已知名称到整数 0/1 或 bool 的映射；`None` 和 `{}` 表示不条件化。
  查询和证据可以重叠，此时相应查询位在接受记录中固定为证据取值。
- `build_circuit` 每次返回独立、未测量的线路。空证据使有效 Grover 幂次为 0，直接返回 `A`。

整数预算和幂次拒绝 bool。除 `max_grover_power` 可以为 0、幂次可以为非负整数外，
预算均须为正整数。`amplification_powers` 必须是非空有限 `Sequence`，可包含重复值或全零；
不消耗任意迭代器。调用前会验证**整个序列**，即使首轮可能达到目标，也必须满足：

\[
\begin{aligned}
\operatorname{len}(powers)&\le max\_rounds,\\
\max(powers)&\le max\_grover\_power,\\
\operatorname{len}(powers)\,shots\_per\_round&\le max\_total\_shots,\\
\sum_k effective\_power_k\,shots\_per\_round&\le max\_grover\_queries.
\end{aligned}
\]

空证据的有效幂次均为 0，但提供的幂次仍须满足 `max_grover_power`。
超限直接抛出 `ValueError`，不会裁剪序列、隐式重试或改用态矢量选择幂次。
这些上限限制工作量，不保证上限内的任意线路都适合具体后端。

## 采样与结果

每个实际执行轮次提交 `shots_per_round` shots，测量查询和证据位置的排序并集。
Sampler 约定中，第 `i` 个请求量子比特对应字符串 `bits[-1-i]`。
完整原始结果先由公共 `validate_sampler_results` 核验，再按全部证据过滤，最后按查询顺序聚合。

累计接受数首次达到 `target_accepted` 时停止，保留终止轮的全部接受计数；因此接受数可以超过目标。
序列结束仍未达到目标则返回已有的部分估计。

`QBayesianResult` 是冻结、slotted 的 dataclass，映射均为复制后的不可变快照：

| 字段或属性 | 含义 |
| --- | --- |
| `query`, `evidence` | 查询顺序与规范化为 0/1 的证据 |
| `counts` | 稀疏 `Mapping[tuple[int, ...], int]` 接受频数，仅保存正计数 |
| `probabilities` | `counts / accepted`；没有接受样本时为 `None` |
| `accepted`, `attempted_shots`, `target_accepted` | 实际接受、实际采样和目标数量 |
| `rounds` | `QBayesianRound(grover_power, shots, accepted)` 的元组，记录有效幂次 |
| `status` | `QBayesianStatus.TARGET_REACHED` 或 `BUDGET_EXHAUSTED` |
| `num_grover_queries` | `sum(round.grover_power * round.shots)` |
| `num_state_preparation_queries` | `sum((2 * round.grover_power + 1) * round.shots)`，合计 `A` 和 `A†` |

有估计时，未出现的元组代表经验零；`probabilities=None` 表示没有估计。
零接受不证明 `P(e)=0`，放大后的接受率也不是原始证据概率。
组件不返回置信区间、逐 shot 历史或网络概率认证。公共结果构造会检查计数、概率、轮次与状态的一致性。

无效输入和 Sampler 契约错误使用英文 `TypeError`/`ValueError`；后端执行异常原样传播。
算法自身不分配态矢量或完整联合概率表。单轮线路大小随幂次与制备/Oracle 大小增长。
条件分布保持不变的推导要求理想执行；有噪声时结果只表示实际后选择经验分布。
`StatevectorSampler` 不提供 seed，因此这里不承诺有限 shots 结果可复现。

## 可运行示例：联合分布与证据放大

下面的基态索引按小端解释：`index = q + 2*e`。原始 `P(e=1)=0.25`，
`P(q=1 | e=1)=0.8`；一次 Grover 迭代将理想证据接受概率提高到 1。
有限 shots 的经验后验会波动。

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
amplified = inference.build_circuit({"e": 1}, grover_power=1)
print("Amplified circuit qubits:", amplified.num_qubits)
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
print("Grover queries:", result.num_grover_queries)
assert result.accepted >= 128
assert result.probabilities is not None
```

## 可运行示例：有限预算内没有接受样本

这个制备线路保持 `e=0`，而查询要求 `e=1`。调用严格执行三个 8-shot 轮次后返回无估计。
一般任务中同样的返回值只能说明给定预算未观察到证据。

```python
from cqlib import Circuit

from cqlib_algorithms.algorithms import QBayesian, QBayesianStatus
from cqlib_algorithms.operators.sampler import StatevectorSampler

inference = QBayesian(Circuit(2), {"q": 0, "e": 1}, StatevectorSampler())
result = inference.infer(
    ["q"],
    {"e": 1},
    target_accepted=1,
    shots_per_round=8,
    amplification_powers=(0, 1, 2),
)
print(result.status.value, result.accepted, result.attempted_shots)
print("Posterior:", result.probabilities)
assert result.status is QBayesianStatus.BUDGET_EXHAUSTED
assert result.accepted == 0 and result.attempted_shots == 24
assert result.probabilities is None
```

验证环境为 Python 3.11.15、Cqlib 2.0.0b6。项目源码保持 Python 3.10 兼容语法，
该 Core 版本自身要求 Python 3.11+。
