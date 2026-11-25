# Pauli 测量规划与分组

本页说明如何把 Pauli 可观测量划分为可共同测量的组，并为每组生成局域测量基线路。
构造参数、返回类型与异常以源码 docstring 为准。

## 组件选择

- `PauliGroupingStrategy(mode="none" | "qwc")`：按输入顺序稳定分组。`none`
  不合并非恒等项；`qwc` 采用逐比特对易（QWC）分组。
- `CommutingObservableGrouping`：按完整的辛对易关系分组，只提供代数分组结果。
- `PauliMeasurementPlanner(circuit, terms, grouping="qwc", prefix="theta")`：接收
  `StatePreparation`、`Ansatz` 或已有线路，生成 `MeasurementPlan`。
- `plan.run(sampler, shots=..., parameter_bindings=..., seed=...)`：内部构建各组
  的基变换线路，一次提交全部测量组，按组顺序返回原始采样结果。

QWC 要求任意两个 Pauli 字符串在每个量子比特位置上相同，或至少有一个为恒等算符。
因此，同一组可以通过单比特基变换完成测量。完整对易的条件更宽松，例如
`XX` 与 `YY` 对易，但不满足 QWC，不能直接共用局域乘积测量基。

## 基变换与量子比特顺序

对每个待测 Pauli 项，规划器使用以下基变换：

- <img src="../../assets/formulas/inline/inline-4b68ab3847feda7d.svg" alt="X" style="vertical-align: -0.000pt;">：测量前施加 <img src="../../assets/formulas/inline/inline-44bd7ae60f478fae.svg" alt="H" style="vertical-align: -0.000pt;">；
- <img src="../../assets/formulas/inline/inline-18f5384d58bcb1bb.svg" alt="Y" style="vertical-align: -0.000pt;">：测量前依次施加 <img src="../../assets/formulas/inline/inline-eb8d072dc7420869.svg" alt="S^\dagger" style="vertical-align: -1.000pt;">、<img src="../../assets/formulas/inline/inline-44bd7ae60f478fae.svg" alt="H" style="vertical-align: -0.000pt;">；
- <img src="../../assets/formulas/inline/inline-bbeebd879e1dff69.svg" alt="Z" style="vertical-align: -0.000pt;"> 和 <img src="../../assets/formulas/inline/inline-a83dd0ccbffe39d0.svg" alt="I" style="vertical-align: -0.000pt;">：不施加基变换。

`PauliString.get_pauli(q)` 按线路中量子比特的插入顺序对应
`Circuit.qubits[q]`，量子比特的数值标识不作为位置。生成的线路不包含终端测量；
`plan.run` 内部通过 `SamplerRequest` 指定 `plan.qubits`，由 Sampler 在执行副本上追加测量。
需要检查或自行调度线路时，可使用 `plan.build_circuit(group_index)`。

## 使用示例

```python
from cqlib.qis import PauliString
from cqlib_algorithms.operators import (
    PauliMeasurementPlanner,
    StatevectorSampler,
    UniformSuperposition,
)

terms = [PauliString.from_str("IX"), PauliString.from_str("XI")]
plan = PauliMeasurementPlanner(UniformSuperposition(2), terms).plan()
results = plan.run(StatevectorSampler(), shots=256, seed=7)
for group, result in zip(plan.groups, results, strict=True):
    print(group.term_indices, group.basis, result.counts)
# (0, 1) ('X', 'X') {'00': 256}
```

规划器构建并复制输入组件；参数化输入保留其活跃符号，执行时才绑定。因此同一个计划
可以在不同参数点重复使用：

```python
import math
from cqlib.qis import PauliString
from cqlib_algorithms.operators import (
    PauliMeasurementPlanner,
    RealAmplitudesAnsatz,
    StatevectorSampler,
)

plan = PauliMeasurementPlanner(
    RealAmplitudesAnsatz(1, reps=0),
    [PauliString.from_str("Z")],
    prefix="trial",
).plan()
for angle in (0.0, math.pi):
    results = plan.run(
        StatevectorSampler(),
        shots=32,
        parameter_bindings={"trial_0": angle},
    )
    print(results[0].counts)
# {'0': 32}, then {'1': 32}
```

绑定必须恰好覆盖计划中的全部活跃符号，且值为有限实数。`prefix` 只为新建的制备或
拟设符号命名；组件显式配置的符号按自身命名规则保留。修改调用方的组件或绑定映射，
不会改变已保存的计划或已提交的线路。

若要测量 QFT、算术或其他酉组件从全零态制备的状态，可替换前例的初态来源：

```python
from cqlib_algorithms.operators import QFT, UserCircuitPreparation

prepared = UserCircuitPreparation(QFT(2).build_circuit())
plan = PauliMeasurementPlanner(prepared, terms).plan()
results = plan.run(StatevectorSampler(), shots=256, seed=7)
```

上述 QFT 线路没有活跃符号。自定义制备线路也可保留参数，在 `plan.run` 时完整绑定。

## 输入约束与能力边界

输入线路必须至少包含一个量子比特，并通过 Core 校验；线路及其嵌套组件不得包含测量、
重置或经典控制。所有 Pauli 项必须等宽且与线路宽度一致。空项列表合法并产生空计划。
纯恒等项不生成读出组，其索引记录在 `identity_indices` 中；相位和哈密顿量系数应由
Estimator 处理。

由自定义线路适配得到的状态制备保留辅助位；Pauli 项也必须覆盖
该完整寄存器。`run` 的输出为各测量组的计数，需结合 `groups`、`term_indices` 和
`identity_indices` 解释；空计划返回空列表，纯恒等项无需提交后端。

规划器不计算期望值，不选择 shots，也不执行读出校准、误差缓解、设备布局或纠缠联合
测量基综合。完整对易分组仅表示代数关系；只有同时满足 QWC 时，才能交给当前局域基
规划器共同测量。
