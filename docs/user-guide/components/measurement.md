# Pauli 测量规划与分组

本页说明如何把 Pauli 可观测量划分为可共同测量的组，并为每组生成局域测量基线路。
构造参数、返回类型与异常以源码 docstring 为准。

## 组件选择

- `PauliGroupingStrategy(mode="none" | "qwc")`：按输入顺序稳定分组。`none`
  不合并非恒等项；`qwc` 采用逐比特对易（QWC）分组。
- `CommutingObservableGrouping`：按完整的辛对易关系分组，只提供代数分组结果。
- `PauliMeasurementPlanner(circuit, terms, grouping="qwc")`：生成
  `MeasurementPlan`，其中包含分组元数据和各组的未测量基变换线路。

QWC 要求任意两个 Pauli 字符串在每个量子比特位置上相同，或至少有一个为恒等算符。
因此，同一组可以通过单比特基变换完成测量。完整对易的条件更宽松，例如
`XX` 与 `YY` 对易，但不满足 QWC，不能直接共用局域乘积测量基。

## 基变换与量子比特顺序

对每个待测 Pauli 项，规划器使用以下基变换：

- $X$：测量前施加 $H$；
- $Y$：测量前依次施加 $S^\dagger$、$H$；
- $Z$ 和 $I$：不施加基变换。

`PauliString.get_pauli(q)` 按线路中量子比特的插入顺序对应
`Circuit.qubits[q]`，量子比特的数值标识不作为位置。生成的线路不包含终端测量；
应通过 `SamplerRequest` 指定 `plan.qubits`，由 Sampler 在执行副本上追加测量。

## 使用示例

```python
from cqlib import Circuit
from cqlib.qis import PauliString
from cqlib_algorithms.operators.measurement import PauliMeasurementPlanner
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.sampler import SamplerRequest

preparation = Circuit(2)
preparation.h(0)
terms = [PauliString.from_str("IX"), PauliString.from_str("ZI")]
plan = PauliMeasurementPlanner(preparation, terms).plan()

for group_index, group in enumerate(plan.groups):
    basis_circuit = plan.build_circuit(group_index)
    request = SamplerRequest(basis_circuit, plan.qubits, shots=1000)
    result = StatevectorSampler().run([request])[0]
    print(group.term_indices, group.basis, result.counts)
```

## 输入约束与能力边界

输入线路必须至少包含一个量子比特，并通过 Core 校验；线路及其嵌套组件不得包含测量、
重置或经典控制。所有 Pauli 项必须等宽且与线路宽度一致。空项列表合法并产生空计划。
纯恒等项不生成读出组，其索引记录在 `identity_indices` 中；相位和哈密顿量系数应由
Estimator 处理。

规划器不计算期望值，不选择 shots，也不执行读出校准、误差缓解、设备布局或纠缠联合
测量基综合。完整对易分组仅表示代数关系；只有同时满足 QWC 时，才能交给当前局域基
规划器共同测量。
