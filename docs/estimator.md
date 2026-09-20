# 期望值估计器实现

## StatevectorEstimator

`StatevectorEstimator` 是 `Estimator` 抽象基类的一个本地精确实现，位于
`cqlib_algorithms.operators.estimator`。它一次接收多个既有的 `EstimatorRequest`，
按输入顺序返回 `EstimatorResult`。空批次返回空列表。本组件不引入独立的 PUB
数据结构，也不提供提交任务或平台后端能力。

## 数学定义与输入约束

对已绑定参数的线路 `U` 和有限、Hermitian 的 `PauliString` 或 `Hamiltonian`，
结果为

```text
expectation = <0...0| U† O U |0...0>
```

量子位 0 是计算基索引的最低有效位。可观测量位置 `i` 对应输入线路
`circuit.qubits[i]` 的插入顺序；该规则对非连续或重排的 Core `Qubit` 标识也适用。
参数绑定必须恰好覆盖线路当前的 `used_symbols`，由 `EstimatorRequest.validate()`
核验，并在每次执行前再次核验。执行期间不能修改请求中的可变 Core 线路和可观测量。

`StatevectorEstimator` 仅接受 `shots=None`，正整数 shots 的请求会抛出
`ValueError`；有限采样由下文的 `SamplingEstimator` 处理。输入线路须仅包含酉门和可选 barrier；测量、
reset、delay、经典数据及动态控制被拒绝，检查覆盖嵌套的 `CircuitGate`。
零量子比特的 Pauli 恒等算子返回 1，空 Hamiltonian 返回 0。所有结果的
`variance=0.0` 是确定性估计值的统计方差，**不是**单次物理测量的可观测量方差。
因此本组件不生成采样噪声、误差棒或读出误差缓解结果。

## 示例

```python
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.protocols.estimator import EstimatorRequest

circuit = Circuit(1)
circuit.ry(0, Parameter("theta"))
request = EstimatorRequest(
    circuit,
    PauliString.from_str("Z"),
    {"theta": 0.5},
)
result = StatevectorEstimator().run([request])[0]
# result.expectation == cos(0.5), result.variance == 0.0
```

## Core 兼容性与复杂度

当前固定的 Cqlib Core `develop` 提交
`c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85` 的 `Statevector.expectation`
对含奇数个 Y 的 Pauli 返回错误符号。组件仍使用 Core
`Statevector.from_circuit` 制备精确态，但采用 Core 暴露的 Pauli 位掩码和相位，
按 `P|j> = phase × y_phase × (-1)^popcount(j & z_mask) |j xor x_mask>`
计算期望值。该兼容路径对每个 Pauli 项扫描 `2**n` 个幅度，不构造 `2**n × 2**n`
稠密算子。当前 Core Python binding 的 `Statevector.data` 返回一次幅度拷贝，
因此每请求的峰值额外空间为 `O(2**n)`；组件仅读取一次并复用于全部 Pauli 项。
对 `m` 个 Hamiltonian 项，计算时间为 `O(m × 2**n)`，不包含 Core 线路仿真成本。

请求契约先对 Hamiltonian 做严格的代数 Hermitian 核验。有限精度求和可能留下
微小虚部；仅当其绝对值不超过 `1e-12 × (1 + Σ|coefficient|)` 时才取实部，
否则抛出 `ValueError`，不会把显著复数结果伪装成实期望。

## 范围对应

- 56 态矢量估计器：本组件实现。
- 57 Pauli 期望值估计：精确路径和有限-shot 路径均接受 Pauli 与 Hamiltonian，
  不拆出重复的公共类。
- 58 批量估计：直接使用已冻结的 `Estimator.run(requests)`，不重复定义 PUB。
- 59 估计结果误差分析：本精确组件仅提供估计值方差 0；shot 标准误差和误差棒
  不属于精确估计器。

## SamplingEstimator

`SamplingEstimator` 是 `Estimator` 的有限-shot 实现。构造时注入任意显式继承
`Sampler` 的组件，所以它既不绑定本地模拟器，也不绑定具体硬件平台：

```python
from cqlib import Circuit
from cqlib.qis import PauliString
from cqlib_algorithms.operators.estimator import SamplingEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.estimator import EstimatorRequest

circuit = Circuit(1)
circuit.h(0)
request = EstimatorRequest(
    circuit,
    PauliString.from_str("X"),
    shots=1000,
)
result = SamplingEstimator(StatevectorSampler()).run([request])[0]
```

实现先将全局 Pauli phase 吸收到系数中，再按 `(num_qubits, x_mask, z_mask)` 稳定
合并重复项并去除精确零项。身份项直接加入常数；每个非身份 Pauli 项独立追加局域基
旋转：X 使用 H，Y 使用 Sdg 后接 H，Z 不旋转。生成的线路保持未测量状态，再由注入的
Sampler 负责建立私有目标测量。因此不会与 `SamplerRequest` 的测量职责冲突。

`EstimatorRequest.shots` 在这里定义为**单个 Estimator 请求的总测量预算**。若规范化后
有 `G` 个非身份项，则按稳定顺序均匀分配：前 `shots % G` 项各多一次；`shots < G`
会被拒绝。纯身份或空 Hamiltonian 不调用 Sampler，统计方差为 0。当前正确性优先路径
逐项独立测量，不重复实现 `PauliMeasurementPlanner` 已有的 QWC 分组。现有规划器
可以提供未测量的分组基线路，但分组估计还需要明确组内协方差和总 shots 分配合同；
在这些统计语义冻结前，本组件保留可独立验证的逐项路径。

对某项的完整原始 counts，位串最右字符对应线路插入顺序中的位置 0。令
`mask = x_mask | z_mask`，单次本征值为
`(-1)**popcount(int(bitstring, 2) & mask)`。两次及以上采样时，返回的
`EstimatorResult.variance` 是均值估计量方差的无偏样本估计：

```text
(1 - sample_mean**2) / (term_shots - 1)
```

各项使用独立采样，所以总方差是 `coefficient**2` 加权后的方差之和。任一项只有一次
采样时，方差返回 `None`，不会伪报为 0。标准误差可由 `sqrt(result.variance)` 取得。

SamplingEstimator 有意只接受完整原始频数，即每个 counts 总数必须等于分配给该项的
shots。Core `ExecutionResult` 本身允许过滤或后选择后的频数少于 shots，但在没有显式
条件分布契约时使用这种结果会静默改变目标期望值，因此本组件明确拒绝。拟概率、读出
误差缓解、平台提交与轮询也不在该组件范围内。
