# 量子隐形传态

`QuantumTeleportation` 实现 Bennett 等人提出的标准三量子比特隐形传态协议，并在
Cqlib 的本地理想纯态模拟器中执行真实的中间测量与经典条件纠正。它把未知单量子比特
态从 Alice 的输入位置转移到 Bob 的量子比特；物理载体本身并未从 Alice 移动到 Bob。

## 协议与量子比特约定

三个线路位置固定为：

- $`q_0=Q`$：待传输的输入态；
- $`q_1=A`$：Alice 持有的 Bell 对半边；
- $`q_2=B`$：Bob 持有的 Bell 对半边。

组件先复用 `GHZStatePreparation(2)` 在 `A, B` 上制备

```math
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}
```

随后在 `Q, A` 上执行 `CX(Q, A)` 和 `H(Q)`，形成 Bell 基分析。每个 shot 都从同一
预测量态的独立副本开始，并真实调用 Core 的 `Statevector.measure`：

1. 测量 `Q`，得到 `phase_bit`；
2. 测量 `A`，得到 `bit_flip_bit`；
3. 若 `bit_flip_bit == 1`，Bob 对 `B` 施加 X；
4. 若 `phase_bit == 1`，Bob 对 `B` 施加 Z。

测量二比特依固定 `(phase_bit, bit_flip_bit)` 顺序记录。理想协议中四种结果各以
$`1/4`$ 概率出现，并且完成相应纠正后 Bob 的约化态与输入态相同。

该过程消耗一对预共享 Bell 纠缠（1 ebit）和由 Alice 发送给 Bob 的两个经典比特。
Bob 必须收到经典结果后才能选择纠正，因此协议不允许超光速通信。Alice 的测量会破坏
原输入态，Bob 获得的是状态转移结果而不是额外副本，所以协议不违反不可克隆定理。

## 请求与执行

```python
import math

from cqlib_algorithms import QuantumTeleportation
from cqlib_algorithms.operators import ArbitraryStatePreparation

preparation = ArbitraryStatePreparation([1 / math.sqrt(3), 1j * math.sqrt(2 / 3)])
result = QuantumTeleportation().run(preparation, shots=32)
print(result.minimum_fidelity)  # approximately 1
print(result.all_succeeded)  # True
```

`run(state_preparation, *, shots, fidelity_tolerance=1e-12)` 接收 `StatePreparation` 组件，返回一个结果。组件使用前缀 `teleportation_input` 构建，其线路必须为固定的一量子比特 Cqlib `Circuit`。算法通过
`UserCircuitPreparation` 对线路按插入位置进行快照和连续编号，因此支持非连续的原始
量子比特标识，并保留全局相位。线路不得包含活跃参数、测量、reset、delay、经典控制
或嵌套的动态操作，而且必须能被 Core 展开并静态求逆。

批量入口 `run(requests)` 接收 `QuantumTeleportationRequest`，每项请求均可直接传入
状态制备组件，按顺序返回结果列表。沿用上例的 `preparation`：

```python
from cqlib_algorithms import QuantumTeleportationRequest
from cqlib_algorithms.operators import BasisStateInitialization

requests = [
    QuantumTeleportationRequest(preparation, shots=8),
    QuantumTeleportationRequest(BasisStateInitialization(1, 1), shots=16),
]
results = QuantumTeleportation().run(requests)
print([item.all_succeeded for item in results])  # [True, True]
```

组件在请求构造时只构建一次，请求保存固定的 `StatePreparation` 和独立的
`preparation_circuit` 快照。自定义制备线路通过 `UserCircuitPreparation(circuit)` 包装后传入。
调用方应在请求存续期间保持请求中保存的线路不变，执行前会重新校验。空批次返回空列表。

这里的“未知”是协议语义：测量结果和 Bob 的纠正规则不读取、估计或传送输入态的经典
振幅。由于本组件还要在模拟器内计算验证保真度，调用方仍须提供能够制备该态的线路，
模拟器也必然能够访问这一线路；这不等同于现实协议中 Alice 或 Bob 获得了未知态的
经典描述。

`build_premeasurement_circuit(state_preparation)` 返回完成输入制备、Bell 对制备和
Bell 分析门之后、尚未测量的独立三量子比特线路，便于线路检查和教学展示。实际执行
不会用相干控制或延迟测量线路替换经典通信步骤。

## 结果与保真度

`QuantumTeleportationResult` 保存：

- `measurements`：每个 shot 的 `(phase_bit, bit_flip_bit)`；
- `fidelities`：每次纠正后 Bob 约化态相对输入纯态的保真度；
- `bell_counts`、`bell_probabilities`：固定 `00, 01, 10, 11` 顺序的经验统计；
- `minimum_fidelity`、`mean_fidelity`：逐 shot 保真度汇总；
- `successful_shots`、`success_rate`、`all_succeeded`：以
  $`\mathtt{fidelity}\geq1-\mathtt{fidelity\_tolerance}`$ 判断的数值验收结果。

每个 shot 完成测量和纠正后，组件把三量子比特纯态转换为 Core `DensityMatrix`，调用
`partial_trace([2])` 取得 Bob 的一量子比特约化态，再通过
`state_fidelity_pure_mixed` 与输入纯态比较。仅对绝对值不超过 `1e-12` 的边界浮点
越界进行 `[0, 1]` 夹取；更大的越界会作为错误拒绝，不通过放宽容差掩盖问题。
`fidelity_tolerance` 只决定数值成功阈值，不代表统计置信度、信道噪声预算或安全界。

## 范围与边界

本实现是真实执行测量—经典比特—条件纠正语义的理想本地模拟，而不是以下能力：

- 不提交动态线路到远程后端或 QPU；
- 不建模噪声、退相干、纠缠分发损耗或经典通信延迟；
- 不做量子态层析、误差缓解或容错编码；
- 不传输物质、经典描述或可复制的未知态；
- 不把模拟器内部的过程解释为现实中的空间分离实验。

因此该组件适合协议验证、教学和理想算法组合。真实网络实现还需要纠缠分发、同步、
认证经典信道、设备控制以及适当的噪声和安全模型。

## 理论来源

- C. H. Bennett, G. Brassard, C. Crepeau, R. Jozsa, A. Peres, and
  W. K. Wootters, *Teleporting an unknown quantum state via dual classical
  and Einstein-Podolsky-Rosen channels*, Physical Review Letters 70,
  1895–1899 (1993), DOI: 10.1103/PhysRevLett.70.1895.
- W. K. Wootters and W. H. Zurek, *A single quantum cannot be cloned*,
  Nature 299, 802–803 (1982), DOI: 10.1038/299802a0.
