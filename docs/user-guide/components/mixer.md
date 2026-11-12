# Mixer 组件

`cqlib_algorithms.operators.mixer` 提供受约束变分线路使用的一参数 Mixer。所有组件都
实现 `Ansatz`，只声明一个逻辑参数，并严格使用 `{prefix}_beta` 作为活跃符号。

## 数学与编码约定

两个 one-hot 位之间的交换生成元固定为

<p align="center">
  <img src="../../assets/formulas/exchange-generator.svg" alt="The one-hot exchange generator between basis values i and j.">
</p>

因此精确局部演化使用 <img src="../../assets/formulas/inline/inline-6ffdd3dca3ed4a50.svg" alt="R_{XX}(\beta)" style="vertical-align: -4.000pt;"> 后接 <img src="../../assets/formulas/inline/inline-6e808a32b2f3803d.svg" alt="R_{YY}(\beta)" style="vertical-align: -4.000pt;">。这一定义保持 Hamming weight，
且没有被忽略的全局相位。one-hot 值 `a` 对应局部量子比特 `a`；复合编码仍采用量子
比特 0 为最低有效位的 Core 约定。

必须区分两类完整 Mixer：

- simultaneous Mixer 是 <img src="../../assets/formulas/inline/inline-ba8e03eb1533a6c6.svg" alt="\exp(-i\beta\sum_kH_k)" style="vertical-align: -15.000pt;">。当 <img src="../../assets/formulas/inline/inline-0231a655bf13a98d.svg" alt="H_k" style="vertical-align: -3.000pt;"> 不对易时，其线路由显式
  `EvolutionStrategy` 决定，默认 `auto(1)` 是一阶乘积近似，不能解释为精确演化。
  实现以完整 XY exchange 为乘积公式的原子项，避免把 XX 和 YY 拆散后离开 one-hot
  子空间；支持 exact、auto、first-order 和 second-order，拒绝会打乱 Pauli 子项的
  randomized 模式。
- partitioned Mixer 是有序乘积 <img src="../../assets/formulas/inline/inline-8ce26653ddbe02ad.svg" alt="\prod_k\exp(-i\beta H_k)" style="vertical-align: -15.000pt;">。顺序是定义的一部分，所有
  局部块共享同一个 <img src="../../assets/formulas/inline/inline-f3f3804480e8551a.svg" alt="\beta" style="vertical-align: -3.000pt;">；它一般不等于对应 simultaneous Mixer。

实现依据是 Hadfield 等人的 *From the Quantum Approximate Optimization Algorithm to a
Quantum Alternating Operator Ansatz*（arXiv:1709.03489v2）中的 nearby-values、null-swap、
parity、color-parity 和 time-color 构造。

## 公共组件

| 组件 | 宽度 | 语义 |
| --- | ---: | --- |
| `NearbyValuesMixer(d, r, strategy=...)` | <img src="../../assets/formulas/inline/inline-18ac3e7343f01689.svg" alt="d" style="vertical-align: -1.000pt;"> | cyclic <img src="../../assets/formulas/inline/inline-454349e422f05297.svg" alt="r" style="vertical-align: -1.000pt;">-nearby simultaneous one-hot Hamiltonian |
| `RingMixer(d, strategy=...)` | <img src="../../assets/formulas/inline/inline-18ac3e7343f01689.svg" alt="d" style="vertical-align: -1.000pt;"> | <img src="../../assets/formulas/inline/inline-55f8fe51377aded4.svg" alt="r=1" style="vertical-align: -1.000pt;"> 的 nearby-values 预设 |
| `FullyConnectedMixer(d, strategy=...)` | <img src="../../assets/formulas/inline/inline-18ac3e7343f01689.svg" alt="d" style="vertical-align: -1.000pt;"> | <img src="../../assets/formulas/inline/inline-113c2bb85476812f.svg" alt="r=d-1" style="vertical-align: -1.000pt;"> 的 nearby-values 预设，保留循环移位重数 |
| `NullSwapMixer(d, null_value=..., strategy=...)` | <img src="../../assets/formulas/inline/inline-18ac3e7343f01689.svg" alt="d" style="vertical-align: -1.000pt;"> | <img src="../../assets/formulas/inline/inline-a316e5f4e0cd35bb.svg" alt="\sum_{a\ne\mathrm{null}}E(\mathrm{null},a)" style="vertical-align: -17.000pt;"> 的完整 simultaneous 演化 |
| `ParityRingMixer(d, repetitions=...)` | <img src="../../assets/formulas/inline/inline-18ac3e7343f01689.svg" alt="d" style="vertical-align: -1.000pt;"> | odd、even、必要时 closing 的精确有序交换层 |
| `ControlledBitFlipMixer(m, control_state=...)` | <img src="../../assets/formulas/inline/inline-7d7eadee71a7e564.svg" alt="m+1" style="vertical-align: -2.000pt;"> | 默认 NOR（全零）控制态上的 <img src="../../assets/formulas/inline/inline-a2a98fa9443f82b5.svg" alt="\exp(-i\beta X)" style="vertical-align: -4.000pt;"> |
| `ControlledNullSwapMixer(m, control_state=...)` | <img src="../../assets/formulas/inline/inline-7df413cd05f3cecc.svg" alt="m+2" style="vertical-align: -2.000pt;"> | 默认 NONE（全零）控制态上的精确 null/value 局部交换 |
| `ControlledSwapMixer(m, control_state=...)` | <img src="../../assets/formulas/inline/inline-7df413cd05f3cecc.svg" alt="m+2" style="vertical-align: -2.000pt;"> | 默认 NONE（全零）控制态上的精确 value/value 局部交换 |
| `ValueSelectivePermutationSwapMixer()` | <img src="../../assets/formulas/inline/inline-4b227777d4dd1fc6.svg" alt="4" style="vertical-align: -0.000pt;"> | 两位置、两值之间的精确四体跃迁 |
| `ValueIndependentPermutationSwapMixer(d)` | <img src="../../assets/formulas/inline/inline-f86b4d3007fdc561.svg" alt="2d" style="vertical-align: -1.000pt;"> | 两个 direct one-hot 寄存器之间不区分值的交换 |
| `ColorParityPermutationSwapMixer(n, repetitions=...)` | <img src="../../assets/formulas/inline/inline-12dcb271d25c745a.svg" alt="n^2" style="vertical-align: -1.000pt;"> | direct one-hot 排列的 color × parity 有序 Mixer |
| `TimeSwapMixer(...)` | 见下文 | 固定工作对和起始时间的绝对 one-hot 四体交换 |
| `TimeColorMixer(...)` | 见下文 | 按时间再按完整图 edge color 排序的调度 Mixer |
| `PartitionedMixer(n, parts)` | <img src="../../assets/formulas/inline/inline-1b16b1df538ba12d.svg" alt="n" style="vertical-align: -1.000pt;"> | 通用有序局部 Mixer 组合器 |

`NullSwapMixer` 是完整的 simultaneous Hamiltonian Mixer，不表示某一个 null/value 局部
交换。受控的两个类是用于组合完整问题 Mixer 的局部构件。

## 四体排列跃迁

`ValueSelectivePermutationSwapMixer` 的局部量子比特顺序为
<img src="../../assets/formulas/inline/inline-08874be49d402ea8.svg" alt="(u_i,v_j,u_j,v_i)" style="vertical-align: -5.000pt;">。其生成元只连接 <img src="../../assets/formulas/inline/inline-8fdd2282b959138b.svg" alt="|1100\rangle" style="vertical-align: -4.000pt;"> 与 <img src="../../assets/formulas/inline/inline-270a62f50e3ff316.svg" alt="|0011\rangle" style="vertical-align: -4.000pt;">，展开为系数
<img src="../../assets/formulas/inline/inline-50c4600018ccf787.svg" alt="\pm1/8" style="vertical-align: -4.000pt;"> 的八个互相对易 Pauli 四体项，因此线路在完整四量子比特空间上精确。

`ValueIndependentPermutationSwapMixer` 对每个无序值对依次应用上述精确块。不同值对
在 direct one-hot 可行子空间中作用于互不相交的两态扇区，所以该乘积在可行子空间内
等于 value-independent simultaneous 演化；对非法 bitstring 不声明二者相等。

`ColorParityPermutationSwapMixer` 使用 <img src="../../assets/formulas/inline/inline-c9291ec3623b9730.svg" alt="\mathtt{position}\times n+\mathtt{value}" style="vertical-align: -4.000pt;"> 映射量子比特。值对采用
确定性的 round-robin 完整图 edge coloring，位置采用 odd、even、必要时 closing 的
环形 parity 分层；执行顺序固定为 color 在外、parity 在内。

## 绝对时间调度 Mixer

处理时间 <img src="../../assets/formulas/inline/inline-6158f77c3833268f.svg" alt="p_i" style="vertical-align: -3.000pt;"> 必须为正整数，默认 horizon 为 <img src="../../assets/formulas/inline/inline-4459542fa3a66d99.svg" alt="\sum_i p_i" style="vertical-align: -15.000pt;">。工作 <img src="../../assets/formulas/inline/inline-de7d1b721a1e0632.svg" alt="i" style="vertical-align: -1.000pt;"> 拥有
<img src="../../assets/formulas/inline/inline-cd6cba2451831a44.svg" alt="\mathtt{horizon}-p_i+1" style="vertical-align: -3.000pt;"> 个 one-hot 量子比特，分别表示起始时间
<img src="../../assets/formulas/inline/inline-7052e1da7379bb25.svg" alt="0,\ldots,\mathtt{horizon}-p_i" style="vertical-align: -3.000pt;">。
`TimeSwapMixer` 精确连接

<p align="center">
  <img src="../../assets/formulas/time-swap-transition.svg" alt="(si,sj)=(t,t+pi)⟷(t+pj,t).">
</p>

`TimeColorMixer` 遍历时间 <img src="../../assets/formulas/inline/inline-accd4fcc676585f1.svg" alt="t=0,\ldots,\mathtt{horizon}" style="vertical-align: -3.000pt;">，再遍历工作对的确定性 edge color；每个颜色
内的工作对互不相交。无效的 <img src="../../assets/formulas/inline/inline-a7a7a6aca3cb85fd.svg" alt="t+p_i+p_j&gt;\mathtt{horizon}" style="vertical-align: -5.000pt;"> 局部块不会生成。

## 通用组合与 QAOA

`MixerApplication(mixer, qubits)` 描述一个局部 Mixer 及其位置映射。
`PartitionedMixer` 允许相邻应用重叠，并严格按输入顺序组合；每个
局部 Mixer 必须恰有一个参数并遵守共享 beta 命名契约。

`QAOA` 保留原有 `mixer_operator=Hamiltonian` 接口，同时新增 `mixer=Ansatz`。两者互斥：
线路 Mixer 已经定义了综合顺序，因此不能再传 `mixer_strategy`。线路 Mixer 必须与成本
寄存器同宽、声明一个参数，并为每层生成恰好 `{prefix}_{layer}_beta`。

```python
from cqlib_algorithms.algorithms import QAOA
from cqlib_algorithms.operators.mixer import ParityRingMixer

# estimator、optimizer 和 cost 由应用注入。
algorithm = QAOA(estimator, optimizer, mixer=ParityRingMixer(4), reps=2)
circuit = algorithm.build_circuit(cost)
```

## 边界

- Mixer 只构造参数化酉线路，不负责选择可行初态、成本函数或优化器。
- simultaneous 非对易误差由调用方选择的受支持 `EvolutionStrategy` 决定；
  randomized Pauli 子项顺序会破坏 exchange 块，因此构造时拒绝。
- 受控组件的 `control_state` 是小端整数；`None` 遵循论文的 NOR/NONE
  谓词，表示所有控制位为 0。显式整数仍可选择任意计算基控制态。
- direct one-hot 与 absolute one-hot 组件仅承诺从合法编码态出发保持相应可行子空间；
  对非法 bitstring 的作用不解释为问题级语义。
- 排列与调度组件的宽度分别按 <img src="../../assets/formulas/inline/inline-12dcb271d25c745a.svg" alt="n^2" style="vertical-align: -1.000pt;"> 和绝对时间寄存器规模增长，未引入辅助量子比特。
