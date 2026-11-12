# 线路组合组件

本文档说明 `cqlib_algorithms.operators.composite` 中由基础算子组合而成的复合
算子。它们也可从 `cqlib_algorithms.operators` 导入；具体算子不从包顶层导出。
所有 `build_circuit()` 调用都返回新的 Cqlib `Circuit`，不执行采样、测量或后端
任务。组件只负责可复用的线路组合；算法承诺、执行策略和经典后处理仍由算法层负责。

## Hadamard 相位查询

`HadamardPhaseQuery(function, output_mask=None)` 构建未测量线路
<img src="../../assets/formulas/inline/inline-678b057a76c6a024.svg" alt="H^{\otimes n}\rightarrow S_f\rightarrow H^{\otimes n}" style="vertical-align: -5.000pt;">。<img src="../../assets/formulas/inline/inline-292ad28eb29afab7.svg" alt="S_f" style="vertical-align: -5.000pt;"> 完全复用
`PhaseOracle`，因此单输出函数和多输出
`output_mask` 奇偶归约的类型、端序与边界语义都与基础 Oracle 一致。该组件不判断
函数是否满足常/平衡或线性承诺，也不追加测量；Deutsch-Jozsa、
Bernstein-Vazirani 及后续同类查询算法可在其上实现各自的采样与经典判定。

这条现有组合链路同时就是 Deutsch-Jozsa/Bernstein-Vazirani 的 Oracle 构造能力：
`BooleanFunction` / `BooleanVectorFunction` 表示经典函数，`PhaseOracle` 将其变换为
相位 Oracle，`HadamardPhaseQuery` 再组成完整查询线路，因此不另设语义重复的
“DJ/BV Oracle Constructor”类型。需要保留显式输出寄存器的标准 XOR Oracle 时，
使用基础算子 `BitFlipOracle`。

```python
from cqlib_algorithms.operators import HadamardPhaseQuery
from cqlib_algorithms.operators.oracle import BooleanFunction

function = BooleanFunction.from_truth_table([False, True, True, False])
query = HadamardPhaseQuery(function).build_circuit()
```

## 隐藏线性函数电路

`HiddenLinearFunctionCircuit(adjacency_matrix)` 根据非空、方形、对称的二进制矩阵
<img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 构建未测量线路 <img src="../../assets/formulas/inline/inline-25adb2506c201b2c.svg" alt="H^{\otimes n}\rightarrow D_A\rightarrow H^{\otimes n}" style="vertical-align: -3.000pt;">。
矩阵的第 <img src="../../assets/formulas/inline/inline-de7d1b721a1e0632.svg" alt="i" style="vertical-align: -1.000pt;"> 行和第 <img src="../../assets/formulas/inline/inline-de7d1b721a1e0632.svg" alt="i" style="vertical-align: -1.000pt;"> 列对应
量子比特 `i`；量子比特 `0` 是 Cqlib 计算基索引的最低有效位。对二进制向量
`x`，二次型定义为

<p align="center">
  <img src="../../assets/formulas/quadratic-form.svg" alt="The quadratic-form phase polynomial modulo 4.">
</p>

中间对角线路满足 <img src="../../assets/formulas/inline/inline-538401c258b9f3d2.svg" alt="D_A|x\rangle=i^{q_A(x)}|x\rangle" style="vertical-align: -4.000pt;">：对角元素 <img src="../../assets/formulas/inline/inline-a79cb618645ab6e9.svg" alt="A_{ii}=1" style="vertical-align: -3.000pt;"> 对应量子
比特 <img src="../../assets/formulas/inline/inline-de7d1b721a1e0632.svg" alt="i" style="vertical-align: -1.000pt;"> 上的 `S` 门，非对角元素 <img src="../../assets/formulas/inline/inline-ce4da43677b72153.svg" alt="A_{ij}=1" style="vertical-align: -5.000pt;"> 对应 `CZ(i,j)`。输入矩阵在构造
时复制为只读快照，后续修改调用方矩阵不会改变组件；每次 `build_circuit()` 都返回
新的线路。

从全零态运行并测量时，所有可能输出 `z` 都满足：对二进制零空间
`Ker(A)` 中的任意 <img src="../../assets/formulas/inline/inline-2d711642b726b044.svg" alt="x" style="vertical-align: -1.000pt;">，有 <img src="../../assets/formulas/inline/inline-b7f031ebceedad5f.svg" alt="q_A(x)=2z\cdot x\pmod 4" style="vertical-align: -4.000pt;">。本组件只构建查询
线路，不追加测量、Sampler 或经典解验证。

```python
from cqlib_algorithms.operators import HiddenLinearFunctionCircuit

adjacency_matrix = [
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
]
query = HiddenLinearFunctionCircuit(adjacency_matrix)
circuit = query.build_circuit()
```

构造器接受任意对称二进制矩阵。Bravyi、Gosset 与 Koenig 的浅线路量子优势结论
还要求问题图具有规定的二维局域结构，并依赖可并行的门调度；仅构造本组件不能据此
声称任意矩阵都具有常深度优势。理论背景见
[Quantum advantage with shallow circuits](https://arxiv.org/abs/1704.00690)。

## Grover 迭代单元

`GroverOperator(oracle, state_preparation=None, reflection_qubits=None)` 构建一次
完整的 Grover 迭代

<img src="../../assets/formulas/inline/inline-c9e811bb8b618c0f.svg" alt="Q=D_RO" style="vertical-align: -3.000pt;">，<img src="../../assets/formulas/inline/inline-c4694f2e93d5c4e7.svg" alt="O" style="vertical-align: -1.000pt;"> 为搜索寄存器上的相位 Oracle。对选定量子比特集合 <img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;">，
<img src="../../assets/formulas/inline/inline-bde2db4990707f0b.svg" alt="P_{0,R}" style="vertical-align: -5.000pt;"> 是“<img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;"> 中所有量子比特均为零”的子空间投影，
<img src="../../assets/formulas/inline/inline-42354451ccc4d747.svg" alt="S_{0,R}=2P_{0,R}-I" style="vertical-align: -5.000pt;">，而 <img src="../../assets/formulas/inline/inline-2f134cfea0b13e0e.svg" alt="D_R=AS_{0,R}A^\dagger" style="vertical-align: -5.000pt;">。返回线路的时间
顺序为 Oracle、<img src="../../assets/formulas/inline/inline-265dc6d72a56872a.svg" alt="A^\dagger" style="vertical-align: -0.000pt;">、<img src="../../assets/formulas/inline/inline-eb79126375827d63.svg" alt="S_{0,R}" style="vertical-align: -5.000pt;">、<img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;">。部分文献写作 <img src="../../assets/formulas/inline/inline-3bf919dc0076368c.svg" alt="-Q" style="vertical-align: -3.000pt;">，两者只差一个
全局相位；本组件采用不带前置负号的约定，并保证矩阵意义上的精确全局相位。

`oracle` 必须是作用在搜索寄存器上的合法、可静态求逆的 Cqlib `Circuit`，量子比特
<img src="../../assets/formulas/inline/inline-b8bbb3a7ebd17a45.svg" alt="q_0" style="vertical-align: -3.000pt;"> 为最低有效位。典型输入是 `PhaseOracle(function).build_circuit()`。振幅放大
假设酉 Oracle，因此包含测量、重置、动态控制等不可静态求逆操作的线路在构造时被
拒绝。非连续源量子比特 ID 按 `Circuit.qubits` 的位置映射到 <img src="../../assets/formulas/inline/inline-80df05cab2d1ce42.svg" alt="0,\ldots,n-1" style="vertical-align: -3.000pt;">。

构造器仅验证输入是合法且可静态求逆的线路。它不会通过指数规模的矩阵
分析证明线路必然是对角相位 Oracle；输入是否满足相位 Oracle 语义由调用方负责。

不给出 `state_preparation` 时，<img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 只在 `reflection_qubits` 上作用 Hadamard 门。
当 <img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;"> 是全部量子比特时，<img src="../../assets/formulas/inline/inline-6b18e13177ca9362.svg" alt="P_{0,R}=|0\ldots0\rangle\langle0\ldots0|" style="vertical-align: -5.000pt;">，<img src="../../assets/formulas/inline/inline-4044c8a85cc77c30.svg" alt="D_R" style="vertical-align: -3.000pt;"> 就是
`GroverDiffusion(n)` 的 <img src="../../assets/formulas/inline/inline-6a2fe82b1bb881f1.svg" alt="2|s\rangle\langle s|-I" style="vertical-align: -4.000pt;">。当 <img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;"> 是真子集时，这是在未反射量子
比特各计算基分块内进行的部分扩散，其 <img src="../../assets/formulas/inline/inline-c59dc4e44ff99288.svg" alt="+1" style="vertical-align: -2.000pt;"> 特征子空间维数为
<img src="../../assets/formulas/inline/inline-ea1e801eda3ecc49.svg" alt="2^{n-|R|}" style="vertical-align: -0.000pt;">，不是单态反射。

给出 `state_preparation` 时，<img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 是作用于整个搜索寄存器的自定义线路。
仅当 <img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;"> 包含所有量子比特时，<img src="../../assets/formulas/inline/inline-4044c8a85cc77c30.svg" alt="D_R" style="vertical-align: -3.000pt;"> 才是关于 <img src="../../assets/formulas/inline/inline-3b7df70cc118381d.svg" alt="A|0\ldots0\rangle" style="vertical-align: -4.000pt;"> 的单态反射
<img src="../../assets/formulas/inline/inline-280ee76a18e3b5fd.svg" alt="2A|0\ldots0\rangle\langle0\ldots0|A^\dagger-I" style="vertical-align: -4.000pt;">。当 <img src="../../assets/formulas/inline/inline-8c2574892063f995.svg" alt="R" style="vertical-align: -1.000pt;"> 是真子集时，<img src="../../assets/formulas/inline/inline-4044c8a85cc77c30.svg" alt="D_R" style="vertical-align: -3.000pt;"> 是关于
<img src="../../assets/formulas/inline/inline-f193dcec98af8966.svg" alt="A\operatorname{range}(P_{0,R})" style="vertical-align: -5.000pt;"> 的子空间反射。`state_preparation` 必须可静态求逆且
量子比特数与 Oracle 一致。

`reflection_qubits` 是搜索量子比特索引的可迭代对象，不允许为空、重复或越界，
布尔值不作为整数接受；构造时排序并冻结为只读元组，重复值直接拒绝而不是
静默去重。省略时选择全部搜索量子比特。

组件不执行迭代次数选择、测量统计或完整搜索算法；这些工作由上层 Grover
算法根据问题规模、解的数量和执行后端完成。

## 可运行示例

```python
from cqlib_algorithms.operators import GroverOperator
from cqlib_algorithms.operators.oracle import BooleanFunction, PhaseOracle

function = BooleanFunction.from_truth_table(
    [False] * 7 + [True]  # 在 3 量子比特上标记 |111>
)
oracle = PhaseOracle(function).build_circuit()

# 标准 Grover 迭代：Oracle + 全扩散
iteration = GroverOperator(oracle).build_circuit()

# 部分扩散：只在量子比特 0 和 2 上反射
partial = GroverOperator(oracle, reflection_qubits=(0, 2)).build_circuit()
```

更多反射与 Oracle 的数学约定见
[布尔 Oracle 与反射算子](oracle.md)。

## QSP 与 QSVT

`QuantumSignalProcessing` 构建单量子比特 SU(2) 信号处理序列。调用方可以直接给出
相位序列，也可以从满足奇偶性、有界性和可分解条件的多项式系数求解相位。
`ProjectorControlledPhase` 提供 QSVT 序列使用的投影控制相位。

`QuantumSingularValueTransformation` 将相位序列与 `BlockEncoding` 组合，对编码矩阵
的奇异值实施多项式变换。`RealValuedQSVT` 提供实多项式入口，并负责 QSP/QSVT
相位约定转换。组件返回线路，不执行输入态制备、测量或结果反演。

相位求解是数值过程。当前 Haah 谱因子分解在约 16 阶以上可能因共轭根配对条件数
恶化而拒绝输入；一般 QSVT 还会继承 `UnitaryMatrixOracle` 的两量子比特 KAK
边界和酉特征分解容差限制。调用方应把这些异常解释为当前综合路径的支持边界，而不是
自动放宽容差或假定目标多项式无效。

```python
import numpy as np

from cqlib_algorithms.operators import QuantumSingularValueTransformation
from cqlib_algorithms.operators.oracle import UnitaryDilation

encoding = UnitaryDilation(np.array([[0.25]], dtype=np.complex128))
operator = QuantumSingularValueTransformation(encoding, phases=(0.0, 0.0))
circuit = operator.build_circuit()
```
