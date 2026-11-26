# 线路组合组件

本文档说明 `cqlib_algorithms.operators.composite` 中由基础算子组合而成的复合
算子。下文组件均可从 `cqlib_algorithms.operators` 导入。
所有 `build_circuit()` 调用都返回新的 Cqlib `Circuit`，不执行采样、测量或后端
任务。组件只负责可复用的线路组合；算法承诺、执行策略和经典后处理仍由算法层负责。

## Hadamard 相位查询

`HadamardPhaseQuery(function, output_mask=None)` 构建未测量线路
$`H^{\otimes n}\rightarrow S_f\rightarrow H^{\otimes n}`$。$`S_f`$ 完全复用
`PhaseOracle`。构造时可直接传入该组件，也可传入布尔函数并由库内构造 Oracle；预配置 Oracle 的输出掩码须与显式 `output_mask` 一致。因此单输出函数和多输出
`output_mask` 奇偶归约的类型、端序与边界语义都与基础 Oracle 一致。该组件不判断
函数是否满足常/平衡或线性承诺，也不追加测量；Deutsch-Jozsa、
Bernstein-Vazirani 及后续同类查询算法可在其上实现各自的采样与经典判定。

这条现有组合链路同时就是 Deutsch-Jozsa/Bernstein-Vazirani 的 Oracle 构造能力：
`BooleanFunction` / `BooleanVectorFunction` 表示经典函数，`PhaseOracle` 将其变换为
相位 Oracle，`HadamardPhaseQuery` 再组成完整查询线路，因此不另设语义重复的
“DJ/BV Oracle Constructor”类型。需要保留显式输出寄存器的标准 XOR Oracle 时，
使用基础算子 `BitFlipOracle`。

```python
from cqlib_algorithms.operators import HadamardPhaseQuery, PhaseOracle
from cqlib_algorithms.operators.oracle import BooleanFunction

function = BooleanFunction.from_truth_table([False, True, True, False])
query = HadamardPhaseQuery(PhaseOracle(function))
circuit = query.build_circuit()
```

该 `query` 可直接传给 `DeutschJozsa(sampler).run(query, shots=16)` 或
`BernsteinVazirani(sampler).run(query, shots=16)`。调用方需按所选算法满足函数承诺。
替换的 Oracle 应实现同一个函数及输出掩码，生成完全绑定的同宽静态酉线路。

## 隐藏线性函数电路

`HiddenLinearFunctionCircuit(adjacency_matrix)` 根据非空、方形、对称的二进制矩阵
$`A`$ 构建未测量线路 $`H^{\otimes n}\rightarrow D_A\rightarrow H^{\otimes n}`$。
矩阵的第 $`i`$ 行和第 $`i`$ 列对应
量子比特 `i`；量子比特 `0` 是 Cqlib 计算基索引的最低有效位。对二进制向量
`x`，二次型定义为

```math
q_A(x)=\sum_iA_{ii}x_i+2\sum_{i\lt j}A_{ij}x_ix_j\quad(\mathrm{mod}\ 4)
```

中间对角线路满足 $`D_A\vert x\rangle=i^{q_A(x)}\vert x\rangle`$：对角元素 $`A_{ii}=1`$ 对应量子
比特 $`i`$ 上的 `S` 门，非对角元素 $`A_{ij}=1`$ 对应 `CZ(i,j)`。输入矩阵在构造
时复制为只读快照，后续修改调用方矩阵不会改变组件；每次 `build_circuit()` 都返回
新的线路。

从全零态运行并测量时，所有可能输出 `z` 都满足：对二进制零空间
`Ker(A)` 中的任意 $`x`$，有 $`q_A(x)=2z\cdot x\pmod 4`$。本组件只构建查询
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

$`Q=D_RO`$，$`O`$ 为搜索寄存器上的相位 Oracle。对选定量子比特集合 $`R`$，
$`P_{0,R}`$ 是“$`R`$ 中所有量子比特均为零”的子空间投影，
$`S_{0,R}=2P_{0,R}-I`$，而 $`D_R=AS_{0,R}A^\dagger`$。返回线路的时间
顺序为 Oracle、$`A^\dagger`$、$`S_{0,R}`$、$`A`$。部分文献写作 $`-Q`$，两者只差一个
全局相位；本组件采用不带前置负号的约定，并保证矩阵意义上的精确全局相位。

`oracle` 接收 `PhaseOracle` 或 `BitPatternPhaseOracle` 组件。
当搜索条件只涉及部分位时，按位模式生成迭代线路即可避免枚举自由位的赋值。显式读取 `marked_states` 会列出全部满足条件的索引。量子比特
$`q_0`$ 为最低有效位。振幅放大
假设酉 Oracle，因此包含测量、重置、动态控制等不可静态求逆操作的线路在构造时被
拒绝。非连续源量子比特 ID 按 `Circuit.qubits` 的位置映射到 $`0,\ldots,n-1`$。

构造器校验组件的寄存器宽度及其线路的静态可逆性。自定义 Oracle 可继承相应组件类，
生成线路须保持其声明的相位标记语义。

不给出 `state_preparation` 时，$`A`$ 只在 `reflection_qubits` 上作用 Hadamard 门。
当 $`R`$ 是全部量子比特时，$`P_{0,R}=\vert 0\ldots0\rangle\langle0\ldots0\vert `$，$`D_R`$ 就是
`GroverDiffusion(n)` 的 $`2\vert s\rangle\langle s\vert -I`$。当 $`R`$ 是真子集时，这是在未反射量子
比特各计算基分块内进行的部分扩散，其 $`+1`$ 特征子空间维数为
$`2^{n-\vert R\vert }`$，不是单态反射。

给出 `state_preparation` 时，可传入 `StatePreparation` 组件，生成作用于整个搜索寄存器的 $`A`$。
仅当 $`R`$ 包含所有量子比特时，$`D_R`$ 才是关于 $`A\vert 0\ldots0\rangle`$ 的单态反射
$`2A\vert 0\ldots0\rangle\langle0\ldots0\vert A^\dagger-I`$。当 $`R`$ 是真子集时，$`D_R`$ 是关于
$`A\mathrm{range}(P_{0,R})`$ 的子空间反射。`state_preparation` 必须可静态求逆且
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
oracle = PhaseOracle(function)

# 标准 Grover 迭代：Oracle + 全扩散
iteration = GroverOperator(oracle).build_circuit()

# 部分扩散：只在量子比特 0 和 2 上反射
partial = GroverOperator(oracle, reflection_qubits=(0, 2)).build_circuit()
```

更多反射与 Oracle 的数学约定见
[布尔 Oracle 与反射算子](oracle.md)。

## 相位估计线路

`PhaseEstimationCircuit(unitary, m)` 将受控酉幂和逆 QFT 组合为可复用的相位估计线路。标准 QPE 在系统寄存器接入初态，HHL 则在特征值倒数旋转两侧分别使用正向和逆向组件。酉输入类型、位序及可运行示例见[相位估计](../algorithms/phase-estimation.md)。

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
