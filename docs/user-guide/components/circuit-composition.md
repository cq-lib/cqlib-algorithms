# 线路组合组件

本文档说明 `cqlib_algorithms.operators.composite` 中由基础算子组合而成的复合
算子。它们也可从 `cqlib_algorithms.operators` 导入；具体算子不从包顶层导出。
所有 `build_circuit()` 调用都返回新的 Cqlib `Circuit`，不执行采样、测量或后端
任务。组件只负责可复用的线路组合；算法承诺、执行策略和经典后处理仍由算法层负责。

## Hadamard 相位查询

`HadamardPhaseQuery(function, output_mask=None)` 构建未测量线路
`H**n -> S_f -> H**n`。`S_f` 完全复用 `PhaseOracle`，因此单输出函数和多输出
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

## Grover 迭代单元

`GroverOperator(oracle, state_preparation=None, reflection_qubits=None)` 构建一次
完整的 Grover 迭代

`Q = D_R O`，`O` 为搜索寄存器上的相位 Oracle。对选定量子比特集合 `R`，
`P_{0,R}` 是“`R` 中所有量子比特均为零”的子空间投影，
`S_{0,R} = 2P_{0,R} - I`，而 `D_R = A S_{0,R} A†`。返回线路的时间
顺序为 Oracle、`A†`、`S_{0,R}`、`A`。部分文献写作 `-Q`，两者只差一个
全局相位；本组件采用不带前置负号的约定，并保证矩阵意义上的精确全局相位。

`oracle` 必须是作用在搜索寄存器上的合法、可静态求逆的 Cqlib `Circuit`，量子比特
`q0` 为最低有效位。典型输入是 `PhaseOracle(function).build_circuit()`。振幅放大
假设酉 Oracle，因此包含测量、重置、动态控制等不可静态求逆操作的线路在构造时被
拒绝。非连续源量子比特 ID 按 `Circuit.qubits` 的位置映射到 `0..n-1`。

构造器仅验证输入是合法且可静态求逆的线路。它不会通过指数规模的矩阵
分析证明线路必然是对角相位 Oracle；输入是否满足相位 Oracle 语义由调用方负责。

不给出 `state_preparation` 时，`A` 只在 `reflection_qubits` 上作用 Hadamard 门。
当 `R` 是全部量子比特时，`P_{0,R}=|0...0><0...0|`，`D_R` 就是
`GroverDiffusion(n)` 的 `2|s><s|-I`。当 `R` 是真子集时，这是在未反射量子
比特各计算基分块内进行的部分扩散，其 `+1` 特征子空间维数为
`2**(n-|R|)`，不是单态反射。

给出 `state_preparation` 时，`A` 是作用于整个搜索寄存器的自定义线路。
仅当 `R` 包含所有量子比特时，`D_R` 才是关于 `A|0...0>` 的单态反射
`2A|0...0><0...0|A†-I`。当 `R` 是真子集时，`D_R` 是关于
`A range(P_{0,R})` 的子空间反射。`state_preparation` 必须可静态求逆且
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
