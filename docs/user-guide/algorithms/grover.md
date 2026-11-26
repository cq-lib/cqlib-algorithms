# Grover 搜索算法

`Grover` 在有限布尔函数的输入空间中放大被标记状态的测量概率。它是完整算法组件：
负责选择迭代次数、构造搜索线路、调用 `Sampler`，并从实际返回的 counts 计算搜索
成功率和最可能结果。

## 数学定义

设 $`N=2^n`$，其中 $`M`$ 个输入满足布尔谓词。均匀初态中被标记子空间的初始
概率为

```math
a=\frac{M}{N}=\sin^2\theta .
```

相位 Oracle 对被标记状态施加负号，扩散反射随后把状态在好、坏两个子空间张成的
平面内旋转。重复 $`k`$ 次后的总成功概率为

```math
P_k=\sin^2((2k+1)\theta).
```

批量请求没有显式指定 `iterations` 时，组件在首个连续最大值附近比较相邻非负整数，
选择成功概率最大且 Oracle 调用更少的整数。单解问题因而使用
$`O(\sqrt{N})`$ 次抽象 Oracle 调用。理论依据见
[原始搜索算法](https://doi.org/10.1145/237814.237866)与
[振幅放大的一般化](https://doi.org/10.1090/conm/305/05215)。

## 接口与组件组合

`Grover(iteration=..., sampler=...)` 接收组装好的一次迭代和采样器。
调用 `run(iterations=r, shots=s)` 后，算法从迭代组件取得初态制备，追加 `r` 次迭代，
一次采样并返回一个 `GroverResult`。组装时使用以下组件：

- `UniformSuperposition` 构造 $`H^{\otimes n}\vert 0\ldots0\rangle`$；
- `PhaseOracle` 根据标量函数或选定的向量输出奇偶值标记状态；
- `GroverOperator` 构造一次 Oracle 与扩散反射迭代；
- `Sampler` 对完整搜索寄存器进行有限-shot 测量。

单次搜索要求 `GroverOperator` 使用 `PhaseOracle` 或 `BitPatternPhaseOracle` 提供标记集合，
并对整个搜索寄存器执行反射。初态必须已完全绑定；参数化制备或拟设可先用
`UserCircuitPreparation.from_component(component, bindings, prefix=...)` 固定参数。
此入口必须显式指定迭代次数，以适配初始标记概率。
要批量执行均匀初态搜索，使用
`Grover(sampler).run([GroverRequest(...), ...])`，所有请求交给采样器一次执行，结果按输入
顺序返回。批量请求省略 `iterations` 时按标记比例选择次数。向量布尔函数还需给出
`output_mask`，标记条件是所选输出位的异或值。

`GroverResult` 保存按计算基整数索引的完整 `counts`、标记状态、实际迭代次数和
shots，并提供：

- `success_count` 与 `success_probability`；
- `modal_states`：并列最高频状态，不擅自打破统计平局；
- `modal_bitstrings`：以量子比特编号递增的 little-endian 布尔元组；
- `marked_modal_states`：最高频结果中满足谓词的状态。

## 边界

- 零解问题会在执行前被拒绝，因为标准 Grover 搜索无法返回不存在的解；
- 全部状态均为解时自动选择零次迭代，均匀测量的每个结果都成功；
- 自动迭代公式适用于标准均匀初态和完整扩散反射；使用自定义初态时，按初始成功
  振幅选择迭代次数。子空间反射可直接通过底层振幅放大组件构建；
- 结果判定完全来自 Sampler counts。有限 shots、噪声或不完整的硬件执行可能使
  最高频状态不是解，因此调用方应同时检查 `success_probability` 和
  `marked_modal_states`；
- 当前布尔函数以完整真值表表示，Oracle 综合具有指数级经典输入和预处理成本。
  $`O(\sqrt{N})`$ 是抽象黑盒查询复杂度，不包含该显式综合成本；
- 每条线路包含 `iterations` 次 Oracle 调用；多个 shots 会重复执行整条线路，不能
  把多-shot 任务描述为总共只调用这些次数。

## 单次搜索示例

两比特均匀初态有四个候选态。仅标记索引 2（`10`）时，一次迭代使该态的理想测量概率达到 1。

```python
from cqlib_algorithms import Grover
from cqlib_algorithms.operators import (
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
    UniformSuperposition,
)

target = BooleanFunction.from_marked_states(2, [2])
iteration = GroverOperator(
    oracle=PhaseOracle(target),
    state_preparation=UniformSuperposition(2),
)
grover = Grover(iteration=iteration, sampler=StatevectorSampler())
result = grover.run(iterations=1, shots=256)
print(result.counts)  # (0, 0, 256, 0)
```

## 批量搜索示例

```python
from cqlib_algorithms import Grover, GroverRequest
from cqlib_algorithms.operators import BooleanFunction, StatevectorSampler

requests = [
    GroverRequest(BooleanFunction.from_marked_states(2, [index]), shots=32)
    for index in (1, 3)
]
results = Grover(StatevectorSampler()).run(requests)
assert [r.modal_states for r in results] == [(1,), (3,)]
```
