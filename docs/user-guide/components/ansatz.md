# Ansatz 与 QAOA 演化层

本页用于选择和组合已实现的变分线路。所有具体组件实现公共 `Ansatz` 协议，每次
构建返回独立 `Circuit`；构造参数和异常以源码 docstring 为准。

## 公共约定

- 公共构建流程统一校验 prefix、组件配置和生成的 Core 线路。
- `num_qubits` 必须是大于零的 Python `int`，`bool` 不作为整数接受。
- `reps` 必须是非负 Python `int`。`reps=0` 是合法退化配置。
- `prefix` 必须是字符串，用于为新建的符号参数命名。
- TwoLocal 系列默认纠缠拓扑为 `EntanglementTopology.linear()`；其他组件的默认值
  在各自章节说明。
- TwoLocal 系列的参数顺序为“旋转层、量子比特、该量子比特上的旋转门位置”。

## TwoLocal 系列

### `TwoLocalAnsatz`

```python
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib.circuit.gates import StandardGate
from cqlib_algorithms.operators import TwoLocalAnsatz

ansatz = TwoLocalAnsatz(
    4,
    reps=2,
    rotation_gates=(StandardGate.RY, StandardGate.RZ),
    entanglement_gate=StandardGate.CX,
    entanglement=EntanglementTopology.circular(),
)
circuit = ansatz.build_circuit("theta")
```

它直接组合并委托 Core `TwoLocal`。旋转门只允许未绑定的 `RX`、`RY`、`RZ`，
纠缠门只允许未绑定的 `CX`、`CY`、`CZ`。每个 repetition 包含一个旋转层和一个
纠缠层；默认在末尾增加一个旋转层。参数数量为

```text
num_qubits * len(rotation_gates) *
(reps + (0 if skip_final_rotation_layer else 1))
```

空 `rotation_gates` 合法，此时线路可以只包含纠缠门。

### `RealAmplitudesAnsatz`

`RealAmplitudesAnsatz` 将 TwoLocal 固定为 `RY` 旋转与 `CX` 纠缠。所有固定参数下
的门矩阵均为实数。参数数量为
`num_qubits * (reps + (not skip_final_rotation_layer))`。

### `EfficientSU2Ansatz`

`EfficientSU2Ansatz` 将每个单比特参数块固定为依次执行 `RY`、`RZ`，纠缠门固定为
`CX`。参数数量为
`2 * num_qubits * (reps + (not skip_final_rotation_layer))`。

### `HardwareEfficientAnsatz`

`HardwareEfficientAnsatz` 是本库冻结的一种数字门模型实现：每个单比特参数块为
`RZ`、`RX`、`RZ` Euler 序列，之后按 Core 拓扑执行 `CX`、`CY` 或 `CZ` 纠缠。
参数数量为
`3 * num_qubits * (reps + (not skip_final_rotation_layer))`。

“硬件高效拟设”并没有唯一线路定义；这个组件不会自动查询设备、选择本征门或执行
拓扑映射。调用方应传入与目标设备相符的 `EntanglementTopology` 和纠缠门，后续仍需
经过编译和映射流程。

## `TwoQubitUnitaryAnsatz`

该组件提供固定宽度为 2 的局域变分块，适合用作更大线路中的可训练双量子比特层。
它有两个相互独立的选项：`parameterization` 选择低深度 `simple` 或表达能力更强的
`general`，`structure` 选择矩阵为实数的 `real` 或允许复数振幅的 `complex`。

| 模式 | 参数数 | 固定纠缠结构 | 表达边界 |
| --- | ---: | --- | --- |
| simple + real | 2 | `CX(0 -> 1)` | 两个 RY 与一个 CX 组成的低深度实矩阵子族 |
| simple + complex | 2 | `CX(0 -> 1)` | RX、RY 与一个 CX 组成的低深度复矩阵子族 |
| general + real | 6 | 两个 `CX(1 -> 0)` | 覆盖连通的 `SO(4)` 实正交矩阵族 |
| general + complex | 15 | `CX(1 -> 0)`、`CX(0 -> 1)`、`CX(1 -> 0)` | 在忽略全局相位时覆盖一般双量子比特酉变换 |

```python
from cqlib_algorithms.operators import TwoQubitUnitaryAnsatz

block = TwoQubitUnitaryAnsatz(
    parameterization="general",
    structure="complex",
)
circuit = block.build_circuit("unitary")
```

参数按门在线路中出现的顺序命名为 `unitary_0`、`unitary_1` 等。量子比特 0 遵循
Cqlib 的小端约定，是计算基 `|q1 q0>` 的最低有效位；上表箭头左侧为 CX 控制位，
右侧为目标位。`simple` 只是一种两参数启发式结构，不能表示任意实或复的双量子比特
酉矩阵；`general + real` 只覆盖行列式为 `+1` 的连通实正交分支；
`general + complex` 的 15 个参数描述忽略全局相位后的自由度。本组件不接受目标矩阵，
也不负责矩阵综合、辅助量子比特、重复层或把局域块铺设到多量子比特拓扑。

## `PauliTwoDesignAnsatz`

该组件先在每个量子比特上执行 `RY(pi/4)`，然后执行 `reps + 1` 个参数化旋转层。
每个参数槽的旋转轴在构造时从 `RX`、`RY`、`RZ` 中均匀抽样一次并冻结。前
`reps` 个旋转层之后依次执行偶边 `(0,1), (2,3), ...` 和奇边
`(1,2), (3,4), ...` 的 `CZ` brickwork 纠缠层。

```python
from cqlib_algorithms.operators import PauliTwoDesignAnsatz

ansatz = PauliTwoDesignAnsatz(4, reps=2, seed=7)
circuit = ansatz.build_circuit("theta")
```

参数顺序为“旋转层、量子比特”，数量为 `num_qubits * (reps + 1)`。相同的非空
`seed` 在本实现的不同实例间生成相同轴序列；同一实例重复构建始终确定。
`seed=None` 允许不同实例使用不同轴。不同实现或不同随机数生成器即使使用相同
seed，也不保证产生相同的随机轴序列。

## 算符演化系列

### `EvolvedOperatorAnsatz`

对有序 Hermitian Hamiltonian 序列 `O_0, ..., O_(m-1)`，每个 repetition 按输入
顺序追加

```text
exp(-i * theta_0 * O_0), ..., exp(-i * theta_(m-1) * O_(m-1))
```

参数按 repetition、operator 排列并
命名为 `{prefix}_0, {prefix}_1, ...`，总数为 `reps * m`。所有 Hamiltonian 在
构造时必须满足有限、Hermitian、非空和同宽约束。

```python
from cqlib.circuit.ansatz import EvolutionStrategy
from cqlib.qis import Hamiltonian, PauliString
from cqlib_algorithms.operators import EvolvedOperatorAnsatz

zz = Hamiltonian.from_list([(PauliString.from_str("ZZ"), 1.0)])
x = Hamiltonian.from_list([(PauliString.from_str("IX"), 1.0)])
ansatz = EvolvedOperatorAnsatz(
    [zz, x],
    reps=2,
    strategy=EvolutionStrategy.auto(1),
)
circuit = ansatz.build_circuit("theta")
```

默认 `remove_identity_operators=True`，纯 identity Hamiltonian 因只产生全局相位而
不占参数槽；如果参数化全局相位本身需要保留，可将其设为 `False`。若全部算符被
移除或 `reps=0`，返回保持原寄存器宽度的无参数 identity 线路。

### `HamiltonianVariationalAnsatz`

单一 Hamiltonian 输入会复用 `CommutingObservableGrouping`，按稳定 first-fit 规则
划分全局对易块；每块使用一个独立演化参数。也可以传入显式的有序 Hamiltonian 块
序列，此时每块必须内部对易，不要求不同块之间对易。分组后全部线路构造、identity
移除和参数管理委托 `EvolvedOperatorAnsatz` 的精确演化配置。

```python
from cqlib.qis import Hamiltonian, PauliString
from cqlib_algorithms.operators import HamiltonianVariationalAnsatz

problem_hamiltonian = Hamiltonian.from_list(
    [
        (PauliString.from_str("XX"), 0.7),
        (PauliString.from_str("ZZ"), -0.4),
    ]
)
ansatz = HamiltonianVariationalAnsatz(problem_hamiltonian, reps=2)
circuit = ansatz.build_circuit("hva")
```

自动分组只依据 Pauli 项的代数对易关系，不承诺所得块数最少，也不根据具体物理模型
重新排序项。显式块适合调用方已经具有物理分解的情况。

## `ExcitationPreservingAnsatz`

该组件交替执行局域 RZ 层和保持计算基总激发数的交换块。每条拓扑边共享一个参数：

```text
RXX(theta / 2) -> RYY(theta / 2)
```

`mode="fsim"` 时再追加一个独立参数的受控 `Phase(-phi)`。默认拓扑为 full，
而不是 TwoLocal 系列的 linear。`skip_unentangled_qubits=True` 会从 RZ 层排除未出现在
任何边中的量子比特；`skip_final_rotation_layer=True` 会省略最后的 RZ 层。

```python
from cqlib.circuit.ansatz import EntanglementTopology
from cqlib_algorithms.operators import ExcitationPreservingAnsatz

ansatz = ExcitationPreservingAnsatz(
    4,
    reps=2,
    mode="fsim",
    entanglement=EntanglementTopology.linear(),
)
circuit = ansatz.build_circuit("theta")
```

若参与 RZ 层的量子比特数为 `r`、边数为 `e`，参数数目为

```text
r * (reps + final_rotation_layer) + reps * e * (1 or 2)
```

其中 iswap 模式每边一个参数，fsim 模式每边两个参数。线路保持输入态的 Hamming
weight；若从全零态直接开始，它不会自行产生非零激发，因此实际使用通常需要先制备
目标激发数的参考态。

## 结构与随机基线

### `SimplifiedTwoDesignAnsatz`

线路先在每个量子比特上施加一个可训练 RY。每个 repetition 先访问 even 最近邻边，
再访问 odd 最近邻边；每条边依次执行 CZ、左端 RY、右端 RY。参数完全按门出现顺序
编号，总数为

```text
num_qubits + 2 * reps * (num_qubits - 1)
```

该结构用于可训练性和变分线路基准，是确定性的简化 two-design 结构，不表示任意
有限宽度和深度都形成严格 unitary two-design。

### `RandomLayersAnsatz`

每层包含恰好 `rotations_per_layer` 个参数化旋转；在每个旋转之前，以
`entanglement_ratio` 的概率最多插入一个随机有向双量子比特纠缠门。因此每层门数
不超过 `2 * rotations_per_layer`。旋转轴、目标量子比特、纠缠边和门顺序全部在
组件构造时抽样并冻结。

```python
from cqlib_algorithms.operators import RandomLayersAnsatz

ansatz = RandomLayersAnsatz(
    5,
    reps=3,
    rotations_per_layer=5,
    entanglement_ratio=0.25,
    seed=17,
)
circuit = ansatz.build_circuit("random")
```

`seed` 默认为 42；相同配置和 seed 产生相同架构。`seed=None` 允许不同实例随机，
但同一实例重复构建仍保持不变。`entanglement_ratio` 必须满足 `0 <= value <= 1`；
值为 1 时，每个多量子比特旋转槽前恰有一个纠缠门。单量子比特线路始终禁用纠缠
抽样。参数数量固定为
`reps * rotations_per_layer`，随机插入的纠缠门不增加参数。

## 张量网络启发式线路

`MPSAnsatz`、`TTNAnsatz` 与 `MERAAnsatz` 统一使用双参数局域块：先在边的左右量子
比特分别施加独立 `RY`，再执行左控右靶的 CX。参数严格按门出现顺序命名为
`{prefix}_0`、`{prefix}_1` 等。它们提供的是受张量网络连接结构启发的幺正变分线路，
不导入任意经典张量、不管理键维数，也不声称实现张量收缩。

### `MPSAnsatz`

`MPSAnsatz(num_qubits)` 要求至少两个量子比特，并按
`(0,1), (1,2), ..., (n-2,n-1)` 从左到右对每条链边应用一次局域块。参数数量为
`2 * (n - 1)`。

### `TTNAnsatz`

`TTNAnsatz(num_qubits)` 在每层从左到右配对活跃量子比特，应用局域块后保留每对的
右量子比特；奇数宽度时将最后一个未配对量子比特带入下一层。归约到根后增加一个
可训练 RY。所有量子比特仍保留在线路中，“保留”只定义后续连接。共有 `n-1` 个
双量子比特块，参数数量为 `2 * (n - 1) + 1`，最终根为量子比特 `n-1`。

### `MERAAnsatz`

`MERAAnsatz(num_qubits)` 要求宽度至少为 2 且为 2 的幂。每个尺度先在相邻配对边界
应用解纠缠位置块，再在 `(a0,a1), (a2,a3), ...` 应用等距位置块，并保留每对右侧
量子比特进入下一尺度；最后在根上增加一个可训练 RY。两类位置都使用相同的幺正
局域块但参数互不共享，并不表示公开任意非幺正等距张量。对 `n=2**L`，块数为
`2*n - 2 - L`，参数数量为 `4*n - 3 - 2*L`。

```python
from cqlib_algorithms.operators import MERAAnsatz, MPSAnsatz, TTNAnsatz

mps = MPSAnsatz(4).build_circuit("mps")
ttn = TTNAnsatz(5).build_circuit("ttn")
mera = MERAAnsatz(8).build_circuit("mera")
```

## QAOA 单层演化

`QAOACostEvolution` 构造

```text
exp(-i * gamma * H_C)
```

并将唯一参数命名为 `{prefix}_gamma`。`QAOAMixerEvolution` 构造

```text
exp(-i * beta * H_M)
```

并将唯一参数命名为 `{prefix}_beta`。后者默认使用横场混合哈密顿量
`H_M = sum_i X_i`，也可接收同宽的自定义 Hermitian `Hamiltonian`。

```python
from cqlib.qis import Hamiltonian, PauliString
from cqlib_algorithms.operators import QAOACostEvolution, QAOAMixerEvolution

cost = Hamiltonian(2)
cost.add_term(PauliString.from_str("ZZ"), 0.5)

cost_layer = QAOACostEvolution(cost).build_circuit("qaoa")
mixer_layer = QAOAMixerEvolution(2).build_circuit("qaoa")
```

默认 `EvolutionStrategy.auto(steps=1)`：对相互对易项使用精确乘积，对不对易项使用
一阶 Lie–Trotter。也可以显式传入 Core `EvolutionStrategy`。组件仅表示单个成本层
或混合层，不包含初态、层间参数管理和完整 QAOA 线路。

## 理论与接口参考

- Farhi、Goldstone、Gutmann，
  [A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028)，
  给出交替成本与混合哈密顿量演化的 QAOA 定义。
- Kandala 等，
  [Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets](https://arxiv.org/abs/1704.05018)，
  说明按设备可用相互作用设计浅层变分线路的思想。
- Grant 等，
  [Hierarchical quantum classifiers](https://doi.org/10.1038/s41534-018-0116-9)，
  给出层次量子分类器中的低深度双量子比特参数块。
- Vatan 与 Williams，
  [Optimal quantum circuits for general two-qubit gates](https://doi.org/10.1103/PhysRevA.69.032315)，
  给出一般复双量子比特酉与实正交双量子比特变换的最优 CNOT 分解框架。
- Nakata 等，
  [Unitary 2-designs from random X- and Z-diagonal unitaries](https://arxiv.org/abs/1502.07514)，
  以及 McClean 等，
  [Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173)，
  是 Pauli two-design 类随机线路的理论和应用参考。
- Wecker 等，
  [Progress towards practical quantum variational algorithms](https://arxiv.org/abs/1507.08969)，
  给出 Hamiltonian 变分线路的物理启发式组织方法。
- Gard 等，
  [Efficient symmetry-preserving state preparation circuits for the variational quantum eigensolver algorithm](https://arxiv.org/abs/1904.10910)，
  说明固定粒子数子空间中变分线路的价值及守恒结构。
- Cerezo 等，
  [Cost function dependent barren plateaus in shallow parametrized quantum circuits](https://arxiv.org/abs/2001.00550)，
  给出本文采用的简化 two-design 线路结构和可训练性分析。

这些论文用于核对数学结构和接口组织。本项目的默认值、随机序列和门顺序以本文档
及 Cqlib 公共协议为准。
