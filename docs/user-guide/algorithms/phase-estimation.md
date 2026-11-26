# 相位估计

标准 QPE 和迭代 QPE 共用 `PhaseEstimationProblem`，由待估计的酉算子和系统初态组成。Hamiltonian 相位估计（HPE）直接接收 `Hamiltonian` 和状态制备组件。这三种算法均可调用一次 `run()` 得到单任务结果。

## 数学约定

对酉算子 $`U`$ 的本征态 $`\vert \psi\rangle`$，归一化本征相位定义为

```math
U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle,\qquad \phi\in[0,1).
```

系统量子比特采用 Cqlib 小端序。$`m`$ 个估计量子比特的整数结果 $`y`$ 表示
$`\phi=y/2^m`$。所有状态制备和酉线路必须固定、完全绑定、静态且同宽；量子比特按
插入位置映射到连续位置。

## PhaseEstimationCircuit

`PhaseEstimationCircuit` 是无状态制备、无测量的复合线路算子。它依次执行：

1. 对估计寄存器施加 Hadamard；
2. 第 $`j`$ 个估计量子比特控制 $`U^{2^j}`$；
3. 对估计寄存器施加逆 QFT。

当前 Core 尚无把任意线路直接提升为量子受控线路的通用接口，因此实现从
`Circuit.to_matrix` 构造 $`\operatorname{diag}(I,U^k)`$。该过程保留 $`U`$ 的全局相位，但系统宽度的
内存复杂度为指数级；构造前执行 256 MiB 稠密工作集上限检查。数值酉门也不保证每个
远端后端都能直接接受或分解。

`PhaseEstimationCircuit(unitary, m)` 接收完全绑定的静态 `Circuit`。若目标来自算术、Oracle、QFT 或其他算子，先调用对应的 `build_circuit()`。目标线路必须包含完整寄存器和目标操作的绝对全局相位；相位估计分析这条线路的完整酉变换。

`approximation_degree` 直接传递给现有逆 `QFT`。估计位数限制为 52，确保公开的
binary64 相位结果仍具有逐位分辨能力，并避免超大指数转换产生非契约异常。

## 酉目标线路与初态组件

先把编码器或拟设绑定为固定制备组件，再传入相位估计问题。以下例子在两个算法中复用同一个
组件组合，输入态是被标记的单比特基态，标准 QPE 和迭代 QPE 均得到相位 `0.5`。

```python
import math
from cqlib.circuit import StandardGate
from cqlib_algorithms import (
    IterativePhaseEstimation,
    PhaseEstimation,
    PhaseEstimationProblem,
)
from cqlib_algorithms.operators import (
    AngleEncoding,
    BitPatternPhaseOracle,
    StatevectorSampler,
    UserCircuitPreparation,
)

preparation = UserCircuitPreparation.from_component(
    AngleEncoding(1, StandardGate.RY),
    {"x_0": math.pi},
    prefix="x",
)
unitary = BitPatternPhaseOracle(1, {0: 1}).build_circuit()
problem = PhaseEstimationProblem(unitary, preparation)
qpe = PhaseEstimation(StatevectorSampler()).run(
    problem,
    num_evaluation_qubits=3,
    shots=32,
)
iqpe = IterativePhaseEstimation(StatevectorSampler()).run(
    problem,
    num_iterations=3,
    shots_per_iteration=32,
)
print(qpe.phase, iqpe.phase)  # 0.5 0.5
```

目标也可以由 QSP 等算子生成，初态继续使用状态制备组件：

```python
import math
from cqlib_algorithms import PhaseEstimation, PhaseEstimationProblem
from cqlib_algorithms.operators import (
    QuantumSignalProcessing,
    StatevectorSampler,
    UniformSuperposition,
)

unitary = QuantumSignalProcessing([0.0, 0.0], signal_angle=math.pi / 4).build_circuit()
problem = PhaseEstimationProblem(unitary, UniformSuperposition(1))
result = PhaseEstimation(StatevectorSampler()).run(
    problem,
    num_evaluation_qubits=3,
    shots=32,
)
print(result.phase)  # 0.125
```

此处 QSP 对均匀单比特叠加态的本征值为 `exp(i*pi/4)`，归一化相位为 `1/8`。
作为酉输入时使用组件的完整线路；只制备同一个物理态的两个组件，其完整酉算子和全局相位
仍可能不同。BlockEncoding 与 QSVT 包含辅助寄存器，QPE 估计完整酉的本征相位，
编码矩阵的本征值或奇异值需按具体算法另行解释。


## 标准 QPE

`PhaseEstimationProblem(unitary, state_preparation)` 接收完整酉 `Circuit` 和完全绑定的
`StatePreparation`。构造时保存独立快照，`state_preparation` 为固定制备组件，
`preparation_circuit` 为执行快照，应保持不变；
`PhaseEstimationRequest` 另外给出估计位数、shots 和可选 QFT 近似度。算法支持
输入态为本征态叠加：若

```math
|\psi\rangle=\sum_j c_j|\phi_j\rangle,
```

则理想采样以 $`\vert c_j\vert ^2`$ 的权重观察对应本征相位。结果返回经验众数相位、整数网格点
和只包含已观测网格点的稀疏概率映射；众数并列时选择最小网格点。线路内逻辑酉调用
总数为 $`2^m-1`$，`num_unitary_queries` 是再乘 shots 的采样加权查询数。

下例用三个估计比特测量矩阵在 `|1⟩` 上的本征相位。该本征值为 `exp(3πi/4)`，对应归一化相位 `3/8`。改变制备的基态即可选择另一个本征值。

```python
import numpy as np

from cqlib_algorithms import PhaseEstimation, PhaseEstimationProblem
from cqlib_algorithms.operators import (
    BasisStateInitialization,
    StatevectorSampler,
    UnitaryMatrixOracle,
)

unitary = UnitaryMatrixOracle(np.exp(1j * np.pi / 4) * np.diag([1, 1j])).build_circuit()
preparation = BasisStateInitialization(1, 1)
problem = PhaseEstimationProblem(unitary, preparation)
sampler = StatevectorSampler()
qpe = PhaseEstimation(sampler)
phase = qpe.run(problem, num_evaluation_qubits=3, shots=32)
assert phase.phase == 3 / 8
```

## 迭代 QPE

`IterativePhaseEstimation` 使用一个估计辅助比特，以独立线路从最低有效位向最高有效位
提取相位。第 $`k`$ 轮使用受控 $`U^{2^k}`$，并根据已经得到的低位施加经典反馈相位；
每轮重新制备同一系统态。因不同位不在同一次相干演化中关联，该接口要求输入制备态
为一个本征态，不支持用一次运行解析本征态叠加的联合相位分布。

`shots_per_iteration` 控制每位的采样次数，按多数票确定该位，平票稳定选择 `0`。
结果按最高有效位到最低有效位返回 `bitstring` 和每位观测到 `1` 的概率。该实现用多个
静态线路和 Sampler 调用完成轮间反馈，不要求后端支持动态线路。

沿用上一例的 `problem` 和 `sampler`。三轮受控幂依次为 4、2、1，每轮将先前已测得的低位用于反馈，最终位串按高位在前排列。

```python
from cqlib_algorithms import IterativePhaseEstimation

bits = IterativePhaseEstimation(sampler).run(
    problem, num_iterations=3, shots_per_iteration=8
)
assert bits.bitstring == "011"
```

## Hamiltonian 相位估计

对有限 Hermitian Pauli Hamiltonian `H`，HPE 选择正比例因子

```math
s=\frac{\pi}{B},\qquad U=e^{isH},
```

其中 `bound` 是本征值绝对值的上界。默认取各 Pauli 项有效系数绝对值之和的两倍，
将精确本征值对应的有符号相位限制在 −1/4 到 1/4 之间。
Core 的演化约定为 $`e^{-itH}`$，实现以 $`t=-s`$ 构造上述 $`U`$。相位 $`\phi`$ 先映射
到有符号区间，再恢复本征值：

```math
\widetilde{\phi}=\phi\;(\phi<1/2),\quad\widetilde{\phi}=\phi-1\;(\phi\geq1/2),\qquad\lambda=\frac{2\pi\widetilde{\phi}}{s}
```

估计寄存器至少需要两个量子比特才能区分正负相位。手动设置较紧的 `bound` 可缩小能量网格间距，
但应给相位端点留出可分辨的余量。能量 $`+\mathtt{bound}`$ 与 $`-\mathtt{bound}`$ 都会映射到相位 $`1/2`$。
对非对易 Pauli 项，Core 使用指定 `TrotterMode` 和
`trotter_steps` 合成演化，结果对应这一乘积公式近似；默认使用二阶公式、一步。
对易项由 Core 使用精确单遍分解。

对 `H = Z`，默认谱界为 2。输入 `|0⟩` 时，测得相位 1/4、能量 +1。换成 `|1⟩`，测得相位 3/4，对应有符号相位 −1/4、能量 −1。

```python
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms import HamiltonianPhaseEstimation
from cqlib_algorithms.operators import BasisStateInitialization, StatevectorSampler

hamiltonian = Hamiltonian.from_pauli(PauliString.from_str("Z"))
energy = HamiltonianPhaseEstimation(StatevectorSampler()).run(
    hamiltonian,
    state_preparation=BasisStateInitialization(1, 0),
    num_evaluation_qubits=3,
    shots=32,
)
assert energy.phase_result.phase == 0.25
assert energy.eigenvalue == 1.0
```

## 批量执行

以标准 QPE 为例，下面沿用前文的矩阵和 `qpe`，一次提交两个不同初态。三个算法均支持 `run([Request, ...])`，空批次返回空列表，输出按输入顺序排列。标准 QPE 与 HPE 各一次提交整批线路，IQPE 按请求顺序逐位采样和反馈。

```python
from cqlib_algorithms import PhaseEstimationRequest

zero = PhaseEstimationProblem(unitary, BasisStateInitialization(1, 0))
requests = [PhaseEstimationRequest(item, 3, 32) for item in (zero, problem)]
assert [item.phase for item in qpe.run(requests)] == [1 / 8, 3 / 8]
```

## 能力边界

- 标准 QPE 与 HPE 可返回本征态叠加产生的经验分布；IQPE 要求单一本征态承诺。
- 有限 shots 结果是经验估计，不附加置信区间或噪声缓解保证。
- 相位精度由估计位数、shots、QFT 近似度和后端误差共同决定。
- HPE 还包含 Hamiltonian 缩放界与非对易项演化近似误差。
- 稠密受控矩阵路径适合小系统验证，不表示多量子比特线路具有多项式经典构造成本。

## 理论与接口参考

- M. A. Nielsen、I. L. Chuang，*Quantum Computation and Quantum
  Information*，相位估计章节。
- A. Y. Kitaev，*Quantum measurements and the Abelian Stabilizer Problem*，
  arXiv:quant-ph/9511026。
