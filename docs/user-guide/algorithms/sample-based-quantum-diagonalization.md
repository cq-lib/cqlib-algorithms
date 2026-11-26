# 基于采样的量子对角化（SQD）

`SQD` 从量子线路产生的 Slater determinant 样本构造多个经典选定子空间，在每个
子空间中投影并对角化电子 Hamiltonian，再用所得低能态的轨道占据数反复修复粒子数
错误的原始样本。它面向“量子设备负责提供重要 determinant，经典计算负责高精度求解
Ritz 问题”的量子—经典协同场景。已有硬件 counts 可以直接传给
`compute_from_counts()`，不需要为了后处理而重新执行线路。

## 数学流程

设 `state_preparation` 在 $`M=2N_{\mathrm{orb}}`$ 个 Jordan–Wigner 量子比特上制备
$`\vert \Psi\rangle`$。计算基测量给出带噪声的 determinant 样本集合

```math
\widetilde{\mathcal{X}}=\{x\mid x\sim\widetilde{P}_{\Psi}(x)\},\qquad x\in\{0,1\}^{M}
```

本库电子结构采用交错自旋轨道顺序：位置 $`2p`$ 表示空间轨道 $`p`$ 的 alpha
自旋，位置 $`2p+1`$ 表示 beta 自旋；Core 计数字符串按小端量子比特语义解释。
alpha、beta 粒子数分别由偶数、奇数位置的 Hamming weight 得到。

SQD 依次执行：

1. 未提供 `initial_occupancies` 时，先从原始样本中后选择出满足目标
   $`(N_\alpha,N_\beta)`$ 的配置，构造并求解首轮选定子空间，再按第 5 步从 Ritz 态
   计算第一组占据数
   $`n_{p\sigma}`$。这一步是论文定义的 setup round，而不是直接把原始样本频率当作
   低能态占据数。若没有正确粒子扇区样本，调用方必须提供物理归一的
   `initial_occupancies`；提供后会跳过 setup round 并直接开始恢复。
2. 对粒子数错误的每个自旋扇区翻转恰好
   $`\vert N_{x,\sigma}-N_\sigma\vert `$ 个比特。候选位按当前比特与平均占据数的不一致程度赋权，
   因而低占据的错误 `1` 和高占据的错误 `0` 更容易被修复。正确扇区不被改变。
3. 默认按恢复后 determinant 的经验频率无放回抽取 `num_batches` 个独立批次，每批
   至多 `samples_per_batch` 个唯一 determinant。启用 `symmetrize_spin=True` 时，改按
   论文式 (11) 抽取约 $`\sqrt{\mathtt{samples\_per\_batch}}/2`$ 个完整配置，将每个配置的 alpha、
   beta 空间占据串收集到同一个集合，再取该集合与自身的笛卡尔积。这会自动加入
   开壳层配置的自旋交换伙伴并促进总自旋守恒，但不等价于严格投影到某个 $`S^2`$ 本征值。
4. 对每批 $`\mathcal S^{(k)}`$ 构造

   ```math
   H_{\mathcal{S}^{(k)}}=P_{\mathcal{S}^{(k)}}HP_{\mathcal{S}^{(k)}},\qquad P_{\mathcal{S}^{(k)}}=\sum_{x\in\mathcal{S}^{(k)}}|x\rangle\langle x|
   ```

   并求其最低 Ritz 对 $`(E^{(k)},\vert \psi^{(k)}\rangle)`$。投影采用 Pauli 位掩码稀疏
   作用，不生成完整 $`2^M\times2^M`$ Hamiltonian。
5. 按论文定义对所有批次等权平均新占据数：

   ```math
   n_{p\sigma}=\frac{1}{K}\sum_{k=1}^{K}\langle\psi^{(k)}|\widehat{n}_{p\sigma}|\psi^{(k)}\rangle
   ```

6. 当相邻轮最低能量变化不超过 `energy_tolerance`，且最大轨道占据数变化不超过
   `occupancy_tolerance` 时收敛；否则继续恢复，最多执行 `max_iterations` 轮。
   `callback` 在每轮完成后收到不可变的 `SQDIteration`，可用于进度记录和保留中间结果。

`include_configurations`、跨轮 `carryover_threshold`、可注入求解器和 callback 是面向工程
复用与可观测性提供的扩展接口，不是论文式 (1)--(4) 所定义的必要数学步骤。它们不改变
配置恢复、选定子空间投影和批次占据数平均的核心定义。

每个 $`E^{(k)}`$ 都是对应选定子空间中的变分上界。增加有物理贡献的 determinant
通常会降低该上界，但有限-shot 样本、随机分批与恢复过程不保证每轮能量单调下降。
返回的 `ground_state` 因而取所有已完成轮次中最低的 Ritz 结果。

## 输入契约

- `ElectronicSystem` 可保存限制性或自旋分辨实积分；Hamiltonian 使用现有
  `ElectronicEnergy` 并按 Jordan–Wigner 映射。
- `run(system, state_preparation, ...)` 直接接收 `StatePreparation`；算法内部以
  `"sqd"` 为前缀构建线路，并复制执行对象。构建结果必须完全绑定、静态、酉，宽度恰好为
  `system.num_spin_orbitals`。线路位置 (i) 必须表示自旋轨道模式 (i)；不能把其他
  费米映射产生的量子比特直接当作 occupation bits。
- `shots`、`samples_per_batch`、`num_batches` 和 `max_iterations` 必须为正整数。
- `max_iterations` 限制对角化轮次总数；没有显式初始占据数时，setup round 也计入该
  上限。因此要执行至少一次 setup 和一次恢复，应设置 `max_iterations >= 2`。
- `initial_occupancies` 若提供，必须按 alpha/beta 交错顺序排列，每项位于 `[0, 1]`，
  且两个自旋扇区分别求和为 `num_alpha` 和 `num_beta`。
- `seed` 同时派生独立的 Sampler seed 与经典恢复 seed；后端若遵守 Sampler seed
  契约，完整运行可复现。
- `compute_from_counts()` 接受长度严格等于自旋轨道数的二进制字符串和非负整数频数；
  shots 由频数总和推导。该入口适合离线保存的硬件结果。
- `include_configurations` 必须由互不重复的目标粒子扇区 determinant 组成，数量不能
  超过 `samples_per_batch`。它适合保证 Hartree–Fock 参考态等已知重要配置不被抽样丢失。
- `symmetrize_spin=True` 只适用于 `num_alpha == num_beta`，并要求
  `samples_per_batch >= 4`。此时容量约束作用于共享空间占据串集合的笛卡尔积；因此实际
  determinant 数可能小于 `samples_per_batch`，而用户强制配置及跨轮保留配置也必须在
  闭包后仍满足该上限。
- `carryover_threshold=None` 关闭跨轮保留；否则按 Ritz 振幅而不是概率阈值筛选，并在
  容量不足时优先保留绝对振幅最大的配置。
- `SQD(subspace_solver=...)` 可以注入外部选定 CI 求解器。求解器接收
  `(ElectronicSystem, configurations)`，必须返回覆盖完全相同配置的
  `SQDSubspaceResult`。默认实现可通过 `solve_sqd_subspace()` 单独调用。
- `recover_sqd_configurations()` 公开配置恢复步骤，便于检查或组合自己的后处理流程。

## 结果解释

`SQDResult` 提供：

- `energy` 与 `ground_state`：所有轮次、所有批次中最低的投影 Ritz 能量及其稀疏
  determinant 展开；
- `orbital_occupancies`：最后完成轮次的 alpha/beta 交错占据数；
- `ground_state_occupancies`：由返回的最低能量 `ground_state` 本身计算的占据数；当最低
  能量来自较早轮次时，它与最终自洽占据数可能不同；
- `best_iteration_index`：产生最低能量状态的轮次；
- `iterations`：逐轮 `SQDIteration` 历史，每轮保存全部 `SQDSubspaceResult`、平均
  占据数和本轮后选择或恢复后的唯一 determinant 数；
- `converged`：能量和占据数两个收敛条件是否同时满足；
- `valid_raw_shots`：恢复前已经处于目标粒子扇区的 shots 数。

达到 `max_iterations` 但未收敛时仍返回有限结果，并将 `converged` 设为 `False`。
`SQDSubspaceResult.configurations[i]` 与 `amplitudes[i]` 一一对应；configuration 是
小端交错自旋轨道 occupation 的非负整数。

## 示例

下面从 Hartree–Fock determinant 采样一个两空间轨道的一电子体系。该例的子空间只有
一个 determinant，主要用于展示接口；研究计算应使用能覆盖相关 determinant 的态制备
线路，并对 shots、批次数和子空间大小做收敛研究。

```python
import numpy as np

from cqlib_algorithms import SQD
from cqlib_algorithms.extension.electronic_structure import (
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
)
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.operators.state_preparation import (
    HartreeFockStatePreparation,
)

system = ElectronicSystem(
    ElectronicIntegrals(
        constant=0.3,
        one_body=np.diag([-0.8, 0.4]),
        two_body=np.zeros((2,) * 4),
    ),
    ElectronicOccupation(num_spatial_orbitals=2, num_alpha=1, num_beta=0),
)
preparation = HartreeFockStatePreparation(system.occupation)

result = SQD(StatevectorSampler()).run(
    system,
    preparation,
    shots=1024,
    samples_per_batch=128,
    num_batches=4,
    max_iterations=5,
    seed=7,
)

print(result.energy)
print(result.converged)
print(result.ground_state.configurations)
```

理想输出约为 `-0.5`、`True` 和 `(1,)`。替换 `preparation` 即可改变采样态，注入
`subspace_solver` 可替换单批经典对角化求解器；两者均需遵守相同占据编码和结果约定。
`compute()` 与 `run()` 使用相同的组件输入和计算流程。自定义制备线路通过
`UserCircuitPreparation(circuit)` 包装为 `StatePreparation` 后传入。

如果 counts 已经由设备生成，可以省略线路和 Sampler：

```python
result = SQD().compute_from_counts(
    system,
    {"0001": 900, "0100": 124},
    samples_per_batch=128,
    include_configurations=(0b0001,),
    num_batches=4,
    seed=7,
)
```

闭壳层问题可以显式打开论文中的自旋配置重组：

```python
closed_shell_system = ElectronicSystem(
    ElectronicIntegrals(0.0, np.zeros((2, 2)), np.zeros((2,) * 4)),
    ElectronicOccupation.closed_shell(2, 2),
)
result = SQD().compute_from_counts(
    closed_shell_system,
    {"1001": 1024},
    samples_per_batch=256,
    symmetrize_spin=True,
    seed=7,
)
```

若样本包含开壳层 determinant，该模式会把 alpha、beta 空间占据串重组为闭包子空间。
它为 singlet/triplet 线性组合提供所需 determinant 支撑，但没有加入论文式 (12) 的
$`S^2`$ 软约束，因此调用方不能仅凭该开关声称结果是严格的总自旋本征态。

## 适用边界与资源

- 当前公开算法实现电子结构 SQD，并可选提供论文式 (11) 的自旋配置闭包；不包含式 (12)
  的 $`S^2`$ 罚函数、Pauli-only 通用 SQD、轨道优化、Trim SQD、分布式调度或
  excited-state targeting。外部求解器可以替换单批对角化，但批次调度仍由当前进程
  串行执行。
- 内置求解器为单进程参考实现。投影组装约为“子空间维数 × Pauli 项数”，稀疏
  eigensolver 的时间和内存取决于投影矩阵非零元；`samples_per_batch` 是主要资源旋钮。
- SQD 的精度依赖量子样本是否包含目标低能态的重要 determinant。配置恢复只强制
  alpha/beta 粒子数并按占据数重加权，不能凭空保证丢失的量子信号或其他对称性。
- 结果是有限选定子空间的变分估计，不应表述为 full configuration interaction 精确解，
  除非选定子空间已经覆盖完整目标扇区并经过独立验证。

算法和自洽配置恢复依据 Robledo-Moreno 等人的
[Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer](https://doi.org/10.48550/arXiv.2405.05068)。
