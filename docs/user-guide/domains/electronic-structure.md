# 分子电子结构

分子电子结构链把分子或已知积分转换为费米 Hamiltonian，再组合物理初态、变分拟设、
VQE、基于采样的量子对角化（SQD）和 qEOM。积分驱动是可选入口；已有积分的调用方可以直接从
`ElectronicIntegrals` 或 `SpinResolvedElectronicIntegrals` 开始。

## 计算链

```text
MolecularSpecification
→ PySCFDriver（可选）
→ ElectronicStructureData
→ ActiveSpace / SpinResolvedActiveSpace（可选）
→ ElectronicSystem
→ ElectronicEnergy
→ FermionicOp → qubit Hamiltonian
→ HF / Slater / Gaussian reference
→ UCCSD / PUCCD / PUCCSD
→ ElectronicGroundState
→ ElectronicQEOM（可选）

或：

ElectronicSystem + occupation-basis state preparation
→ Sampler
→ SQD configuration recovery
→ selected-subspace diagonalization
```

限制性数据覆盖 RHF 与 ROHF，共用一套空间轨道；`reference="uhf"` 生成 alpha/beta
轨道分离的自旋分辨数据。`ElectronicSystem.from_data()` 保留电子数、轨道能量、
alpha/beta 重叠和能量组成，并可执行显式活性空间变换。默认费米到量子比特转换使用
Jordan–Wigner；自定义 mapper 必须同时用于 Hamiltonian、初态、拟设和观测量。
`PySCFDriver` 在生成数据前验证轨道在 AO overlap 度量下正交归一；直接构造
`ElectronicStructureData` 或 `SpinResolvedElectronicStructureData` 时，调用方必须
保证输入轨道已经满足同一契约，因为后端无关的数据快照不保存完整 AO overlap。

## 选择组件

| 目标 | 组件 |
| --- | --- |
| 分子输入与积分 | `MolecularSpecification`、`PySCFDriver` |
| 已知限制性/自旋分辨积分 | `ElectronicIntegrals`、`SpinResolvedElectronicIntegrals` |
| 活性空间 | `ActiveSpace`、`SpinResolvedActiveSpace` |
| 电子 Hamiltonian | `ElectronicEnergy` |
| 参考态 | `HartreeFockStatePreparation`、`SlaterDeterminantStatePreparation`、`FermionicGaussianStatePreparation` |
| 变分拟设 | `UCCSDAnsatz`、`PUCCDAnsatz`、`PUCCSDAnsatz` |
| 基态和中性激发态 | `ElectronicGroundState`、`ElectronicQEOM` |
| 采样选定子空间基态 | `SQD`、`SQDResult`、`recover_sqd_configurations`、`solve_sqd_subspace` |
| 物理量 | `ParticleNumber`、`Magnetization`、`AngularMomentum` |

## 使用示例

```python
import numpy as np

from cqlib_algorithms import ElectronicGroundState, ElectronicQEOM
from cqlib_algorithms.extension.electronic_structure import (
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    hartree_fock_initial_point,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
from cqlib_algorithms.operators.ansatz import UCCSDAnsatz
from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import GradientDescent

system = ElectronicSystem(
    ElectronicIntegrals(
        constant=0.7,
        one_body=np.diag([-1.0, 0.5]),
        two_body=np.zeros((2,) * 4),
    ),
    ElectronicOccupation.closed_shell(2, 2),
    orbital_energies=np.array([-1.0, 0.5]),
    hartree_fock_energy=-1.3,
    nuclear_repulsion_energy=0.7,
    inactive_energy=0.0,
)
point = hartree_fock_initial_point(system, prefix="chem")
solver = ElectronicGroundState(
    StatevectorEstimator(),
    GradientDescent(ParameterShiftGradient(), maxiter=20, tolerance=1e-12),
    prefix="chem",
)
result = solver.run(system, point, ansatz=UCCSDAnsatz(system.occupation))
excited = ElectronicQEOM(StatevectorEstimator()).run(result)
print(result.total_energy)
print(excited.excitation_energies)
```

`result.total_energy` 包含积分常数；`electronic_energy`、`inactive_energy`、
`correlation_energy` 只在输入数据包含相应来源信息时可用。`particle_number`、
`magnetization` 和 `angular_momentum` 用于检查终态物理扇区。qEOM 接收同一
`ElectronicGroundStateResult`，计算中性、粒子数守恒的激发能。
该例输出约为 `-1.3` 和 `[1.5, 1.5, 3.0]`。将 `UCCSDAnsatz` 换为兼容的
`PUCCDAnsatz` 或 `PUCCSDAnsatz` 可比较不同的变分状态空间；省略 `initial_point` 时
使用所选拟设的零参数初值。qEOM 保留基态结果记录的 mapper，`excitations` 可指定
中性激发池，`num_roots` 限制返回根数。两个求解器的 `compute()` 与 `run()` 使用相同的求解流程。

当目标是从 occupation-basis 样本提取低能 determinant 子空间时，可改用
[`SQD`](../algorithms/sample-based-quantum-diagonalization.md)。SQD 固定使用
Jordan–Wigner occupation 语义，并将 alpha/beta 粒子数恢复、经验频率分批、投影
对角化和轨道占据数自洽更新组成可执行工作流；它不是 VQE 优化器，也不要求逐项测量
Hamiltonian 期望值。调用方既可以注入 `Sampler` 执行线路，也可以通过
`compute_from_counts()` 处理已保存的设备 counts；参考配置强制保留、跨轮高振幅配置
carryover、迭代 callback 和外部子空间求解器注入均由同一工作流支持。

## 适用边界

- 分子驱动只接受实轨道数据；可选依赖未安装时仍可直接使用外部积分。
- MP2 初值只适用于闭壳层、限制性、正则轨道；其他体系使用零初值或显式初值。
- 内置 UCC 是有序因子乘积，不等于非对易生成元总和的单次指数。
- 目前只提供 Jordan–Wigner 默认映射和中性 qEOM；未包含对称性约化、偶极矩、
  约化密度矩阵和自动活性空间选择。
- SQD 当前提供单进程稀疏投影参考求解器、论文式自旋配置闭包，并允许注入外部 SCI
  求解器；工作流尚未提供分布式批次调度、轨道优化、严格 <img src="../../assets/formulas/inline/inline-c8d0b37b4af3bfcc.svg" alt="S^2" style="vertical-align: -1.000pt;"> 罚函数或通用 Pauli
  SQD。大子空间计算必须显式评估经典内存和求解时间。
