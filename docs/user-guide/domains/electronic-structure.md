# 分子电子结构

分子电子结构链把分子或已知积分转换为费米 Hamiltonian，再组合物理初态、变分拟设、
VQE 和 qEOM。积分驱动是可选入口；已有积分的调用方可以直接从
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
| 物理量 | `ParticleNumber`、`Magnetization`、`AngularMomentum` |

## 使用示例

```python
import numpy as np

from cqlib_algorithms import ElectronicGroundState
from cqlib_algorithms.extension.chemistry import (
    ElectronicIntegrals,
    ElectronicOccupation,
    ElectronicSystem,
    hartree_fock_initial_point,
)
from cqlib_algorithms.operators.estimator import StatevectorEstimator
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
result = solver.compute(system, point)
```

`result.total_energy` 包含积分常数；`electronic_energy`、`inactive_energy`、
`correlation_energy` 只在输入数据包含相应来源信息时可用。`particle_number`、
`magnetization` 和 `angular_momentum` 用于检查终态物理扇区。qEOM 接收同一
`ElectronicGroundStateResult`，计算中性、粒子数守恒的激发能。

## 适用边界

- 分子驱动只接受实轨道数据；可选依赖未安装时仍可直接使用外部积分。
- MP2 初值只适用于闭壳层、限制性、正则轨道；其他体系使用零初值或显式初值。
- 内置 UCC 是有序因子乘积，不等于非对易生成元总和的单次指数。
- 目前只提供 Jordan–Wigner 默认映射和中性 qEOM；未包含对称性约化、偶极矩、
  约化密度矩阵和自动活性空间选择。
