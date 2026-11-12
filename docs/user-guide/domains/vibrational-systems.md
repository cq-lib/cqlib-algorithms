# 振动与有限 Modal 系统

振动链从已知 modal 积分开始，在每个振动 mode 内用 one-hot qubit 块表示有限 modal
基。它覆盖稀疏二次量子化算符、映射、VSCF 参考态、UVCC/UVCCSD、VQE 基态以及
严格的 one-hot 泄漏诊断。

## 计算链

```text
ModalSpace + ModalIntegrals
→ VibrationalEnergy
→ VibrationalOp
→ OneHotVibrationalMapper
→ VibrationalSystem
→ VSCFStatePreparation + UVCCSDAnsatz
→ VibrationalGroundState
→ modal occupations + one-hot leakage
```

`ModalSpace((n0, n1, ...))` 为每个 mode 分配连续的 one-hot qubit 块。一个物理基态
在每个块中恰有一个 `1`。`ModalIntegrals` 用稀疏 matrix-unit 表示一模、二模及更高阶
耦合，并在构造时检查 Hermitian 共轭对称；`truncation_order` 可限制保留的耦合阶数。

## 使用示例

```python
from cqlib_algorithms import VibrationalGroundState
from cqlib_algorithms.extension.vibration import (
    ModalIntegrals,
    ModalSpace,
    ModalTransition,
    VibrationalEnergy,
    VibrationalSystem,
)
from cqlib_algorithms.operators import ParameterShiftGradient, StatevectorEstimator
from cqlib_algorithms.operators.optimizer import GradientDescent

space = ModalSpace((2, 2))
integrals = ModalIntegrals.from_terms(
    space,
    (
        ((ModalTransition(0, 1, 1),), 1.0),
        ((ModalTransition(1, 1, 1),), 1.5),
        (
            (ModalTransition(0, 1, 1), ModalTransition(1, 1, 1)),
            0.2,
        ),
    ),
)
system = VibrationalSystem(VibrationalEnergy(integrals))
solver = VibrationalGroundState(
    StatevectorEstimator(),
    GradientDescent(ParameterShiftGradient(), maxiter=20, tolerance=1e-12),
)
result = solver.compute(system)
```

默认参考态占据每个 mode 的 modal 0，也可通过 `reference_modals` 指定。返回结果包含
每个 modal 的占据、每个 mode 的总占据以及
<img src="../../assets/formulas/inline/inline-cc0552fd94289a98.svg" alt="\left(N_{\mathrm{mode}}-I\right)^2" style="vertical-align: -3.000pt;"> 泄漏量；总占据等于 1 只是均值条件，泄漏量为 0 才严格证明状态完全
位于 one-hot 物理子空间。

## 适用边界

- 当前入口是已经给定的 modal 积分，不包含势能面生成、modal 优化或真正的 VSCF
  自洽方程求解。
- one-hot 编码每个 modal 使用一个 qubit，适合有限截断下的小中型问题。
- 内置闭环面向振动基态；未包含振动 qEOM、跃迁强度和光谱工作流。
- `VibrationalOp` 是本仓库的临时有限-modal 中间表示，不是通用无限维玻色算符。
