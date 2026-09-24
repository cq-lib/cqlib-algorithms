# 梯度、量子几何张量与自然梯度组件

`cqlib_algorithms.operators.gradient` 提供变分算法所需的最小梯度组件集，
对应组件清单"自然梯度优化（QGT + 梯度）"：

- `ParameterShiftGradient` —— 精确参数移位梯度，显式继承
  `cqlib_algorithms.protocols.gradient.Gradient` 抽象基类；
- `LinCombEstimatorGradient` —— 基于辅助量子比特 Hadamard test 的精确
  线性组合梯度；
- `FiniteDifferenceGradient` —— 前向或中心有限差分梯度；
- `SPSAGradient` —— 每个非空请求固定两次期望值评估的随机梯度估计；
- `StatevectorQGT` / `QGTResult` —— 精确量子几何张量与实度规；
- `StatevectorQFI` / `QFIResult` —— 纯态量子 Fisher 信息矩阵；
- `NaturalGradient` —— 自然梯度方向 `g·η = ∇E` 的求解器。

数学定义采用 Stokes et al.（arXiv:1909.02108）的对称化约定：
`F_ij = ⟨∂iψ|(1−|ψ⟩⟨ψ|)|∂jψ⟩`，`g_ij = Re F_ij`。端序遵循 Cqlib
小端约定，量子比特位置按线路插入顺序解释，支持非连续量子比特 ID。
算法内部的频率和虚部检查使用各自记录的数值容差；`QGTResult` 的结构校验
使用纯绝对容差 `1e-9`。不通过放宽容差掩盖错误。

## 参数移位规则

每个参数化门按其分支本征值谱分类（`G(angle) = Σ_λ e^{-iλ·angle}P_λ`）：

| 门 | 槽位谱 |
|---|---|
| RX/RY/RZ/RXX/RYY/RZZ/RZX | {±1/2} |
| Phase | {0, −1} |
| CRX/CRY/CRZ 及多控 R*/Phase | {0, ±1/2} |
| U 的 θ 槽 | {±1/2} |
| U 的 φ/λ 槽 | {0, −1} |
| GPhase | 零贡献 |

符号通过仿射表达式（如 `2θ+1`）进入门时，组件对每个门参数槽独立执行
精确局部移位，再按仿射链式系数加权并对同一符号的全部出现点求和。普通
旋转门和相位门的每个出现点使用经典两点规则
（`s=±π/2, w=±1/2`）；受控旋转的局部频率为 `{1/2, 1}`，每个出现点
使用四点规则。该逐出现点算法避免构造共享符号所有门谱的组合频率集，
因此评估次数随出现点数量线性增长。例如，一个成本参数同时出现在 11 个
普通双量子比特旋转中时，恰好产生 22 次评估。它不使用有限差分降级。
非线性参数表达式（如 `sin(θ)`）以及待求导
符号进入不支持的参数化门（XY/XY2P/XY2M/RXY/FSIM、符号化
UnitaryGate）时以 `ValueError` 拒绝，且在任何 estimator 调用之前抛出。
线路中其他已完整绑定、但未列入 `parameter_names` 的参数门作为固定酉演化
保留。CircuitGate 属于当前组件的线路级范围限制，无论是否含待求导符号
都被拒绝。

所有请求的全部局部门槽移位线路压平为一次 `estimator.run` 批处理。
每条评估线路先绑定原请求参数，再只替换目标门槽；原线路保持不变，
量子比特插入顺序、barrier、全局相位和门标签保留。shots 模式的请求原样
传递，梯度统计质量取决于底层估计器。仅出现在
`set_global_phase` 或 `GPhase` 中的符号梯度恒为零，不产生任何评估。

## 线性组合梯度（LinCombEstimatorGradient）

该组件采用 Schuld et al.（arXiv:1811.11184）的辅助量子比特线性组合
测量思路。对仿射门参数 `angle = a·t+b`，若生成元分解为
`G = Σ_k w_k P_k`，则在线路参数因子前插入受辅助量子比特控制的 `P_k`，
辅助位制备为 `RY(π/2)|0⟩`，并测量 `O⊗Y_aux`：

`∂E/∂t = 2a Σ_k w_k ⟨O⊗Y_aux⟩_k`。

普通 Pauli 旋转生成元为 `P/2`；受 `k` 个量子比特控制的 Pauli 旋转展开为
`Σ_S (-1)^|S| Z_S P / 2^(k+1)`。受控 Phase 的生成元展开为控制位与目标位
上投影算子的 Pauli-Z 线性组合，其中纯单位项不影响 Hermitian 期望值导数，
因此不产生评估。`U(θ,φ,λ)` 按执行顺序展开为
`RZ(λ) → RY(θ) → RZ(φ)`，再分别插入对应生成元。PauliString 的 `±1`
相位和 Hamiltonian 每个项的相位、系数均会原样扩展到辅助位。

所有请求、出现点和生成元项压平为一次 Estimator 批处理。一次普通旋转需要
1 个辅助线路；`k` 控旋转需要 `2^k` 个；具有 `k` 个控制位的受控 Phase
需要 `2^(k+1)-1` 个。辅助量子比特使用未被原线路占用的 Core `Qubit` 标识，
可处理非连续或重排的量子比特 ID。`GPhase` 和线路全局相位的导数恒为零，
不调用 Estimator。shots 原样透传，统计误差由底层 Estimator 决定。

## 数值梯度

`FiniteDifferenceGradient(epsilon=1e-6, method="central")` 不分析门生成元，
因此可用于底层 Estimator 能执行的任意参数化线路。中心差分使用
`[E(t+ε)-E(t-ε)]/(2ε)`，每个参数两次评估，截断误差为 `O(ε²)`；
`method="forward"` 使用 `[E(t+ε)-E(t)]/ε`，每个参数两次评估，截断误差为
`O(ε)`。所有请求的评估点压平为单批，调用方需要结合浮点舍入、shots 噪声和
目标函数尺度选择 `epsilon`。

`SPSAGradient(perturbation, seed)` 对每个非空请求抽取一个 Rademacher 向量
`Δ∈{−1,+1}^d`，仅评估 `E(θ+cΔ)` 与 `E(θ−cΔ)`，并返回
`ĝ_i=[E(θ+cΔ)-E(θ−cΔ)]Δ_i/(2c)`。评估数与参数维数无关。整数 seed 可复现；
传入 NumPy `Generator` 时共享其随机流；同一实例的随机状态会随调用推进。
当前实现每次调用只使用一组扰动，不包含多扰动平均。

## QGT（StatevectorQGT）

`compute(request: GradientRequest) -> QGTResult` 复用冻结请求类型的
全部校验，走 Core `Statevector` 态矢量路径：对每个参数，把其每个
出现点的（缩放后）生成元直接插入绑定线路的指令流，精确组装导数态
`|∂jψ⟩`，再计算 `F_ij`。不使用有限差分，也不构造稠密酉矩阵。

U 门在插入前展开为 `[RZ(λ), RY(θ), RZ(φ)]`：φ/λ 槽的生成元与整个
U 门不对易，只有针对展开后的单参数因子，生成元插入位置才是精确的；
全局相位因子 `e^{i(φ+λ)/2}` 对投影内积无影响，安全丢弃。

全局相位不能影响 `F`：其导数贡献与 `|ψ⟩` 平行，被投影算子消除。
这同时中和了当前兼容基线中 `Statevector.from_circuit` 忽略线路全局相位的
已知问题。全路径不调用 `Statevector.expectation`，规避同一基线中奇数 Y
符号反转的已知问题。

`QGTResult(parameter_names, tensor, metric)` 浅冻结并防御性拷贝数组：
`tensor` 为复 Hermitian `F`，`metric` 为实对称 `g = Re F`，形状与
名称一一对应，构造时校验。

## 自然梯度（NaturalGradient）

`NaturalGradient(regularization=0.0).run(estimator, requests)` 内部执行
单批参数移位梯度 + QGT + 线性求解：

- `regularization=0`：最小范数最小二乘（伪逆），容忍奇异度规；
- `regularization=ε>0`：岭正则 `(g+εI)·η = ∇E`。

它不继承 `Gradient` 抽象基类：其输出是几何修正后的方向，不是期望值
的导数。经典优化循环（迭代、步长调度）不在本组件范围内。

## 量子 Fisher 信息（StatevectorQFI）

纯态约定下 `QFI = 4·Re(F) = 4g`。`StatevectorQFI.compute(request)` 复用
`StatevectorQGT` 的精确导数态路径并返回浅冻结的 `QFIResult`；矩阵按
`parameter_names` 排列，要求有限、实对称，输入数组会被防御性复制。空参数请求
返回 `0×0` 矩阵。自然梯度仍使用 `g`，改用 QFI 只会把相同步长下的方向缩放
为四分之一。

## 使用示例

```python
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.protocols import Estimator, EstimatorRequest, GradientRequest
from cqlib_algorithms.operators.gradient import (
    FiniteDifferenceGradient,
    LinCombEstimatorGradient,
    NaturalGradient,
    ParameterShiftGradient,
    SPSAGradient,
    StatevectorQFI,
    StatevectorQGT,
)

theta = Parameter("theta")
circuit = Circuit(2)
circuit.h(0)
circuit.ry(1, theta)
circuit.cx(0, 1)

request = GradientRequest(
    EstimatorRequest(circuit, PauliString.from_str("ZZ"), {"theta": 0.31}),
    ("theta",),
)

qgt = StatevectorQGT().compute(request)
qfi = StatevectorQFI().compute(request)


def evaluate(estimator: Estimator):
    gradient = ParameterShiftGradient().run(estimator, [request])[0]
    lin_comb = LinCombEstimatorGradient().run(estimator, [request])[0]
    finite_difference = FiniteDifferenceGradient().run(estimator, [request])[0]
    spsa = SPSAGradient(perturbation=0.01, seed=7).run(estimator, [request])[0]
    direction = NaturalGradient().run(estimator, [request])[0]
    return gradient, lin_comb, finite_difference, spsa, direction
```

`estimator` 是任何实现 `Estimator` 抽象基类的同步批量精确估计器。

## 已知限制

- CircuitGate（含符号）被拒绝。这是组件当前的线路级范围限制，不是 Core
  参数暴露缺口；显式提供第三个参数绑定后，CircuitGate 的
  `operation.params` 与普通酉子线路 `decompose()` 均可用。支持 CircuitGate
  仍需设计递归展开、链式求导和嵌套位置映射；
- 待求导符号进入符号化 UnitaryGate 时被拒绝；只含未请求参数且已完整绑定
  的门可以作为固定酉演化保留；
- 测量、重置、延时与经典控制指令被拒绝（静态酉线路要求）；
- 数值边界（非线性表达式、非有限值）显式抛 `ValueError`，不做静默
  降级。

Core 接口基线：`cqlib 1.4.0b1`（develop 提交
`c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85`）。本组件使用
`Circuit.operations`、`Circuit.assign_parameters`、
`Statevector.from_circuit/from_state/apply_circuit/data`、
`Parameter.derivative/symbols/is_constant/evaluate` 与
`StandardGate` 门枚举。

## 理论依据

- Schuld et al., *Evaluating analytic gradients on quantum hardware*,
  [arXiv:1811.11184](https://arxiv.org/abs/1811.11184)：辅助量子比特
  线性组合测量以及标准双本征值生成元的两点参数移位规则。
- Wierichs et al., *General parameter-shift rules for quantum gradients*,
  [arXiv:2107.12390](https://arxiv.org/abs/2107.12390)：基于有限 Fourier
  频谱的一般参数移位与函数重构。本实现对每个门槽的局部生成元谱应用精确
  移位，并按多变量链式法则聚合共享符号的出现点。
- Spall, *Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation*, IEEE Transactions on Automatic
  Control 37(3), 1992：SPSA 的双边同时扰动估计。
- Stokes et al., *Quantum Natural Gradient*,
  [arXiv:1909.02108](https://arxiv.org/abs/1909.02108)：投影 QGT、实部度规
  与自然梯度线性系统。
