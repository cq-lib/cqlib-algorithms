# 量子误差缓解

本页说明 `cqlib_algorithms.error_mitigation` 的组件执行、算法连接和方法选择。
各方法通过线路变换、额外测量或经典后处理估计目标可观测量；适用条件、偏差和
方差需要结合实际噪声与资源评估。

## 直接组合组件执行

`*_executor(base, ...)` 返回可替换、可继续包装的 `QemExecutor`。
独立任务直接调用 `run(preparation, observable, ...)`，输入状态制备组件和可观测量，
返回 `EstimatorResult`，其中 `expectation` 是期望值，`variance` 是方法可提供的估计方差。

```python
import math
from cqlib.qis import PauliString
from cqlib_algorithms.error_mitigation import QemExecutor, zne_executor
from cqlib_algorithms.operators import ArbitraryStatePreparation, StatevectorEstimator

preparation = ArbitraryStatePreparation([math.cos(0.2), math.sin(0.2)])
pipeline = zne_executor(
    QemExecutor(estimator=StatevectorEstimator()),
    scale_factors=(1.0, 3.0, 5.0),
)
result = pipeline.run(preparation, PauliString.from_str("Z"))
print(result.expectation)  # approximately cos(0.4) = 0.921061
```

该例使用理想模拟验证调用，输出约为 `0.921061`。真实 ZNE 需要后端反映线路折叠带来的噪声变化。

## 方法替换与嵌套组合

以下四个方法使用同一个有限-shot 后端。修改 `selected` 的选择即可切换方法；
将一个执行器传入另一个执行器即可组合。示例的 `qem_0`、`qem_1` 是角度编码器在
默认前缀 `qem` 下的符号名，`parameter_bindings` 必须完整对应实际使用的符号。
也可传 `prefix="features"`，并相应使用 `features_0`、`features_1`。

```python
from cqlib.circuit import StandardGate
from cqlib.qis import PauliString
from cqlib_algorithms.error_mitigation import (
    QemExecutor,
    cdr_executor,
    lre_executor,
    vd_executor,
    zne_executor,
)
from cqlib_algorithms.operators import (
    AngleEncoding,
    SamplingEstimator,
    StatevectorEstimator,
    StatevectorSampler,
)

base = QemExecutor(estimator=SamplingEstimator(StatevectorSampler()))
pipelines = {
    "ZNE": zne_executor(base, scale_factors=(1.0, 3.0, 5.0), shots=4096, seed=7),
    "LRE": lre_executor(
        base, degree=1, fold_multiplier=1, num_chunks=2, shots=4096, seed=7
    ),
    "CDR": cdr_executor(
        base,
        ideal_estimator=StatevectorEstimator(),
        fraction_non_clifford=0.5,
        num_training_circuits=12,
        shots=4096,
        seed=7,
    ),
    "VD": vd_executor(
        base, copies=2, shots_numerator=4096, shots_denominator=4096, seed=7
    ),
}
pipelines["ZNE + VD"] = vd_executor(
    pipelines["ZNE"],
    copies=2,
    shots_numerator=4096,
    shots_denominator=4096,
    seed=7,
)
selected = pipelines["ZNE + VD"]
result = selected.run(
    AngleEncoding(2, StandardGate.RY),
    PauliString.from_str("ZZ"),
    parameter_bindings={"qem_0": 0.4, "qem_1": 0.9},
)
print(result.expectation)
```

理想态的参考值为 `cos(0.4) * cos(0.9)`，约 `0.57254`。此处使用有限采样，输出会有统计误差。
训练线路的多样性、外推稳定性、多副本噪声独立性和测量预算仍由各方法的适用条件约束。

## 接入参数化算法

`as_estimator()` 返回符合公共 `Estimator` 协议的组件。它通过执行器估计期望值，
保留缓存和已配置的缓解步骤，校验批次请求与结果。算法仍接收自己的拟设、优化器和问题数据。

```python
from cqlib.qis import PauliString
from cqlib_algorithms import VQE
from cqlib_algorithms.error_mitigation import QemExecutor, zne_executor
from cqlib_algorithms.operators import RealAmplitudesAnsatz, StatevectorEstimator
from cqlib_algorithms.operators.optimizer import COBYLAOptimizer

pipeline = zne_executor(
    QemExecutor(estimator=StatevectorEstimator(), use_cache=True),
    scale_factors=(1.0, 3.0, 5.0),
)
solver = VQE(
    pipeline.as_estimator(),
    RealAmplitudesAnsatz(1, reps=0),
    COBYLAOptimizer(maxiter=80),
)
result = solver.run(PauliString.from_str("Z"), {"theta_0": 0.4})
print(result.optimal_value)  # approximately -1.0
```

该例在理想模拟中求 Pauli Z 的基态能量，输出接近 `-1.0`。算法每轮提供当前参数，
缓解层要求绑定恰好覆盖所有活跃符号，并在独立线路中完成表达式求值，再执行折叠、训练线路生成、多副本构造或测量变换。配置了固定 shots 或 seed 的方法优先使用自身设置，
否则沿用请求中的设置。缓存默认关闭，键包含线路、可观测量、参数绑定、shots 和 seed。

REM、T-REx 和经典阴影使用底层采样结果生成期望值估计。`as_estimator()` 暴露这一输出；
`sampler` 属性返回基础采样后端，不能据此认为原始计数已经过缓解。

## 已有线路与批次

`run(preparation_or_circuit, observable, ...)` 和十三种 `execute_with_*` 均接收
`StatePreparation` 或 `Circuit`。`prefix` 用于构建组件，不重命名已有线路的符号。
`estimate(requests)` 与 `sample(requests)` 分别执行期望值和采样批次。
`estimate` 返回按输入顺序排列的结果列表；`execute_with_*` 返回一个期望值浮点数。
独立执行函数同样接收状态制备组件，并支持 `parameter_bindings` 和 `prefix`。

```python
from cqlib import Circuit
from cqlib.qis import PauliString
from cqlib_algorithms.error_mitigation import QemExecutor, execute_with_zne
from cqlib_algorithms.operators import StatevectorEstimator

circuit = Circuit(1)
circuit.ry(0, 0.4)
value = execute_with_zne(
    circuit,
    QemExecutor(estimator=StatevectorEstimator()),
    PauliString.from_str("Z"),
    scale_factors=(1.0, 3.0, 5.0),
)
print(value)
```

## 方法选择

下表的 `base` 是按方法要求配置后端的 `QemExecutor`。`ideal` 为参考 Estimator；
`representations` 是与线路门匹配的拟概率表示序列；`confusion` 为读出混淆矩阵；
`checks` 与 `code_hamiltonian` 指定 QSE 扩展空间和代码 Hamiltonian。这些信息需要按实际问题提供。
REM、T-REx、经典阴影和 DEB 需要在 `base` 中提供采样后端；QSE 根据测量配置可能需要同时提供估计和采样后端。

| 方法 | 主要输入和假设 | 输出语义 | 组件构造示例 |
| --- | --- | --- | --- |
| ZNE | 可通过线路折叠放大噪声，噪声随缩放因子平滑变化 | 外推到零噪声极限的期望值 | `zne_executor(base, scale_factors=(1, 3, 5), shots=4096)` |
| LRE | 可独立缩放线路层 | 多变量 Richardson 外推值 | `lre_executor(base, degree=1, fold_multiplier=1, num_chunks=1, shots=4096)` |
| CDR | 可生成具有不同期望值的近 Clifford 训练线路，并取得参考值 | 从有噪声值到参考值的回归修正 | `cdr_executor(base, ideal_estimator=ideal, num_training_circuits=12, fraction_non_clifford=0.5, shots=4096)` |
| DDD | 线路存在可利用的空闲窗口，给定动态解耦脉冲规则 | 插入解耦序列后的期望值 | `ddd_executor(base, rule=XX, shots=4096)` |
| Pauli Twirling | 目标双量子比特门允许等价 Pauli 共轭 | 多个随机 twirl 线路的平均值 | `pt_executor(base, num_randomizations=16, shots=4096)` |
| DEB | 可通过量子比特标签置换采样设备空间偏差 | 置换结果的对称化估计 | `deb_executor(base, mode="averaging", shots=4096, num_variants=8)` |
| PEA | 已知门级拟概率表示，可采样放大后的噪声模型 | 放大噪声下的平均期望值 | `pea_executor(base, noise_level=0.01, scale_factors=(1, 2, 3), extrapolation_method="richardson", shots=4096)` |
| PEC | 已知门级拟概率表示及其符号和开销 | 带符号与开销补偿的期望值 | `pec_executor(base, quasiprobabilities=representations, precision=0.05, shots=4096)` |
| REM | 读出误差可近似为逐量子比特混淆矩阵 | 由校正分布计算的期望值 | `rem_executor(base, shots=4096, confusion=confusion)` |
| T-REx | 可执行随机读出翻转及相应校准线路 | 读出 twirling 后的期望值 | `trex_executor(base, shots=4096, num_randomizations=32)` |
| Classical Shadows | 可执行随机局域 Pauli 测量 | Pauli 可观测量的 shadow 估计 | `shadows_executor(base, num_total_measurements_shadow=4096)` |
| QSE | 给定扩展算符基和代码 Hamiltonian | 广义本征值问题的最低能量估计 | `qse_executor(base, check_operators=checks, code_hamiltonian=code_hamiltonian, shots=4096)` |
| Virtual Distillation | 能执行多副本 copy-swap 线路，副本噪声近似独立 | $`\operatorname{Tr}(O\rho^M)/\operatorname{Tr}(\rho^M)`$ | `vd_executor(base, copies=2, shots_numerator=4096, shots_denominator=4096)` |

## 缩放、推断与拟概率

线路缩放提供全局折叠、局域折叠、随机门折叠、分层折叠、单位层插入和参数缩放。
外推工厂包括 Richardson、多项式、线性、指数、多项式-指数和自适应指数模型。
拟概率组件提供退极化模型下的线路表示、噪声放大和采样权重；这些表示依赖调用方
提供或接受的噪声模型，不能从任意后端数据中自动推断完整噪声通道。

## 随机性和可复现性

所有公开 seed 都接受 `None` 或非负 Python 整数。固定父 seed 会通过
`spawn_seeds` 为不同线路、Pauli 项或 numerator/denominator 测量派生互相独立的
子 seed，因此可以复现完整批次，而不会让所有子任务重复同一随机序列。

## 适用边界

- 有限-shot 方差可能因缓解权重、外推或拟概率开销显著放大。
- PEC 和 PEA 的正确性依赖拟概率模型与实际噪声的一致性。
- REM 当前使用逐量子比特混淆数据，不表示一般相关读出噪声模型。
- Virtual Distillation 的线路宽度为 $`Mn+1`$，并要求副本噪声近似独立。
- QSE 的结果依赖扩展基、重叠矩阵条件数和广义本征值截断容差。
- 组件只提供同步本地契约；认证、任务提交、排队和异步轮询由外部适配层负责。

每项方法的精确参数、异常、数学公式和论文来源见对应源码 docstring。
