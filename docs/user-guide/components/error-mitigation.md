# 量子误差缓解

本页说明 `cqlib_algorithms.error_mitigation` 提供的执行模型、误差缩放、统计推断和
缓解技术。误差缓解不修正线路本身，也不承诺把有噪声结果恢复为精确答案；调用方必须
根据设备噪声、shots 预算和目标可观测量选择方法，并对缓解前后的偏差与方差进行验证。

## 执行模型

`QemExecutor` 同时封装可选的 `Estimator` 和 `Sampler`。各缓解方法通过
`execute_with_*` 立即执行单个任务，或通过 `*_executor` 生成可继续组合的执行器。
缓存默认关闭；启用时，缓存键包含线路结构、可观测量、参数绑定、shots 和 seed。

```python
from cqlib import Circuit
from cqlib.qis import PauliString
from cqlib_algorithms.error_mitigation import QemExecutor, execute_with_zne
from cqlib_algorithms.operators.estimator import StatevectorEstimator

circuit = Circuit(1)
circuit.ry(0, 0.4)
executor = QemExecutor(estimator=StatevectorEstimator())

value = execute_with_zne(
    circuit,
    executor,
    PauliString.from_str("Z"),
    scale_factors=(1.0, 3.0, 5.0),
)
```

这里使用精确 Estimator 只用于演示接口。真实 ZNE 需要能够反映折叠线路噪声变化的
有噪声执行器，否则不同缩放点的结果相同，不会产生实际缓解效果。

## 方法选择

| 方法 | 主要输入和假设 | 输出语义 |
| --- | --- | --- |
| ZNE | 可通过线路折叠放大噪声，噪声随缩放因子平滑变化 | 外推到零噪声极限的期望值 |
| LRE | 可独立缩放线路层 | 多变量 Richardson 外推值 |
| CDR | 可生成近 Clifford 训练线路，并能取得训练线路参考值 | 从有噪声值到参考值的回归修正 |
| DDD | 线路存在可利用的空闲窗口，给定动态解耦脉冲规则 | 插入解耦序列后的期望值 |
| Pauli Twirling | 目标双量子比特门允许等价 Pauli 共轭 | 多个随机 twirl 线路的平均值 |
| DEB | 可通过量子比特标签置换采样设备空间偏差 | 置换结果的对称化估计 |
| PEA | 已知门级拟概率表示，可采样放大后的噪声模型 | 放大噪声下的平均期望值 |
| PEC | 已知门级拟概率表示及其符号和开销 | 带符号与开销补偿的期望值 |
| REM | 读出误差可近似为逐量子比特混淆矩阵 | 校正后的 counts 或期望值 |
| T-REx | 可执行随机读出翻转及相应校准线路 | 读出 twirling 后的期望值 |
| Classical Shadows | 可执行随机局域 Pauli 测量 | Pauli 可观测量的 shadow 估计 |
| QSE | 给定扩展算符基和代码 Hamiltonian | 广义本征值问题的最低能量估计 |
| Virtual Distillation | 能执行多副本 copy-swap 线路，副本噪声近似独立 | <img src="../../assets/formulas/inline/inline-290e1a115e43c614.svg" alt="\operatorname{Tr}(O\rho^M)/\operatorname{Tr}(\rho^M)" style="vertical-align: -4.000pt;"> |

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
- Virtual Distillation 的线路宽度为 <img src="../../assets/formulas/inline/inline-c0247fd5b4940c66.svg" alt="Mn+1" style="vertical-align: -2.000pt;">，并要求副本噪声近似独立。
- QSE 的结果依赖扩展基、重叠矩阵条件数和广义本征值截断容差。
- 组件只提供同步本地契约；认证、任务提交、排队和异步轮询由外部适配层负责。

每项方法的精确参数、异常、数学公式和论文来源见对应源码 docstring。
