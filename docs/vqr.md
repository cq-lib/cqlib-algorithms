# VQR：直接训练量子线路来预测数值

## 先理解它解决什么问题

回归是在学习输入与数值答案的关系。普通模型可以调整一条直线的斜率；这里调整的是
量子线路中的旋转角度。每个样本先作为线路参数输入，线路执行后，一个可观测量的
期望值就是模型的预测。经典优化器根据“预测离答案多远”调整权重，再重复这一过程。

本实现采用 Mitarai 等的量子线路学习思路中的单输出平方损失版本，
来源为 [Quantum Circuit Learning, arXiv:1803.00745v3](https://arxiv.org/abs/1803.00745v3)，
Phys. Rev. A 98, 032309 (2018)。不包含论文涉及的全部结构或逼近能力。

与 [QSVR](qsvr.md) 的区别是：QSVR 用固定量子核训练经典系数；VQR 直接改变
产生预测的线路权重。VQR 不要求先构造量子核，也没有额外的可训练经典截距。

## 数学定义与职责

完整线路记作 `U(x, theta, b)`，其中 x 是输入，theta 是可训练权重，b 是固定值。
给定有限 Hermitian observable O，预测为：

`f(x) = <0| U(x,theta,b)† O U(x,theta,b) |0>`。

训练目标是 `mean((f(X_i)-y_i)**2)`，没有隐藏正则项、自动归一化或答案裁剪。
例如 O=Z 时精确预测只能落在 [-1,1]；超出范围的目标仍然合法，但这个模型可能
无法拟合，调用方需要自行选择尺度与线路。

算法使用真实的公共 `Estimator` 批量计算预测，线路和 observable 使用 Cqlib Core 类型。
经典 SciPy COBYLA 只优化这个完整平方损失，不将损失伪装成某个量子可观测量。
Estimator 管理实际执行方式；算法不另写模拟器、梯度或后端平台适配。

## 可以直接运行的精确示例

示例只用一个量子比特，因此能用解析 `cos(x+theta)` 产生训练答案和留出答案。
这是一项基础正确性演示，不能据此推断复杂数据上的实用性。

```python
import numpy as np
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString
from cqlib_algorithms.algorithms import VQR
from cqlib_algorithms.operators.estimator import StatevectorEstimator

circuit = Circuit(1)
circuit.ry(0, Parameter("x"))
circuit.ry(0, Parameter("theta"))
model = VQR(
    circuit, ["x"], ["theta"], PauliString.from_str("Z"),
    StatevectorEstimator(), max_evaluations=120,
)
X = np.linspace(-1.4, 1.5, 9)[:, None]
y = np.cos(X[:, 0] + 0.43)
model.fit(X, y, initial_point={"theta": -0.7})
held_out = np.array([[-0.5], [0.2], [1.0]])
prediction = model.predict(held_out)
print("Learned weights:", dict(model.optimal_parameters_))
print("Training MSE:", model.training_loss_)
print("Held-out predictions:", prediction)
print("Optimizer success:", model.optimizer_success_)
assert model.training_loss_ < 1e-8
np.testing.assert_allclose(prediction, np.cos(held_out[:, 0]+0.43), atol=5e-4)
```

使用有限 shots 时，将 `StatevectorEstimator()` 换成已有
`SamplingEstimator(sampler)` 并传 `shots=正整数`。每个样本的请求都携带该 shots；
测量结果进入平方损失，因此经验 MSE 受噪声影响，并不是精确 MSE 的无偏估计。
该路径没有固定噪声下必定收敛的保证。

## 接口与数据要求

```text
VQR(circuit, input_parameters, weight_parameters, observable, estimator,
    parameter_bindings=None, shots=None,
    max_evaluations=200, tolerance=1e-6, initial_step=1.0)
fit(X, y, initial_point=...) -> self
predict(X) -> 一维 float64 预测
score(X, y) -> R²
```

- 数据名与权重名均为非空、无重复、有序字符串序列，固定绑定是名称到有限实数的映射。
  三组名称互斥，并恰好覆盖线路活跃符号。数据列和初值向量都按显式名称顺序绑定。
- X 是有限实数二维矩阵 `(n,d)`，y 是同长度一维数值向量；拒绝 bool、complex、
  字符串、object、非矩形数据、NaN 和无穷大。训练至少一个样本，预测允许 `(0,d)`。
- `initial_point` 必须显式给出每个权重，不能遗漏或添加其他参数。
- observable 为同宽有限 Hermitian `Hamiltonian` 或 `PauliString`。其位置 i 对应
  原线路第 i 个插入量子比特；Pauli 的最右字符为位置 0。原线路 ID 可以乱序或不连续。
- 线路在构造时快照并归一化。只接受静态酉操作及 barrier；不接受测量、reset、delay
  或动态控制。构造时不会为了验证而把符号随意绑定成 0。
- 每一个实际候选都会把整批样本完全绑定并验证后再调用 Estimator；例如后一行
  发生 `1/x` 的除零时，不会已经执行前一行。
- `score` 至少两个答案。R² 可以为负；恒定目标若精确预测为 1，否则为 0。

## 如何理解停止和诊断

`max_evaluations` 是 COBYLA 的目标函数调用预算，至少为权重维数加 2。
`tolerance`、`initial_step` 是有限正数，分别控制停止尺度和初始搜索步长，
不是最终损失误差的承诺。初始诊断与最终独立复核各额外评估一次，
`optimizer_evaluations_` 只记录优化调用，`objective_evaluations_` 为它加 2。
一次目标评估需要为全部训练行请求预测，因此不等于一次量子线路执行。

`initial_loss_` 和 `training_loss_` 分别来自真实初始、最终评估；最终损失没有
盲用优化器缓存的 `fun`。`optimal_parameters_` 为不可变映射；
`optimizer_success_`、`optimizer_status_`、`optimizer_message_` 保留真实终止状态。
预算耗尽但返回有限有效点时，仍可预测，`optimizer_success_` 会说明未成功停止。
这不表示找到全局最优，也不保证损失低于初值。

非法输入、非有限表达式、坏 Estimator/优化器结果或最终评估失败会报错；首次拟合
保持未拟合，失败的重新拟合保留旧完整模型。拟合前访问预测或训练诊断会报错。

## 测试与边界

对应 `tests/algorithms/test_vqr.py`。测试包括真实 COBYLA 对解析余弦数据的训练与留出
预测、Y observable 的符号、乱序量子位的 IZ/ZI、非字典序参数和构造输入快照。
记录型 Estimator 检查整批完全绑定及返回契约；确定性频数配真实 SamplingEstimator
验证 shots；另有奇点/溢出、错误形状、坏初值、缺 SciPy、预算终止、失败重拟合及
空预测等边界。解析参考不通过 VQR 自身生成答案。

经典优化依赖为可选 extra `cqlib-algorithms[machine-learning]`，SciPy 仅在 fit 内导入。
源码兼容 Python 3.10 语法；当前实测 SDK 2.0.0b6 本身需要 Python 3.11+。
当前版本不提供多输出、sample weights、输入预处理、自动 ansatz、交叉验证或噪声缓解。
