# VQR：直接训练量子线路来预测数值

## 先理解它解决什么问题

回归是在学习输入与数值答案的关系。普通模型可以调整一条直线的斜率；这里调整的是
量子线路中的旋转角度。每个样本先作为线路参数输入，线路执行后，一个可观测量的
期望值就是模型的预测。经典优化器根据“预测离答案多远”调整权重，再重复这一过程。

本实现采用 Mitarai 等的量子线路学习思路中的单输出平方损失版本，
来源为 [Quantum Circuit Learning, arXiv:1803.00745v3](https://arxiv.org/abs/1803.00745v3)，
Phys. Rev. A 98, 032309 (2018)。

与 [QSVR](qsvr.md) 的区别是：QSVR 用固定量子核训练经典系数；VQR 直接改变
产生预测的线路权重，预测值直接由可观测量期望值给出。

## 数学定义与职责

完整线路记作 `U(x, theta, b)`，其中 x 是输入，theta 是可训练权重，b 是固定值。
给定有限 Hermitian observable O，预测为：

`f(x) = <0| U(x,theta,b)† O U(x,theta,b) |0>`。

训练目标是 `mean((f(X_i)-y_i)**2)`。预测值的范围由可观测量的谱决定，
例如 O=Z 时为 [-1,1]；线路和目标数据可据此选择尺度。

算法通过公共 `Estimator` 批量计算预测，线路和 observable 使用 Cqlib Core 类型。
SciPy COBYLA 根据整批预测的均方误差更新权重，具体线路执行由 Estimator 完成。

## 可以直接运行的精确示例

示例使用一个量子比特，以解析 `cos(x+theta)` 产生训练和留出数据，展示参数训练与预测。

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
`SamplingEstimator(sampler)` 并传 `shots=正整数`。
每个样本使用指定采样次数，所得期望值估计参与均方误差计算。

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
- X 是有限实数二维矩阵 `(n,d)`，y 是同长度一维实数向量。
  训练至少一个样本，预测允许 `(0,d)`。
- `initial_point` 按名称给出全部权重的初值，与权重参数一一对应。
- observable 为同宽有限 Hermitian `Hamiltonian` 或 `PauliString`。其位置 i 对应
  原线路第 i 个插入量子比特；Pauli 的最右字符为位置 0。原线路 ID 可以乱序或不连续。
- 线路在构造时快照并归一化，包含静态酉操作及可选 barrier。
- 每组候选权重与整批样本绑定、完成数值校验后，再调用 Estimator。
- `score` 至少两个答案。R² 可以为负；恒定目标若精确预测为 1，否则为 0。

## 优化配置与结果

`max_evaluations` 是 COBYLA 的目标函数调用预算，至少为权重维数加 2。
`tolerance`、`initial_step` 是有限正数，分别控制停止尺度和初始搜索步长。
初始诊断与最终复核各评估一次，计数关系为
`objective_evaluations_ = optimizer_evaluations_ + 2`。
一次目标评估为全部训练行计算预测值。

`initial_loss_` 和 `training_loss_` 分别来自初始、最终评估。
`optimal_parameters_` 为不可变映射，`optimizer_success_`、`optimizer_status_`、
`optimizer_message_` 记录终止状态。返回有效参数后，模型可用于预测。

训练在全部计算完成后更新模型；重拟合失败时保留原模型。
输入和执行结果的校验错误通过异常返回，预测与训练诊断在拟合后使用。

## 测试与依赖

对应 `tests/algorithms/test_vqr.py`。测试包括真实 COBYLA 对解析余弦数据的训练与留出
预测、Y observable 的符号、乱序量子位的 IZ/ZI、非字典序参数和构造输入快照。
记录型 Estimator 检查整批完全绑定及返回契约；确定性频数配真实 SamplingEstimator
验证 shots；另有奇点/溢出、错误形状、无效初值、缺 SciPy、预算终止、失败重拟合及
空预测等情况。期望值参考由独立解析公式计算。

经典优化依赖为可选 extra `cqlib-algorithms[machine-learning]`，SciPy 仅在 fit 内导入。
