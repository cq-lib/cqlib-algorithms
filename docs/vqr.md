# 变分量子回归 VQR

## 算法背景

变分量子回归以参数化量子线路构建回归模型。输入样本通过数据参数编码，
模型预测值由指定可观测量的期望值给出。经典优化器根据预测值与目标值的均方误差
更新线路权重，形成量子执行与经典优化相结合的训练过程。

本实现采用 Mitarai 等的量子线路学习思路中的单输出平方损失版本，
来源为 [Quantum Circuit Learning, arXiv:1803.00745v3](https://arxiv.org/abs/1803.00745v3)，
Phys. Rev. A 98, 032309 (2018)。

与 [QSVR](qsvr.md) 基于固定量子核优化经典回归系数的方式相比，VQR 优化的是
预测线路中的可训练参数。

## 数学定义与实现

完整线路记作 `U(x, theta, b)`，其中 x 为输入，theta 为可训练权重，b 为固定参数。
给定有限 Hermitian 可观测量 O，预测值定义为：

`f(x) = <0| U(x,theta,b)† O U(x,theta,b) |0>`。

训练目标是 `mean((f(X_i)-y_i)**2)`。预测值的范围由可观测量的谱决定，
例如 O=Z 时为 [-1,1]；线路和目标数据可据此选择尺度。

算法通过公共 `Estimator` 批量计算预测值，线路和可观测量使用 Cqlib Core 类型。
SciPy COBYLA 根据整批预测的均方误差更新权重，具体线路执行由 Estimator 完成。

## 使用示例：精确期望值计算

示例使用单量子比特线路，以解析函数 `cos(x+theta)` 生成训练数据和留出数据，
验证参数拟合与预测结果。

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

有限采样模式使用 `SamplingEstimator(sampler)`，并设置正整数参数 `shots`。
每个样本使用指定采样次数，所得期望值估计参与均方误差计算。

## 接口与输入要求

```text
VQR(circuit, input_parameters, weight_parameters, observable, estimator,
    parameter_bindings=None, shots=None,
    max_evaluations=200, tolerance=1e-6, initial_step=1.0)
fit(X, y, initial_point=...) -> self
predict(X) -> 一维 float64 预测
score(X, y) -> R²
```

- 数据参数名与权重参数名均为非空、无重复、有序字符串序列，固定绑定是名称到有限实数的映射。
  三组名称互斥，并恰好覆盖线路活跃符号。数据列和初值向量都按显式名称顺序绑定。
- X 是有限实数二维矩阵 `(n,d)`，y 是同长度一维实数向量。
  训练至少一个样本，预测允许 `(0,d)`。
- `initial_point` 按名称给出全部权重的初值，与权重参数一一对应。
- 可观测量为与线路量子比特数相同的有限 Hermitian `Hamiltonian` 或 `PauliString`。其位置 i 对应
  原线路第 i 个插入量子比特；Pauli 的最右字符为位置 0。原线路 ID 可以乱序或不连续。
- 线路在构造时快照并归一化，包含静态酉操作及可选 barrier。
- 每组候选权重与整批样本绑定、完成数值校验后，再调用 Estimator。
- `score` 要求至少两个目标值。R² 可取负值；目标值恒定时，精确预测的评分为 1，否则为 0。

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

测试位于 `tests/algorithms/test_vqr.py`，覆盖 COBYLA 对解析余弦数据的拟合与留出
预测、Y 可观测量的符号、非连续量子比特标识下的 IZ/ZI、显式参数顺序和构造输入快照。
记录型 Estimator 检查批量参数绑定及返回契约；确定性频数与 SamplingEstimator 组合
验证 shots；另有奇点/溢出、错误形状、无效初值、缺 SciPy、预算终止、失败重拟合及
空预测等情况。期望值参考由独立解析公式计算。

经典优化依赖通过 `cqlib-algorithms[machine-learning]` 可选依赖组提供，SciPy 在 `fit` 时加载。
