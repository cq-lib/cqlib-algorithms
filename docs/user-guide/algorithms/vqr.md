# 变分量子回归 VQR

`VQR` 使用参数化量子线路进行回归，以指定可观测量的期望值作为预测值，
通过最小化训练集均方误差优化线路权重。

训练使用 SciPy，可通过 `cqlib-algorithms[machine-learning]` 安装经典计算依赖。

## 使用示例

以下示例采用单量子比特 RY 线路，拟合 `cos(x+0.43)`。

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
    circuit,
    ["x"],
    ["theta"],
    PauliString.from_str("Z"),
    StatevectorEstimator(),
    max_evaluations=120,
)
X = np.linspace(-1.4, 1.5, 9)[:, None]
y = np.cos(X[:, 0] + 0.43)
model.fit(X, y, initial_point={"theta": -0.7})
X_test = np.array([[-0.5], [0.2], [1.0]])
print("Learned weights:", dict(model.optimal_parameters_))
print("Training MSE:", model.training_loss_)
print("Predictions:", model.predict(X_test))
```

## 接口与参数

```text
VQR(circuit, input_parameters, weight_parameters, observable, estimator, *,
    parameter_bindings=None, shots=None,
    max_evaluations=200, tolerance=1e-6, initial_step=1.0)
fit(X, y, *, initial_point) -> self
predict(X) -> numpy.ndarray
score(X, y) -> float
```

| 参数 | 含义 |
| --- | --- |
| `circuit` | 包含数据参数与权重参数的静态酉 Cqlib 线路 |
| `input_parameters` | 有序数据参数名，与输入数据列一一对应 |
| `weight_parameters` | 有序可训练权重名 |
| `observable` | 与线路量子比特数相同的 Hermitian `Hamiltonian` 或 `PauliString` |
| `estimator` | 期望值计算组件，如 `StatevectorEstimator` 或 `SamplingEstimator` |
| `parameter_bindings` | 其余线路参数的固定值 |
| `shots` | 每个样本的采样次数；精确计算时使用默认值 `None` |
| `max_evaluations` | 优化器目标评估次数上限，至少为权重数量加 2 |
| `tolerance`、`initial_step` | 正的优化停止尺度与初始搜索步长 |

三组参数名称互斥，并共同覆盖线路的全部活跃符号。
`X` 为有限实数二维数组 `(n,d)`，`y` 为长度 n 的一维有限实数向量。
训练时通过 `initial_point` 按名称提供全部权重的初值。

可观测量的位置按线路量子比特的插入顺序解释，Pauli 字符串最右字符对应位置 0。
预测值范围由可观测量的谱决定，例如 Z 可观测量对应 [-1,1]。

## 预测与训练结果

`predict(X)` 返回一维预测数组，`score(X,y)` 返回 R²，评分数据至少包含两个样本。
使用采样计算时，可传入 `SamplingEstimator(sampler)` 并指定正整数 shots。

| 属性 | 含义 |
| --- | --- |
| `optimal_parameters_` | 训练后的权重映射 |
| `initial_loss_`、`training_loss_` | 初始与最终训练均方误差 |
| `optimizer_success_`、`optimizer_status_`、`optimizer_message_` | 优化器终止状态 |
| `optimizer_evaluations_`、`objective_evaluations_` | 优化器评估次数与包含初始、最终诊断的总评估次数 |

## 参考文献

- [Mitarai 等：Quantum Circuit Learning](https://arxiv.org/abs/1803.00745v3)。
