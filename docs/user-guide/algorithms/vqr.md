# 变分量子回归 VQR

`VQR` 使用参数化量子线路构建单输出回归模型。每个样本的预测值为指定可观测量的期望值，
训练以整批预测与目标之间的均方误差（MSE）为目标。量子执行由 Estimator 提供，
权重搜索由标量最小化器提供，默认采用 SciPy COBYLA。

## 默认训练

以下示例通过单量子比特 RY 线路拟合 $`\cos(x+0.43)`$。

```python
import numpy as np
from cqlib.circuit.gates import StandardGate
from cqlib.qis import PauliString

from cqlib_algorithms import VQR
from cqlib_algorithms.operators import (
    AngleEncoding,
    RealAmplitudesAnsatz,
    StatevectorEstimator,
)

model = VQR.from_components(
    AngleEncoding(1, StandardGate.RY),
    RealAmplitudesAnsatz(1, reps=0),
    PauliString.from_str("Z"),
    StatevectorEstimator(),
    max_evaluations=120,
)
X = np.linspace(-1.4, 1.5, 9)[:, None]
y = np.cos(X[:, 0] + 0.43)
model.fit(X, y, initial_point={"theta_0": -0.7})
print("Learned weights:", dict(model.optimal_parameters_))
print("Training MSE:", model.training_loss_)
print("Predictions:", model.predict([[-0.5], [0.2], [1.0]]))
```

输入数据 `X` 为有限实数二维数组，每列按 `input_parameters` 绑定；目标 `y` 为同样本数的一维向量。
组件入口先应用数据编码，再应用同宽的拟设。两者的活跃参数分别作为数据和权重，名称须互斥。
`model.input_parameters` 给出数据列的绑定顺序，可通过 `input_parameters` 显式调整。
`model.weight_parameters` 列出 `initial_point` 所需的名称。混合参数或交替编码线路可用 `VQR(circuit, input_parameters, weight_parameters, observable, estimator)` 指定各参数的角色。
可观测量的量子比特位置按线路插入顺序解释，Pauli 字符串最右字符对应位置 0。

`shots=None` 使用精确期望值。有限采样计算可传入 `SamplingEstimator(sampler)`，
并以正整数 `shots` 指定每个样本的采样次数。

## 替换优化器

`optimizer` 接受符合 `ScalarMinimizer` 约定的可调用对象：接收目标函数和一维初始参数数组，
评估候选参数后返回 `ScalarMinimizerResult`。以下示例在三组权重中选择 MSE 最小的一组。

```python
import numpy as np
from cqlib.circuit.gates import StandardGate
from cqlib.qis import PauliString

from cqlib_algorithms import ScalarMinimizerResult, VQR
from cqlib_algorithms.operators import (
    AngleEncoding,
    RealAmplitudesAnsatz,
    StatevectorEstimator,
)


def search(objective, initial_point):
    points = [initial_point.copy(), np.array([0.43]), np.array([1.2])]
    values = [objective(point) for point in points]
    index = int(np.argmin(values))
    return ScalarMinimizerResult(
        tuple(float(value) for value in points[index]),
        values[index],
        True,
        0,
        "Search completed",
        len(points),
    )


X = np.array([[-0.8], [0.2], [1.0]])
y = np.cos(X[:, 0] + 0.43)
model = VQR.from_components(
    AngleEncoding(1, StandardGate.RY),
    RealAmplitudesAnsatz(1, reps=0),
    PauliString.from_str("Z"),
    StatevectorEstimator(),
    optimizer=search,
    max_evaluations=3,
)
model.fit(X, y, initial_point={"theta_0": -0.7})
print("Selected weights:", dict(model.optimal_parameters_))
print("Training MSE:", model.training_loss_)
```

`max_evaluations` 限制优化期间实际调用目标函数的次数。默认 COBYLA 至少需要权重数量加 2，
自定义优化器使用正整数预算。`tolerance` 和 `initial_step` 分别配置默认 COBYLA 的停止尺度和初始搜索尺度。
初始和最终参数处各有一次独立损失评估，总次数由 `objective_evaluations_` 提供。

## 预测与结果

`predict(X)` 按输入顺序返回预测向量，`score(X, y)` 计算 R²，评分数据至少含两个样本。
`optimal_parameters_` 保存选定权重，`initial_loss_` 和 `training_loss_` 提供初始与最终 MSE；
优化器的终止标记、消息与实际评估次数通过 `optimizer_success_`、`optimizer_message_` 和
`optimizer_evaluations_` 读取。

公共参数和异常定义见 `VQR` 的源码 docstring。

## 参考文献

- [Mitarai 等：Quantum Circuit Learning](https://arxiv.org/abs/1803.00745v3)。
