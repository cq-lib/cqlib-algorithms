# 量子核分类 PegasosQSVC

`PegasosQSVC` 使用量子核构建无偏置二分类模型，通过 Pegasos 随机次梯度与球投影更新分类系数。
训练输入为固定的 `FidelityQuantumKernel`，预测结果使用原始数值类别标签。

## 使用示例

以下示例对分别位于 0 和 π 邻域的两类角度数据进行分类。

```python
import numpy as np
from cqlib import Circuit, Parameter

from cqlib_algorithms.algorithms import PegasosQSVC
from cqlib_algorithms.operators import FidelityQuantumKernel

circuit = Circuit(1)
circuit.ry(0, Parameter("x"))
kernel = FidelityQuantumKernel(circuit, ["x"])
X = [[-0.2], [0.0], [0.2], [np.pi - 0.2], [np.pi], [np.pi + 0.2]]
y = [3, 3, 3, 8, 8, 8]
model = PegasosQSVC(kernel, regularization=0.1, num_steps=400, seed=19)
model.fit(X, y)
X_test = [[-0.1], [0.1], [3.0], [3.3]]
print("Predicted classes:", model.predict(X_test))
print("Accuracy:", model.score(X_test, [3, 3, 8, 8]))
```

## 接口与参数

```text
PegasosQSVC(quantum_kernel, *, regularization=0.1, num_steps=1000, seed=None)
fit(X, y) -> self
decision_function(X) -> numpy.ndarray
predict(X) -> numpy.ndarray
score(X, y) -> float
```

| 参数 | 含义 |
| --- | --- |
| `quantum_kernel` | `FidelityQuantumKernel` 实例，精确与采样配置见 [量子核使用说明](qsvr.md) |
| `regularization` | 正则化强度 λ，要求为正数 |
| `num_steps` | 训练更新次数，要求为正整数 |
| `seed` | 非负整数随机种子或 `None`，用于控制训练样本的抽取顺序 |

训练数据 `X` 为非空有限实数二维数组，列数与量子核的数据参数数量一致。
`y` 为同样本数的一维有限实数向量，包含两个不同的类别值。
量子特征线路在训练中保持固定，模型优化对应的分类系数。

## 预测与模型结果

`decision_function` 返回一维决策分数，`predict` 返回类别标签，`score` 返回分类准确率。
两个类别按数值升序对应负类与正类；分数大于等于 0 时预测为正类。
评分标签使用训练时的类别值。

| 属性 | 含义 |
| --- | --- |
| `classes_` | 按升序排列的原始类别 |
| `coefficients_` | 训练后的分类核系数 |
| `n_iter_` | 已执行的训练更新次数 |
| `weight_norm_`、`training_objective_` | 最终权重范数与训练目标值 |

## 参考文献

- [Shalev-Shwartz、Singer、Srebro：Pegasos](https://home.ttic.edu/~nati/Publications/Pegasos.pdf)。
- [Havlíček 等：Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/abs/1804.11326v2)。
