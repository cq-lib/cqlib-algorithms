# 量子核训练 QuantumKernelTrainer

`QuantumKernelTrainer` 组合可训练量子核、损失函数和优化器，利用训练数据调整核参数。
默认损失为类别平衡的核目标对齐，默认优化器为 SciPy COBYLA。两者均可由调用方替换。
训练结果保存参数、损失、优化器结果和可直接求值的训练后量子核。

## 使用示例

以下示例训练 $R_Y(wx)$ 中的参数 $w$，并使用保真度核训练分类器。

```python
from cqlib import Circuit, Parameter
from cqlib_algorithms.algorithms import QuantumKernelTrainer, PegasosQSVC
from cqlib_algorithms.operators import TrainableFidelityQuantumKernel

circuit = Circuit(1)
circuit.ry(0, Parameter("w") * Parameter("x"))
kernel = TrainableFidelityQuantumKernel(circuit, ["x"], ["w"])
trainer = QuantumKernelTrainer(quantum_kernel=kernel, max_evaluations=120)
trainer.fit([[0.0], [1.0]], [-1, 1], initial_point={"w": 0.2})
result = trainer.result_
print("Learned parameters:", dict(result.optimal_parameters))
print("Final alignment:", trainer.alignment_)

classifier = PegasosQSVC(result.quantum_kernel, num_steps=100, seed=4)
classifier.fit([[0.0], [1.0]], [-1, 1])
print("Predicted classes:", classifier.predict([[0.0], [1.0]]))
```

也可直接使用线路入口：`QuantumKernelTrainer(circuit, ["x"], ["w"])`。
默认优化器使用的 SciPy 由 `cqlib-algorithms[machine-learning]` 可选依赖组提供。

## 接口与参数

```text
QuantumKernelTrainer(feature_map=None, input_parameters=None,
    trainable_parameters=None, *, quantum_kernel=None, loss=None, optimizer=None,
    parameter_bindings=None, sampler=None, shots=None,
    max_statevector_qubits=None, max_statevector_bytes=None,
    max_evaluations=200, tolerance=1e-6, initial_step=1.0)
fit(X, y, *, initial_point) -> self
```

| 参数 | 含义 |
| --- | --- |
| `quantum_kernel` | 实现 `TrainableQuantumKernel` 的可训练核，通过 `bind_parameters()` 生成固定参数核 |
| `loss` | 接收训练核矩阵和目标向量、返回有限实数的可调用对象；默认采用 `BalancedKernelAlignmentLoss` |
| `optimizer` | 实现 `ScalarMinimizer` 调用约定的对象，返回 `ScalarMinimizerResult`；默认采用 COBYLA |
| `feature_map`、`input_parameters`、`trainable_parameters` | 线路入口，分别指定 Cqlib 特征线路、有序数据参数与可训练参数 |
| `parameter_bindings` | 线路入口中其余活跃参数的固定值 |
| `sampler`、`shots` | 线路入口中同时提供时采用采样核计算；默认采用精确计算 |
| `max_statevector_qubits`、`max_statevector_bytes` | 线路入口的精确计算预算，默认分别为 20 和 268,435,456，详见 [量子核参数](qsvr.md) |
| `max_evaluations` | 优化期间目标评估次数上限；COBYLA 至少需要可训练参数数量加 2，其他优化器取正整数 |
| `tolerance`、`initial_step` | COBYLA 的停止尺度和初始搜索步长，均为正数 |

`quantum_kernel` 与线路配置分别构成两种入口。线路的输入、可训练和固定参数名称互斥，并共同覆盖全部活跃符号。
训练数据 `X` 为非空有限实数二维数组，列数等于核的 `num_features`；`y` 为同样本数的一维有限实数向量。
默认损失使用两个不同类别值，自定义损失可定义分类或回归目标。
`initial_point` 按名称指定全部可训练参数的初值。

## 自定义损失与优化器

损失函数接收 `(matrix, targets)`。标量优化器接收 `(objective, initial_point)`，调用目标函数后，
返回选中的参数点、目标值、终止信息和实际评估次数。以下示例使用矩阵平方误差与三点搜索。

```python
import numpy as np
from cqlib import Circuit, Parameter
from cqlib_algorithms.algorithms import QuantumKernelTrainer
from cqlib_algorithms.operators import TrainableFidelityQuantumKernel
from cqlib_algorithms.protocols import ScalarMinimizerResult


def loss(matrix, targets):
    return float(np.mean((matrix - np.outer(targets, targets)) ** 2))


def search(objective, initial_point):
    points = [initial_point.copy(), np.array([np.pi / 2]), np.array([np.pi])]
    values = [objective(point) for point in points]
    index = int(np.argmin(values))
    return ScalarMinimizerResult(
        tuple(float(x) for x in points[index]),
        values[index],
        True,
        0,
        "Search completed",
        len(points),
    )


circuit = Circuit(1)
circuit.ry(0, Parameter("w") * Parameter("x"))
kernel = TrainableFidelityQuantumKernel(circuit, ["x"], ["w"])
trainer = QuantumKernelTrainer(
    quantum_kernel=kernel,
    loss=loss,
    optimizer=search,
    max_evaluations=3,
)
trainer.fit([[0.0], [0.5], [1.0]], [-0.2, 0.4, 1.0], initial_point={"w": 0.2})
print("Final loss:", trainer.loss_)
print("Trained matrix:", trainer.result_.kernel_matrix)
```

自定义核继承 `TrainableQuantumKernel`，提供 `num_features`、有序 `trainable_parameters` 和
`bind_parameters(values)`；绑定后返回独立的 `QuantumKernel`，通过 `evaluate(X, Y=None)` 求值。
这两个核契约与 `KernelLoss`、`ScalarMinimizer` 均从 `cqlib_algorithms.protocols` 导入。

## 训练结果

`result_` 返回独立的 `QuantumKernelTrainerResult`。保存该对象后，可继续训练同一训练器，
分别使用各次训练得到的核。

| 结果字段 | 含义 |
| --- | --- |
| `quantum_kernel` | 使用训练后参数的固定核；保真度核可直接传入 QSVR 或 PegasosQSVC |
| `optimal_parameters` | 优化器返回点对应的不可变参数映射 |
| `initial_loss`、`loss` | 初始与最终参数处分别评估的损失 |
| `kernel_matrix` | 与最终损失同次求值产生的只读核矩阵快照 |
| `optimizer_result` | 优化器返回的点、观测损失、评估次数及终止信息 |

训练器也提供 `quantum_kernel_`、`optimal_parameters_`、`initial_loss_`、`loss_`、
`kernel_matrix_` 等便捷属性，其中 `kernel_matrix_` 返回副本。
采用 `BalancedKernelAlignmentLoss` 时，还提供 `classes_`、`initial_alignment_` 和 `alignment_`。
总目标评估次数 `objective_evaluations_` 等于优化器调用次数加初始、最终两次评估。

## 参考文献

- [Hubregtsen 等：Training Quantum Embedding Kernels on Near-Term Quantum Computers](https://arxiv.org/abs/2105.02276v1)：默认平衡核目标对齐的数学依据。
