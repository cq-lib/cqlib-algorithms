# 投影随机次梯度量子核分类 PegasosQSVC

## 算法背景

支持向量分类通过正则化与间隔损失构建分类函数。Pegasos 采用随机次梯度方法，
每次迭代选取一个样本，依次执行权重收缩、基于分类间隔的系数更新和球投影。

`PegasosQSVC` 通过 [保真度量子核](qsvr.md) 计算样本相似度，并在对应核空间中
执行 Pegasos 更新。训练过程优化经典核系数，量子特征线路保持固定。

采用 Shalev-Shwartz、Singer、Srebro 的
[ICML 2007 原文 Figure 1](https://home.ttic.edu/~nati/Publications/Pegasos.pdf)：
采用单样本 `k=1`、球投影、最终迭代点和无独立偏置的配置。量子核依据
[Havlíček 等 arXiv:1804.11326v2](https://arxiv.org/abs/1804.11326v2)。

## 使用示例

示例采用 RY 特征线路，对分别位于 0 和 π 邻域的两类角度数据进行训练，
验证留出样本的分类结果与模型范数。

```python
import numpy as np
from cqlib import Circuit, Parameter
from cqlib_algorithms.algorithms import PegasosQSVC
from cqlib_algorithms.operators import FidelityQuantumKernel

circuit = Circuit(1)
circuit.ry(0, Parameter("x"))
kernel = FidelityQuantumKernel(circuit, ["x"])
X = [[-0.2], [0.0], [0.2], [np.pi-0.2], [np.pi], [np.pi+0.2]]
y = [3, 3, 3, 8, 8, 8]
model = PegasosQSVC(kernel, regularization=0.1, num_steps=400, seed=19)
model.fit(X, y)
held_out = [[-0.1], [0.1], [3.0], [3.3]]
prediction = model.predict(held_out)
print("Predicted classes:", prediction)
print("Held-out accuracy:", model.score(held_out, [3, 3, 8, 8]))
print("Final weight norm:", model.weight_norm_)
assert model.score(held_out, [3, 3, 8, 8]) == 1.0
assert model.weight_norm_ <= 1/np.sqrt(0.1) + 1e-12
```

## 优化目标与迭代规则

内部标签映射为 `y_i in {-1,+1}`，优化目标为：

`lambda * ||w||² / 2 + mean(max(0, 1 - y_i * <w,phi(x_i)>))`。

`regularization` 是 lambda，有限且严格为正。若与常见的
`||w||²/2 + C*sum(hinge_loss)` 形式比较，则 `C=1/(lambda*n)`。

模型通过有符号系数 c 表示权重，`w=sum_j c_j phi(x_j)`。第 t 次迭代包括：

1. 根据更新前的 c 计算所采样样本的分类间隔 margin。
2. 执行 `c *= 1-1/t`；若更新前的 margin 严格小于 1，则在对应样本系数上累加 `y_i/(lambda*t)`。
3. 使用训练核计算权重范数，并执行半径为 `1/sqrt(lambda)` 的球投影。

margin 正好为 1 时执行收缩；第 1 步的收缩会清除初始权重。
算法执行 `num_steps` 次更新，返回最后迭代点。

## 核矩阵与预测

训练对 Gram 矩阵做对称本征分解，将负本征值裁为 0，得到 PSD 正部。
训练 margin、更新、范数、投影和目标值均使用该矩阵。
预测时计算 `kernel.evaluate(X_test, X_train)`，与训练得到的系数 c 相乘，获得决策分数。

## 接口与结果

```text
PegasosQSVC(quantum_kernel, regularization=0.1, num_steps=1000, seed=None)
fit(X, y) -> self
decision_function(X) -> 一维决策分数
predict(X) -> 原始数值类别
score(X, y) -> 准确率
```

X 为有限实二维矩阵，特征数符合固定核；训练非空。y 为同长度一维有限实数向量，
恰好包含两种取值。两个原始类别按升序对应负类和正类，决策分数为 0 时归入正类。
`score` 要求非空，标签必须属于已拟合类别，允许评价集中只出现其中一类。
显式 `(0,d)` 预测返回空数组且不执行量子核。

`num_steps` 为非布尔正整数，seed 为非负整数或 None。每次 `fit` 创建独立的
NumPy 随机数生成器，相同 seed 对应相同的训练样本索引序列。
训练完成后统一更新模型状态，重拟合失败时保留原模型。
更新值、范数或目标值出现非有限数时抛出异常。
平方范数的浮点容差按矩阵和系数规模计算，超出容差时抛出异常。

可读取的诊断包括 `classes_`、`n_features_in_`、`training_data_`、`coefficients_`、
`raw_kernel_matrix_`、`kernel_matrix_`、`minimum_raw_eigenvalue_`、
`psd_correction_norm_`、`weight_norm_`、`training_objective_`、`n_iter_`。
数组属性返回副本，类别为不可变二元组；训练诊断在拟合后读取。

## 测试与计算成本

`tests/algorithms/test_pegasos_qsvc.py` 使用 RY 核对应的三维实特征构造独立欧氏空间参考，
比较显式权重更新与核系数更新的决策分数和范数。
测试还覆盖首步球投影、margin=1、闭式本征分解的 PSD 正部、
原始交叉核预测、精确量子核的留出分类、确定性频数下的采样执行、标签还原、
seed 复现、错误输入、诊断副本和失败重拟合测试。

保存完整训练 Gram 需要 O(n²) 内存；当前每步范数计算为 O(n²)，加上核评估成本。
分类器使用 Core、NumPy 和保真度量子核完成训练与预测。
