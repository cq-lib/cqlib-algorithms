# PegasosQSVC：用量子相似度划分两类样本

## 背景与直观理解

分类是在学习样本属于哪一类。支持向量机希望区分两类的同时保留一定“余量”，
避免所有样本都贴着分类边界。Pegasos 是一种逐步训练规则：每次随机抽一个样本，
先适当收缩已有模型，若这个样本的分类余量不足就更新，然后把模型拉回规定大小的球内。

量子线路在这里提供一把固定的“相似度尺”，也就是 [保真度量子核](qsvr.md)。
Pegasos 的抽样、更新、投影和类别判定均由本模块自行实现；不调用 sklearn 的
SVC、SGDClassifier，也不导入 Qiskit。训练过程中不会调整量子线路的参数。

采用 Shalev-Shwartz、Singer、Srebro 的
[ICML 2007 原文 Figure 1](https://home.ttic.edu/~nati/Publications/Pegasos.pdf)：
单样本 `k=1`、有投影、使用最终迭代点、无独立偏置。量子核依据
[Havlíček 等 arXiv:1804.11326v2](https://arxiv.org/abs/1804.11326v2)。

## 可直接运行的示例

RY 线路把接近 0 的一组角度与接近 pi 的另一组角度分开。真实任务需要独立测试集，
本示例只是小型可分问题的正确性演示。

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

## 数学规则与参数含义

内部标签映射为 `y_i in {-1,+1}`，优化目标为：

`lambda * ||w||² / 2 + mean(max(0, 1 - y_i * <w,phi(x_i)>))`。

`regularization` 是 lambda，有限且严格为正。若与常见的
`||w||²/2 + C*sum(hinge_loss)` 形式比较，则 `C=1/(lambda*n)`，
不能把常见库的 C 数值直接填入 lambda。

本实现存有符号系数 c，`w=sum_j c_j phi(x_j)`。第 t 步：

1. 用更新前 c 算抽中样本的 margin。
2. `c *= 1-1/t`；若旧 margin 严格小于 1，则给抽中位置加 `y_i/(lambda*t)`。
3. 用训练核算范数，将模型投影到半径 `1/sqrt(lambda)` 的球内。

margin 正好为 1 时不更新；第 1 步的收缩会清除初始权重。最终返回最后迭代点，
不挑选历史最好点，也不取平均。`num_steps` 表示更新次数，不代表已经数学收敛。

## 训练核修复和预测的区别

理想纯态保真度核是 PSD；有限 shots 得到的经验矩阵可能不是。投影需要非负平方
范数，因此训练时对称本征分解，将负本征值裁为 0。这得到矩阵的 PSD 正部，
不是伪逆。全部训练 margin、更新、范数、投影和训练目标统一使用修后矩阵。

该修复只定义训练集上的代理特征空间，没有定义一致的全局 PSD 样本外延拓。
预测调用原始矩形 `kernel.evaluate(X_test, X_train)` 与 c 相乘。因此有限 shots
或修复显著时，重新计算的训练行预测也可能不同于修复矩阵下的训练代理得分。
算法不会偷偷把测试行换成训练缓存；这和已交付 QSVR 的语义一致。

## 接口、输出及错误

```text
PegasosQSVC(quantum_kernel, regularization=0.1, num_steps=1000, seed=None)
fit(X, y) -> self
decision_function(X) -> 一维决策分数
predict(X) -> 原始数值类别
score(X, y) -> 准确率
```

X 为有限实二维矩阵，特征数符合固定核；训练非空。y 为同长度一维有限实数向量，
恰好两种值；拒绝 bool、complex、字符串、object、NaN 和无穷大。
两个原始类别按升序对应负类/正类，零分归正类。没有伪造概率输出。
`score` 要求非空，标签必须属于已拟合类别，允许评价集中只出现其中一类。
显式 `(0,d)` 预测返回空数组且不执行量子核。

`num_steps` 为非布尔正整数。seed 为非负整数或 None，每次 fit 重建局部 RNG；
固定 seed 可复现，不修改全局 NumPy 随机状态。新结果全部准备好后才替换状态；
失败重拟合保留旧模型。极端参数导致非有限更新、范数或目标时明确报错。
浮点级负平方范数按矩阵和系数规模界定，显著负值不会静默夹成 0。

可读取的诊断包括 `classes_`、`n_features_in_`、`training_data_`、`coefficients_`、
`raw_kernel_matrix_`、`kernel_matrix_`、`minimum_raw_eigenvalue_`、
`psd_correction_norm_`、`weight_norm_`、`training_objective_`、`n_iter_`。
数组都返回副本，类别为不可变二元组；首次拟合前读取这些诊断会报错。

## 测试、成本和限制

`tests/algorithms/test_pegasos_qsvc.py` 用 RY 核的独立三维实特征直接更新欧氏 w，
然后与核系数训练的决策和范数比较，避免用第二套相同核算法自证。
另有首步必须触发的球投影、严格 margin=1、闭式本征分解的 PSD 正部、
原始矩形预测、真实精确核的留出分类、确定性频数的真实 shots 路径、标签还原、
seed 复现、错误输入、诊断副本和失败重拟合测试。

保存完整训练 Gram 需要 O(n²) 内存；当前每步范数计算为 O(n²)，加上核评估成本。
性能优先级低于规则正确性，不声称继承经典稀疏 Pegasos 的大数据规模优势。
此分类器不需要 SciPy/sklearn，但下层 Core/NumPy 与核执行器仍必需。
当前不支持多分类、sample weights、截距、warm start、partial_fit 或概率校准。
