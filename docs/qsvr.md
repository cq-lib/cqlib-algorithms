# 保真度量子核与 QSVR

`FidelityQuantumKernel` 计算固定量子特征线路的纯态保真度核；`QSVR` 基于该核训练经典
单输出 ε-SVR。核组件位于 `operators/quantum_kernel/fidelity_quantum_kernel.py`，
从 `cqlib_algorithms.operators` 导入；QSVR 位于 `algorithms/qsvr.py`，
从 `cqlib_algorithms.algorithms` 导入。原算法层的核导出保留为兼容入口。

## 数学定义与来源

特征线路与核定义为

\[
|\phi(x)\rangle=U(x)|0\cdots0\rangle,\qquad
K(x,y)=|\langle\phi(x)|\phi(y)\rangle|^2
       =\operatorname{Tr}(\rho_x\rho_y),
\quad \rho_x=|\phi(x)\rangle\langle\phi(x)|.
\]

密度算符的 Hilbert–Schmidt 内积说明理想精确核是半正定的。量子特征空间与重叠核构造
参考 [Havlíček 等，arXiv:1804.11326v2](https://arxiv.org/abs/1804.11326v2)。
该量子核构造在本实现中应用于回归任务。

ε-SVR 训练目标采用

\[
\min_{w,b}\frac12\|w\|^2+
C\sum_i s_i\max\{0,|y_i-(\langle w,\Phi(x_i)\rangle+b)|-\epsilon\},
\]

其中 `s_i=1`，或由正的 `sample_weight[i]` 给出。核空间预测为

\[
f(x)=\sum_i(\alpha_i-\alpha_i^*)K(x_i,x)+b,
\quad \sum_i(\alpha_i-\alpha_i^*)=0,
\quad 0\le\alpha_i,\alpha_i^*\le C s_i.
\]

目标与对偶形式来源于 [Smola 与 Schölkopf，A tutorial on support vector regression，2004](https://alex.smola.org/papers/2004/SmoSch04.pdf)
（[DOI](https://doi.org/10.1023/B:STCO.0000035301.49549.88)）。
经典求解使用 [scikit-learn `SVR(kernel="precomputed")`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)。

## FidelityQuantumKernel 接口

```text
FidelityQuantumKernel(feature_map, input_parameters, *, parameter_bindings=None,
                      sampler=None, shots=None, max_statevector_qubits=20,
                      max_statevector_bytes=268_435_456)
evaluate(X, Y=None) -> numpy.ndarray
```

`feature_map` 是 Cqlib `Circuit`。`input_parameters` 是非空、唯一、有序的活跃符号名称序列，
第 `j` 列数据绑定第 `j` 个名称，顺序由调用方明确给出。`parameter_bindings` 给出其他符号
的固定有限实数绑定，必须与输入符号不相交，两者恰好覆盖全部活跃符号。
线路可以重复使用同一输入符号，固定参数由 `parameter_bindings` 设置。

构造时复用 `UserCircuitPreparation` 复制线路，并按量子比特插入位置归一化为连续位置。
两种执行方式都要求绑定后线路静态酉、可逆。构造时绑定固定参数并检查线路结构；
求值时按实际数据行绑定输入参数并校验数值。例如 `RY(1/x)` 可以用于非零输入。
所有不同数据行完成校验后，再执行保真度计算和相同行快捷处理。
`feature_map` 属性返回独立副本，`parameter_bindings` 是不可变映射，
`input_parameters` 是元组，`num_features` 给出输入列数。

| 配置 | 执行方式 |
| --- | --- |
| `sampler=None, shots=None` | 默认精确本地路径：Core `Statevector.from_circuit` 与 `state_fidelity_pure` |
| 同时提供 `Sampler` 和正整数 `shots` | 通过现有 `ComputeUncomputeFidelity` 批量估计全零概率 |

`sampler` 和 `shots` 成对提供。精确路径在一次调用中复用不同数据行的态矢量；
采样路径对每个执行配对使用指定的 `shots`。
两种方式都将**完全相同的数据行**的核值直接设为 1，包括对角项；有噪声的 sampler
也采用这个归一化约定。不同数据行按所选执行方式求值。

`X` 和 `Y` 是有限实数二维数组 `(n_samples, num_features)`，普通整数也可作为输入。
输入数值直接绑定线路参数。
省略 `Y` 时仅计算上三角并镜像，返回精确对称的 Gram 矩阵；提供 `Y` 时返回
`(len(X), len(Y))` 的交叉核，保留行顺序。
形状为 `(0,d)` 的输入返回相应空矩阵。每次调用返回独立的 float64 数组，
态矢量复用范围为本次求值。

精确路径分配态矢量前检查 qubit 上限，以及

\[
16\times2^{num\_qubits}\times number\_of\_retained\_distinct\_states
\le max\_statevector\_bytes.
\]

默认最多 20 qubits、256 MiB 的保留 complex128 幅度数据；两个输入集合的不同数据行共同计数。
该预算统计保留的态矢量幅度，Core 临时缓冲、线路和输出矩阵的存储另计。
输出矩阵尺寸在分配前检查；分配失败转换为 `ValueError`。两个上限为正整数，可按需设置。
精确核仅夹取不超过 `1e-12` 的 `[0,1]` 浮点边界越界；更大偏差或非有限 Core 输出报错。
核组件返回原始保真度矩阵，训练矩阵处理由 QSVR 完成。

## QSVR 接口与模型状态

```text
QSVR(quantum_kernel, *, C=1.0, epsilon=0.1, tol=1e-3,
     max_iter=-1, enforce_psd=True)
fit(X, y, *, sample_weight=None) -> QSVR
predict(X) -> numpy.ndarray
score(X, y, *, sample_weight=None) -> float
```

`quantum_kernel` 为 `FidelityQuantumKernel`。`C`、`tol` 为有限正实数，
`epsilon` 为有限非负实数。`max_iter` 为 -1 或正整数，
`enforce_psd` 为 bool。`fit` 要求非空二维 `X` 和匹配长度的一维有限实数 `y`，
目标为单个数值；权重若提供，须为匹配长度的一维有限严格正数。
输入校验在量子核求值前完成。

默认将训练核 `K_raw` 的特征值裁剪至非负：

\[
K_{raw}=V\operatorname{diag}(\lambda)V^T,\qquad
K_{fit}=V\operatorname{diag}(\max(\lambda,0))V^T.
\]

重构并对称化后得到用于求解的 `K_fit`，同时保留原始矩阵 `K_raw`。
`enforce_psd=False` 时使用 `K_raw` 训练。
预测时计算交叉核 `K(X_new, X_train)`，结合训练得到的系数及截距计算预测值。

`fit` 返回自身，训练完成后更新模型；重拟合失败时保留旧模型。
求解状态由 `fit_status_` 记录，触及迭代上限时同时保留求解器的 convergence warning。

`predict` 在拟合前抛出 `ValueError`；输入 `(0,d)` 返回空 `(0,)`，无量子或求解器调用。
`score` 是单输出 `R² = 1 - weighted_MSE / weighted_variance`，至少需要两个目标值，
可能为负；常量目标完美预测为 1，否则为 0。

| 只读诊断属性 | 含义 |
| --- | --- |
| `is_fitted`, `n_features_in_` | 是否已拟合、训练输入列数 |
| `training_data_` | 训练量子输入的副本，保持样本顺序 |
| `raw_kernel_matrix_`, `kernel_matrix_` | 原始训练核与求解器使用的训练核副本 |
| `support_` | 支持样本在 `training_data_` 中的索引副本 |
| `dual_coef_`, `intercept_` | 形状 `(1,n_support)` 的带符号系数与长度 1 的截距副本 |
| `fit_status_`, `n_iter_` | 求解状态（0 表示正常结束）及迭代数 |
| `psd_correction_norm_` | `||K_fit-K_raw||_F` |
| `minimum_raw_eigenvalue_` | 原始对称训练核的最小特征值 |

除 `is_fitted` 外，这些属性均在拟合后读取，数组属性返回独立副本。

scikit-learn 在 `fit` 时加载。通过 `cqlib-algorithms[machine-learning]` 安装经典计算依赖，
其中 scikit-learn 版本范围为 `>=1.4,<2`。

## 使用示例：精确核与回归

一量子比特 `RY(x)` 的独立解析核为 `cos((x-y)/2)^2`。
以下数据选自 `sin(x)`，展示训练、预测和解析核校验。

```python
import numpy as np
from cqlib import Circuit, Parameter

from cqlib_algorithms.algorithms import QSVR
from cqlib_algorithms.operators import FidelityQuantumKernel

feature_map = Circuit(1)
feature_map.ry(0, Parameter("angle"))
kernel = FidelityQuantumKernel(feature_map, ["angle"])
X_train = np.linspace(-1.5, 1.5, 7).reshape(-1, 1)
y_train = np.sin(X_train[:, 0])
model = QSVR(kernel, C=10.0, epsilon=0.01, tol=1e-8).fit(X_train, y_train)
X_test = np.array([[-0.75], [0.25], [1.0]])
print("Predictions:", np.round(model.predict(X_test), 6))
print("Training R2:", round(model.score(X_train, y_train), 6))
print("Fit status / iterations:", model.fit_status_, model.n_iter_)
print("PSD correction norm:", model.psd_correction_norm_)
analytic = np.cos((X_train[:, None, 0] - X_train[None, :, 0]) / 2) ** 2
np.testing.assert_allclose(kernel.evaluate(X_train), analytic, rtol=0, atol=1e-12)
assert model.is_fitted and model.fit_status_ == 0
```

## 使用示例：有限采样

示例输入包含三行数据，其中一行重复；实际执行两个数据配对，各使用 64 shots，总计 128 shots。
两个配对均对应正交态，结果确定。相同配置的量子核可作为 `QSVR` 的输入，
训练、预测和评分按需执行配对采样。

```python
import math

import numpy as np
from cqlib import Circuit, Parameter

from cqlib_algorithms.algorithms import QSVR
from cqlib_algorithms.operators import FidelityQuantumKernel
from cqlib_algorithms.operators.sampler import StatevectorSampler

feature_map = Circuit(1)
feature_map.ry(0, Parameter("angle"))
kernel = FidelityQuantumKernel(
    feature_map, ["angle"], sampler=StatevectorSampler(), shots=64
)
matrix = kernel.evaluate([[0.0], [math.pi], [0.0]])
print("Sampled Gram matrix:")
print(matrix)
np.testing.assert_array_equal(matrix, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
model = QSVR(kernel, C=1.0, epsilon=0.1).fit([[0.0], [math.pi]], [0.0, 1.0])
print("Sampled model predictions:", model.predict([[0.0], [math.pi]]))
assert model.fit_status_ == 0
```

精确态矢量存储为 O(2^q)，训练 Gram 矩阵存储为 O(n²)，其中 q 为量子比特数，n 为训练样本数。
