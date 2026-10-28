# QuantumKernelTrainer：训练量子“相似度尺”

## 为什么要训练核

量子核给出两个输入经线路编码后的相似程度。固定的线路不一定适合当前数据：
本该区分的样本可能被看成很相似。核训练利用已知类别调整线路，让相似度表更符合
分类任务。它训练的是一把尺，输出的仍是核；真正的回归或分类交给 QSVR、PegasosQSVC。
输出类型 `FidelityQuantumKernel` 是 `operators/quantum_kernel/` 中的共享算子，
可从 `cqlib_algorithms.operators` 导入。

采用 Hubregtsen 等
[arXiv:2105.02276v1](https://arxiv.org/abs/2105.02276v1) §IV Eq.(27)
及不平衡类别缩放的核目标对齐，在原始核矩阵上按类别样本数平衡目标。

## 先跑一个小例子

`RY(w*x)` 中 w 控制不同输入被转开多少。只有两个训练点时，初始 w 很小会让它们
几乎相同；训练后可接近正交。示例展示核参数训练，以及与分类器、回归器的组合。

```python
import numpy as np
from cqlib import Circuit, Parameter
from cqlib_algorithms.algorithms import QuantumKernelTrainer, PegasosQSVC, QSVR

circuit = Circuit(1)
circuit.ry(0, Parameter("w") * Parameter("x"))
trainer = QuantumKernelTrainer(circuit, ["x"], ["w"], max_evaluations=120)
trainer.fit([[0.0], [1.0]], [-1, 1], initial_point={"w": 0.2})
print("Initial alignment:", trainer.initial_alignment_)
print("Final alignment:", trainer.alignment_)
print("Learned parameters:", dict(trainer.optimal_parameters_))
assert trainer.alignment_ > 0.70

kernel = trainer.quantum_kernel_
classifier = PegasosQSVC(kernel, num_steps=100, seed=4)
classifier.fit([[0.0], [1.0]], [-1, 1])
assert classifier.score([[0.0], [1.0]], [-1, 1]) == 1.0
regressor = QSVR(kernel, C=10, epsilon=0)
regressor.fit([[0.0], [1.0]], [0.0, 1.0])
np.testing.assert_allclose(regressor.predict([[0.0], [1.0]]), [0.0, 1.0], atol=1e-5)
```

## 到底优化了什么

先将两类按数值升序映射为 `z_i=-1,+1`，再令
`t_i=z_i/n_class(i)`，其中分母是该样本所在类别的训练样本数。
目标矩阵为 `T=t*t.T`。候选线路给出原始 Gram `K_theta` 后：

`alignment = sum(K_theta*T)/(||K_theta||_F * sum(t**2))`；
`loss = -alignment`。

类别平衡总是启用，均衡数据时与直接使用 ±1 标签的对齐相同。
K 和 T 按上述原始矩阵计算 Frobenius 内积与范数，采样模式使用本次估计的 K。
对齐范围为 [-1,1]，计算时处理机器精度量级的越界。

两点不同类的例子里，若交叉相似度为 k，则对齐为
`(1-k)/sqrt(2*(1+k*k))`，k=0 时对齐值为 `1/sqrt(2)`。

## 实现如何连接现有组件

构造时保存完整符号线路、显式数据参数、显式训练参数及固定参数的快照。
每个候选从这个符号模板重新构造公开 `FidelityQuantumKernel`，把候选权重放进
固定绑定后调用 `evaluate(X)`。因此复用已有核的精确/采样执行、去重、数值绑定验证、
静态酉校验及态矢量资源管理。

经典 COBYLA 只接收完整的标量负对齐目标。初始与最终诊断各实际计算一次；最终
`kernel_matrix_`、`alignment_`、`loss_` 来自同一次评估，采样模式下也如此。

## 接口与数据

```text
QuantumKernelTrainer(feature_map, input_parameters, trainable_parameters,
    parameter_bindings=None, sampler=None, shots=None,
    max_statevector_qubits=20, max_statevector_bytes=268435456,
    max_evaluations=200, tolerance=1e-6, initial_step=1.0)
fit(X, y, initial_point=...) -> self
```

数据参数、训练参数均为非空、有序、唯一名称序列；两者及固定绑定互斥并恰好覆盖
全部活跃符号。数据列与训练向量按显式顺序对应，初值必须精确覆盖所有训练参数。
参数表达式按实际样本和候选权重校验，`1/x` 或 `1/w` 可用于各自的有效定义域。
除零、溢出或无效线路通过异常返回。

X 为有限实二维矩阵、非空、列数匹配；y 为同长度一维有限实数向量，恰好两种不同数值。
原线路可使用乱序或不连续量子位 ID，按插入位置归一化，包含静态酉操作及可选 barrier。

默认执行精确核计算；同时提供公共 Sampler 和正整数 shots 时执行有限采样。
`max_statevector_qubits/bytes` 配置态矢量的量子比特数和保留幅度存储预算，
输出矩阵和模拟器临时存储另计。

## 结果、停止及错误

`quantum_kernel_` 可直接交给 QSVR 或 PegasosQSVC 进行训练和预测。
`optimal_parameters_` 为不可变映射，`kernel_matrix_`
返回副本，`classes_` 为原始类别二元组；`n_features_in_` 是训练的特征数。
`initial_alignment_`、`alignment_`、`loss_` 反映实际初始和最终目标。

`max_evaluations` 是优化器的目标调用预算，至少为训练参数数目加 2；
`tolerance`、`initial_step` 是有限正数。预算外额外的初始、最终诊断各一次，
`objective_evaluations_ = optimizer_evaluations_ + 2`。
每次目标评估计算完整核矩阵，通常涉及 O(n²) 个样本对；采样次数由每对的 shots 决定。

`optimizer_success_/status_/message_` 记录停止信息，最终有效参数用于构造输出核。
平坦目标同样可返回训练结果，例如所有输入共同追加一个酉变换时保真度保持不变。
训练完成后统一更新状态，重拟合失败时保留旧训练核。训练诊断在拟合后读取。

## 测试与依赖

`tests/algorithms/test_quantum_kernel_trainer.py` 用解析
`cos²(w*(x_i-x_j)/2)` 独立生成核与平衡对齐，覆盖不平衡类别和两点闭式公式。
真实 COBYLA 训练验证对齐提升，输出核实际组合已有 QSVR 和新 PegasosQSVC。
测试还检查非字典序多参数、固定绑定、输入快照、每个候选重新绑定、平坦目标、
确定性 shot 频数下的不定 raw Gram 与最终结果一致性，以及奇点、资源、异常返回、
缺 SciPy、预算、失败重拟合等边界。

SciPy 在 fit 时加载，通过 `cqlib-algorithms[machine-learning]` 安装。
