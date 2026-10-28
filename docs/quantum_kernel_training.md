# QuantumKernelTrainer：训练量子“相似度尺”

## 为什么要训练核

量子核给出两个输入经线路编码后的相似程度。固定的线路不一定适合当前数据：
本该区分的样本可能被看成很相似。核训练利用已知类别调整线路，让相似度表更符合
分类任务。它训练的是一把尺，输出的仍是核；真正的回归或分类交给 QSVR、PegasosQSVC。
输出类型 `FidelityQuantumKernel` 是 `operators/quantum_kernel/` 中的共享算子，
可从 `cqlib_algorithms.operators` 导入。

采用 Hubregtsen 等
[arXiv:2105.02276v1](https://arxiv.org/abs/2105.02276v1) §IV Eq.(27)
及不平衡类别缩放的核目标对齐：不中心化，按类别样本数平衡。
这是选择特征的监督代理目标，不保证测试准确率一定提高，也不保证找到全局最佳核。
没有调用其他框架的核训练算法，不实现 Qiskit 的接口或其默认 SVC 损失。

## 先跑一个小例子

`RY(w*x)` 中 w 控制不同输入被转开多少。只有两个训练点时，初始 w 很小会让它们
几乎相同；训练后可接近正交。小例子用于验证组合是否正确，不用于展示速度优势。

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

类别平衡总是启用，均衡数据时与不缩放标签的对齐相同。不对 K 或 T 做中心化，
不作 PSD 裁剪，不把分类器的损失混进来。即使有限-shot 原始核不定，也直接计算
这个经验对齐值；它可能为负。对齐范围是 [-1,1]，只允许机器精度量级的越界。

两点不同类的例子里，若交叉相似度为 k，则对齐为
`(1-k)/sqrt(2*(1+k*k))`。因此 k=0 时是 `1/sqrt(2)`，并不是 1。
不能因为它没有达到 1 就误判训练失败。

## 实现如何连接现有组件

构造时保存完整符号线路、显式数据参数、显式训练参数及固定参数的快照。
每个候选从这个符号模板重新构造公开 `FidelityQuantumKernel`，把候选权重放进
固定绑定后调用 `evaluate(X)`。因此复用已有核的精确/采样执行、去重、数值绑定验证、
静态酉约束及态矢量资源限制，不读取旧核的私有执行字段，不修改旧核或增加 setter。

经典 COBYLA 只接收完整的标量负对齐目标。初始与最终诊断各实际计算一次；最终
`kernel_matrix_`、`alignment_`、`loss_` 来自同一次评估，采样模式下也如此。
不为报告再额外采样一遍，不把相同参数的旧 shot 结果假装成新执行。

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
构造时不任意选 0 点验证；`1/x` 或 `1/w` 可以在调用方合法域内使用。
实际候选出现除零、溢出或无效线路时明确报错，不自动改成惩罚数或随机重启。

X 为有限实二维矩阵、非空、列数匹配；y 为同长度一维有限实数向量，恰好两种
不同数值。拒绝 bool、complex、字符串、object、NaN、无穷大和非矩形输入。
原线路可以有乱序/不连续量子位 ID，按插入位置归一化。仅支持静态酉操作及 barrier。

不传 sampler/shots 时走精确核；传入公共 Sampler 和正 shots 时走有限采样，两者
必须一起提供。`max_statevector_qubits/bytes` 继承固定核的执行边界：byte 限制
针对保留的态矢量幅度，不是整个进程内存上限，输出矩阵和模拟器临时存储另计。

## 结果、停止及错误

`quantum_kernel_` 可直接交给 QSVR 或 PegasosQSVC；Trainer 自身没有 predict，
因为它没有训练最终分类器。`optimal_parameters_` 为不可变映射，`kernel_matrix_`
返回副本，`classes_` 为原始类别二元组；`n_features_in_` 是训练的特征数。
`initial_alignment_`、`alignment_`、`loss_` 反映实际初始和最终目标。

`max_evaluations` 是优化器的目标调用预算，至少为训练参数数目加 2；
`tolerance`、`initial_step` 是有限正数。预算外额外的初始、最终诊断各一次，
`objective_evaluations_ = optimizer_evaluations_ + 2`。
每次目标都要算完整核矩阵，成本通常含 O(n²) 个样本对，不能把评估数当作 shots。

`optimizer_success_/status_/message_` 保留真实停止信息。预算结束但返回有限可验证
点时仍可以提供训练核，不宣称已收敛或必定改善。合法平坦目标也允许正常返回，
例如给所有输入共同追加同一个酉门可能完全不改保真度。
非法输入、坏核/优化器输出或最终复核失败时，不留下半个新模型；失败重拟合
保留旧训练核。拟合前读取训练诊断会报错。

## 如何验证和当前限制

`tests/algorithms/test_quantum_kernel_trainer.py` 用解析
`cos²(w*(x_i-x_j)/2)` 独立生成核与平衡对齐，覆盖不平衡类别和两点闭式公式。
真实 COBYLA 训练验证对齐提升，输出核实际组合已有 QSVR 和新 PegasosQSVC。
测试还检查非字典序多参数、固定绑定、输入快照、每个候选重新绑定、平坦目标、
确定性 shot 频数下的不定 raw Gram 与最终结果一致性，以及奇点、资源、坏结果、
缺 SciPy、预算、失败重拟合等边界。

SciPy 是 fit 时延迟导入的可选经典依赖，安装 `cqlib-algorithms[machine-learning]`
可获得相关依赖。当前不支持任意 loss、一般回归对齐、多分类、梯度、内层 SVC、
超参数搜索、自动划分数据或全局最优保证。实际数据效果需要单独的测试集评价。
