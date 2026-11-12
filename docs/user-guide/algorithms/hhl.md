# HHL 线性方程组求解

`HHL` 求解 Hermitian 正定线性系统 <img src="../../assets/formulas/inline/inline-4f218032237abba5.svg" alt="Ax=b" style="vertical-align: -1.000pt;">，制备与 <img src="../../assets/formulas/inline/inline-65ac1652270f246b.svg" alt="A^{-1}|b\rangle" style="vertical-align: -4.000pt;"> 成比例的量子态。
它不会返回经典向量 `x`：执行结果是辅助量子比特测得 `1` 后，系统寄存器在计算基上的
条件概率分布。需要完整复振幅或逐项经典解时，应使用经典线性代数方法或在量子态上
继续估计目标可观测量。

## 算法结构

对谱上界 `alpha`，算法构造
<img src="../../assets/formulas/inline/inline-35a5c8f985c00d98.svg" alt="U=\exp(2\pi iA/\alpha)" style="vertical-align: -4.000pt;">，依次执行：

1. 在系统寄存器制备归一化右端项 <img src="../../assets/formulas/inline/inline-443db1954f6815ba.svg" alt="|b\rangle" style="vertical-align: -4.000pt;">；
2. 对 <img src="../../assets/formulas/inline/inline-a25513c7e0f6eaa8.svg" alt="U" style="vertical-align: -1.000pt;"> 进行相位估计，将 <img src="../../assets/formulas/inline/inline-c52635a76801021f.svg" alt="\lambda_j/\alpha" style="vertical-align: -5.000pt;"> 写入评估寄存器；
3. 使用 `ExactReciprocal` 对特征值执行受控倒数旋转；
4. 逆向执行相位估计，复原评估寄存器；
5. 测量系统寄存器和辅助量子比特，并在辅助位为 `1` 的样本上归一化系统分布。

默认组件是 `AmplitudeEncoding`、`PhaseEstimationCircuit` 和 `ExactReciprocal`。
`HHLComponents` 允许注入这些组件的兼容子类；`HHL.build_circuit()` 只构建未测量线路，
`HHL.run()` 通过注入的 `Sampler` 批量执行请求。

## 使用示例

```python
import numpy as np

from cqlib_algorithms import HHL, HHLRequest, LinearSystemProblem
from cqlib_algorithms.operators import StatevectorSampler

operator = np.diag(np.array([1.0, 0.5], dtype=np.complex128))
problem = LinearSystemProblem(operator=operator, rhs=[1.0, 1.0])
request = HHLRequest(
    problem=problem,
    num_evaluation_qubits=4,
    scaling=2.0**-4,
    shots=8192,
    bound=2.0,
    seed=7,
)

result = HHL(StatevectorSampler()).run([request])[0]
print(result.success_probability)
print(dict(result.solution_probabilities))
```

上例的归一化经典解满足 <img src="../../assets/formulas/inline/inline-2c5af535b8de42e4.svg" alt="A^{-1}b\propto(1,2)" style="vertical-align: -4.000pt;">，因此理想计算基概率为
`{0: 0.2, 1: 0.8}`；有限 shots 的结果会在该分布附近波动。

## 输入与结果

| 接口 | 作用 |
| --- | --- |
| `LinearSystemProblem(operator, rhs)` | 保存 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 与 <img src="../../assets/formulas/inline/inline-3e23e8160039594a.svg" alt="b" style="vertical-align: -1.000pt;"> 的独立快照；`operator` 接受 `numpy.ndarray` 或 `Hamiltonian`，`rhs` 接受复向量或静态酉制备线路 |
| `HHLRequest` | 指定评估寄存器宽度、倒数缩放、shots、QFT 近似度、谱上界和随机种子 |
| `HHLComponents` | 注册状态制备、相位估计和倒数旋转的组件类 |
| `HHLResult` | 返回后选择成功率、条件系统分布、实际谱上界、缩放参数和寄存器宽度 |

`solution_probabilities[i]` 对应系统寄存器计算基态 <img src="../../assets/formulas/inline/inline-174310e98fddcbf5.svg" alt="|i\rangle" style="vertical-align: -4.000pt;">，量子比特 0 是最低有效位。
空分布表示本次采样没有出现辅助位为 `1` 的结果，不表示线性系统无解。
`num_unitary_applications` 给出正向和逆向相位估计合计的受控酉查询数
<img src="../../assets/formulas/inline/inline-e74647334e018e23.svg" alt="2(2^m-1)" style="vertical-align: -4.000pt;">。

## 精度、资源与适用边界

- <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 必须是有限、非奇异、维度为 <img src="../../assets/formulas/inline/inline-e8a3fed33f2f1fe0.svg" alt="2^n" style="vertical-align: -0.000pt;"> 的 Hermitian 正定算符；当前实现不自动把
  一般矩阵嵌入 Hermitian 系统。
- 评估寄存器宽度 `m` 决定特征值网格分辨率。特征值被量化到零相位网格点时，倒数
  旋转会丢弃该分支；实现会对过于接近周期端点的谱发出警告。
- `scaling` 必须满足 <img src="../../assets/formulas/inline/inline-50450a1d7a1ad898.svg" alt="0&lt;\mathtt{scaling}\leq2^{-m}" style="vertical-align: -4.000pt;">。缩放越小，辅助位后选择成功率通常越低，
  需要更多 shots 才能稳定估计条件分布。
- 稠密矩阵入口会显式对角化 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 并构造 <img src="../../assets/formulas/inline/inline-cfa68802318c3913.svg" alt="2^n\times2^n" style="vertical-align: -0.000pt;"> 酉矩阵，只适合小规模验证。
  `Hamiltonian` 入口使用二阶乘积公式构造演化，结果还包含 Hamiltonian 演化近似误差。
- 输出只有计算基概率，不包含解态相位，也不提供条件数估计、误差界或经典向量重构。

## 参考文献

- A. W. Harrow, A. Hassidim, and S. Lloyd, “Quantum algorithm for linear
  systems of equations,” *Physical Review Letters* 103, 150502 (2009),
  DOI: 10.1103/PhysRevLett.103.150502.

