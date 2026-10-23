# 当前能力清单

本页只列出当前源码已经导出的能力。参数、返回值和异常以相应源码 docstring 为准。

## 公共协议

| 协议 | 用途 |
| --- | --- |
| `StatePreparation` | 构建固定态或数据编码线路 |
| `Ansatz` | 构建可训练参数化线路 |
| `Sampler` | 批量执行有限-shot 测量采样 |
| `Estimator` | 批量估计可观测量期望值 |
| `Gradient` | 批量计算 Estimator 目标的梯度 |
| `Optimizer` | 批量优化经典参数 |

## 组件

| 类别 | 已实现能力 |
| --- | --- |
| 状态制备 | 基态、均匀叠加、任意纯态、乘积态、GHZ、稠密/稀疏幅度编码、角度/基编码、图像角度编码、BRQI、用户线路与 Feature Map 适配 |
| Ansatz | TwoLocal、RealAmplitudes、EfficientSU2、硬件高效、双量子比特参数化酉、PauliTwoDesign、算符演化、Hamiltonian 变分、激发保持、SimplifiedTwoDesign、随机层、MPS、TTN、MERA、QAOA 成本/混合演化 |
| Mixer | nearby-values、ring、fully-connected、null-swap、受控交换、排列交换、时间调度和有序组合 |
| Oracle 与反射 | 布尔函数、位翻转/相位 Oracle、零态/一般态反射、Grover diffusion 与 Long 相位反射 |
| 线路组合 | Hadamard 相位查询、隐藏线性函数电路、Grover 迭代单元、相位估计电路 |
| 测量 | Pauli none/QWC 分组、全局对易分组、局域基测量规划 |
| 执行 | 本地态矢量 Sampler、回调式后端 Sampler、精确态矢量 Estimator、采样 Estimator |
| 微分与优化 | 参数移位、线性组合、有限差分、SPSA 梯度，QGT、QFI、自然梯度，GradientDescent、Adam、SPSA、COBYLA、L-BFGS-B 优化器 |
| 量子信息 | 纯态或混态任意量子比特二分下的精确量子互信息 |
| 保真度与重叠 | `U2† U1` 酉重叠线路、基于 compute–uncompute 的线路纯态保真度 |
| 线路变换 | 可配置近似阶和输出 SWAP 的 `QFT` |

量子算术子包已经存在，但不属于本轮文档与 docstring 整理范围；其接口保持现状。

## 完整算法

| 类别 | 已实现算法 |
| --- | --- |
| 相位估计 | 标准 QPE、迭代 QPE、Hamiltonian 相位估计 |
| 振幅估计 | 标准 AE、迭代 AE、最大似然 AE、快速 AE |
| 变分算法 | VQE、QAOA |
| Oracle 算法 | Deutsch–Jozsa、Bernstein–Vazirani、Simon |
| 量子信息与协议 | CHSH、BB84 协议实验、量子隐形传态、超密编码 |

未在本页列出的规划项不视为当前能力。当前实现仍可能受模拟规模、显式真值表或稠密
矩阵构造限制，具体边界见对应用户指南和源码 docstring。
