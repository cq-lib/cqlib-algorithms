# 当前能力清单

本页只列出当前源码已经导出的能力。参数、返回值和异常以相应源码 docstring 为准。

## 公共协议

| 协议 | 用途 |
| --- | --- |
| `StatePreparation` | 构建固定态或数据编码线路 |
| `Ansatz` | 构建可训练参数化线路 |
| `Sampler` | 批量执行有限-shot 测量采样 |
| `Estimator` | 批量估计可观测量期望值 |
| `StateFidelity` | 批量估计量子态保真度 |
| `Gradient` | 批量计算 Estimator 目标的梯度 |
| `Optimizer` | 批量优化经典参数 |
| `QuantumKernel`、`TrainableQuantumKernel` | 固定核求值与可训练核参数绑定 |
| `KernelLoss` | 从核矩阵和目标向量计算标量损失 |
| `ScalarMinimizer`、`ScalarMinimizerResult` | 经典标量目标搜索及结果 |

## 组件

| 类别 | 已实现能力 |
| --- | --- |
| 状态制备 | 基态、均匀叠加、任意纯态、乘积态、GHZ、Dicke、稠密/稀疏幅度编码、角度/基编码、图像角度编码、BRQI、用户线路、Feature Map、Hartree–Fock、Slater、费米 Gaussian、晶格参考态与 VSCF |
| Ansatz | TwoLocal、RealAmplitudes、EfficientSU2、硬件高效、双量子比特参数化酉、PauliTwoDesign、算符演化、Hamiltonian 变分、激发保持、SimplifiedTwoDesign、随机层、MPS、TTN、MERA、UCC/UCCSD、PUCCD/PUCCSD、UVCC/UVCCSD、晶格守恒拟设、QAOA 成本/混合演化 |
| Mixer | nearby-values、ring、fully-connected、null-swap、受控交换、排列交换、时间调度和有序组合 |
| Oracle 与反射 | 布尔函数、位翻转/相位 Oracle、零态/一般态反射、Grover diffusion、Long 相位反射、任意酉矩阵综合、酉扩张块编码和 Dicke 权重分布组件 |
| 线路组合 | Hadamard 相位查询、隐藏线性函数电路、Grover 迭代单元、相位估计电路、QSP 与 QSVT |
| 测量 | Pauli none/QWC 分组、全局对易分组、局域基测量规划 |
| 执行 | 本地态矢量 Sampler、回调式后端 Sampler、精确态矢量 Estimator、采样 Estimator |
| 微分与优化 | 参数移位、线性组合、有限差分、SPSA 梯度，QGT、QFI、自然梯度，GradientDescent、Adam、SPSA、COBYLA、L-BFGS-B 优化器 |
| 量子信息 | 纯态或混态任意量子比特二分下的精确量子互信息 |
| 保真度与重叠 | `U2† U1` 酉重叠线路、基于 compute–uncompute 的线路纯态保真度 |
| 误差缓解 | CDR、DDD、DEB、LRE、PEA、PEC、Pauli twirling、QSE、读出缓解、classical shadows、T-REx、Virtual Distillation 和 ZNE |
| 量子核 | `FidelityQuantumKernel`、`TrainableFidelityQuantumKernel`、`BalancedKernelAlignmentLoss` |
| 线路变换 | 可配置近似阶和输出 SWAP 的 `QFT` |

## 科研计算链

| 方向 | 已实现能力 |
| --- | --- |
| 分子电子结构 | RHF/ROHF/UHF 数据、限制性与自旋分辨活性空间、费米 Hamiltonian、Jordan–Wigner 映射、HF/Slater/Gaussian 参考态、UCC 系列、VQE 基态、中性 qEOM、粒子数与自旋诊断 |
| 晶格与自旋 | 通用/线形/矩形/超立方晶格、Fermi–Hubbard、Ising、Heisenberg、守恒扇区、物理参考态与拟设、VQE 基态、占据/磁化/关联观测量 |
| 振动系统 | 有限 modal 积分、稀疏振动算符、one-hot 映射、VSCF 参考、UVCC 系列、VQE 基态、modal 占据与严格 one-hot 泄漏诊断 |

量子算术子包已经存在，但不属于本轮文档与 docstring 整理范围；其接口保持现状。

## 完整算法

| 类别 | 已实现算法 |
| --- | --- |
| 相位估计 | 标准 QPE、迭代 QPE、Hamiltonian 相位估计 |
| 振幅估计 | 标准 AE、迭代 AE、最大似然 AE、快速 AE |
| 搜索 | Grover 搜索 |
| 变分算法 | VQE、VQD、ADAPT-VQE、Sampling VQE、QAOA、PVQD 实时演化、VarQITE 虚时演化、VarQRTE 实时演化、电子结构基态与 qEOM、晶格基态、振动基态 |
| Oracle 算法 | Deutsch–Jozsa、Bernstein–Vazirani、Simon |
| 量子机器学习 | QBayesian、QSVR、VQR、PegasosQSVC、QuantumKernelTrainer |
| 量子信息与协议 | CHSH、BB84 协议实验、量子隐形传态、超密编码 |

未在本页列出的规划项不视为当前能力。当前实现仍可能受模拟规模、显式真值表或稠密
矩阵构造限制，具体边界见对应用户指南和源码 docstring。
