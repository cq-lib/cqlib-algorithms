# cqlib-algorithms

面向 Cqlib 的量子算法组件库。支持 Python 3.10+，使用 uv 管理依赖、pytest 测试，
采用 Apache-2.0 开源协议。当前提供第一批公共组件协议、请求与结果数据结构，以及
首批状态制备、数据编码组件、变分拟设、QAOA 演化层、布尔 Oracle、反射算子、
Pauli 测量规划、同步采样器、期望值估计器、梯度、量子几何、经典优化组件、
线路纯态保真度估计组件、Hadamard 相位查询、Grover 复合迭代单元，以及可复用的量子傅里叶变换
线路构造器，以及完整的 VQE、QAOA、Deutsch-Jozsa、Bernstein-Vazirani、Simon、
CHSH 不等式验证与 BB84 协议实验算法。

## 目录

```text
src/cqlib_algorithms/
├── __init__.py
├── transforms/
│   ├── __init__.py
│   └── qft.py
├── py.typed
├── protocols/
│   ├── state_preparation.py
│   ├── ansatz.py
│   ├── sampler.py
│   ├── estimator.py
│   ├── gradient.py
│   └── optimizer.py
├── operators/
│   ├── ansatz/
│   │   ├── base_ansatz.py
│   │   ├── two_local_ansatz.py
│   │   ├── real_amplitudes_ansatz.py
│   │   ├── efficient_su2_ansatz.py
│   │   ├── hardware_efficient_ansatz.py
│   │   ├── pauli_two_design_ansatz.py
│   │   ├── qaoa_cost_evolution.py
│   │   └── qaoa_mixer_evolution.py
│   ├── composite/
│   │   ├── __init__.py
│   │   ├── grover_operator.py
│   │   └── hadamard_phase_query.py
│   ├── estimator/
│   │   ├── statevector_estimator.py
│   │   └── sampling_estimator.py
│   ├── gradient/
│   │   ├── finite_difference.py
│   │   ├── lin_comb.py
│   │   ├── parameter_shift.py
│   │   ├── spsa.py
│   │   ├── qfi.py
│   │   ├── qgt.py
│   │   └── natural_gradient.py
│   ├── optimizer/
│   │   ├── gradient_descent.py
│   │   ├── adam.py
│   │   └── spsa.py
│   ├── measurement/
│   │   ├── pauli_grouping_strategy.py
│   │   ├── commuting_observable_grouping.py
│   │   └── pauli_measurement_planner.py
│   ├── oracle/
│   │   ├── boolean_function.py
│   │   ├── boolean_vector_function.py
│   │   ├── phase_oracle.py
│   │   ├── bit_flip_oracle.py
│   │   ├── zero_state_reflection.py
│   │   ├── general_state_reflection.py
│   │   └── grover_diffusion.py
│   ├── sampler/
│   │   ├── statevector_sampler.py
│   │   └── backend_sampler.py
│   ├── state_fidelity/
│   │   └── compute_uncompute.py
│   └── state_preparation/
│       ├── basis_state_initialization.py
│       ├── uniform_superposition.py
│       ├── arbitrary_state_preparation.py
│       ├── product_state_preparation.py
│       ├── ghz_state_preparation.py
│       ├── amplitude_encoding.py
│       ├── angle_encoding.py
│       ├── basis_encoding.py
│       ├── sparse_amplitude_encoding.py
│       ├── user_circuit_preparation.py
│       └── feature_map_preparation.py
└── algorithms/
    ├── bernstein_vazirani.py
    ├── bb84.py
    ├── chsh.py
    ├── deutsch_jozsa.py
    ├── qaoa.py
    ├── simon.py
    ├── vqa_result.py
    └── vqe.py
tests/protocols/
tests/operators/state_preparation/
tests/operators/ansatz/
tests/operators/composite/
tests/operators/estimator/
tests/operators/gradient/
tests/operators/optimizer/
tests/operators/measurement/
tests/operators/sampler/
docs/development.md
AGENTS.md
pyproject.toml
LICENSE.txt
```

公共协议及其请求、结果类型集中放在 `protocols/`；具体组件按实际功能增加模块或
子包；可复用的线路变换放在 `transforms/`，完整算法放在 `algorithms/`。
不预建没有实现的分类目录。

## 公共组件协议

- `StatePreparation`：构建状态制备或数据编码线路。
- `Ansatz`：构建参数化变分线路。
- `Sampler`：同步、批量的线路采样协议。
- `Estimator`：同步、批量的期望值估计协议。
- `Gradient`：基于 Estimator 期望值目标的同步、批量梯度协议。
- `Optimizer`：对 Estimator 期望值目标执行同步、批量经典优化的协议。

公共组件契约采用抽象基类，具体组件必须显式继承对应基类并实现其抽象方法。
`StatePreparation` 与 `Ansatz` 具有相同的抽象操作，但属于不同的名义类型层级，
分别表达状态制备参数和可训练参数语义。
请求对象只浅冻结字段；其中的 Cqlib 线路和可观测量仍然可变，执行实现必须在运行前
重新核验请求约束，调用方不得在请求生命周期内修改这些 Core 对象。
`protocols.sampler` 同时提供 `validate_sampler_result` 与
`validate_sampler_results`，供要求完整原始频数的算法统一核验公共 Sampler 返回值。

本库的 `Estimator` 是批量估计协议，与 `cqlib.error_mitigation.Estimator` 的
单请求回调类型处于不同命名空间；同一模块需要同时使用时，应为后者设置明确别名。

```python
from cqlib_algorithms import (
    Ansatz,
    Estimator,
    Gradient,
    Optimizer,
    Sampler,
    StatePreparation,
)
```

## 状态制备组件

具体实现从 `cqlib_algorithms.operators` 或
`cqlib_algorithms.operators.state_preparation` 导入：

- 固定态：`BasisStateInitialization`、`UniformSuperposition`、
  `ArbitraryStatePreparation`、`ProductStatePreparation`、
  `GHZStatePreparation`。
- 数据编码：`AmplitudeEncoding`、`AngleEncoding`、`BasisEncoding`、
  `SparseAmplitudeEncoding`。
- 适配器：`UserCircuitPreparation`、`FeatureMapPreparation`。

所有基态索引采用 Cqlib 的小端约定，即第 0 个量子比特对应最低有效位。任意态和
稠密幅度编码使用均匀受控旋转进行精确线路综合，不依赖稠密酉门。
`SparseAmplitudeEncoding` 仅保存显式支持集，并以 two-level Givens 线路进行精确
综合，不分配长度为 `2**num_qubits` 的稠密向量；这不代表 QRAM 或对数复杂度
数据加载。Angle、Basis 和 Feature Map 组件委托当前兼容基线内的 Cqlib Core 实现。

完整接口、数值约定、综合方法和已知 Core 边界见
[状态制备与数据编码组件](docs/state_preparation.md)。

## Ansatz 与 QAOA 演化层组件

具体实现从 `cqlib_algorithms.operators` 或
`cqlib_algorithms.operators.ansatz` 导入：

- 公共基础实现：`BaseAnsatz`。
- TwoLocal 系列：`TwoLocalAnsatz`、`RealAmplitudesAnsatz`、
  `EfficientSU2Ansatz`、`HardwareEfficientAnsatz`。
- 随机线路：`PauliTwoDesignAnsatz`。
- QAOA 单层构件：`QAOACostEvolution`、`QAOAMixerEvolution`。

`BaseAnsatz` 继承公共 `Ansatz` 抽象契约，集中执行 prefix、配置和 Core 线路校验，
并直接提供线路构建及计数的公共代理实现；普通组件只保留构造配置，特殊组件只覆盖
自身不同的行为。
TwoLocal 系列委托 Cqlib Core 的 `TwoLocal`，
并统一采用“旋转层、量子比特、旋转门位置”的参数顺序。Pauli two-design 在构造时
冻结随机 Pauli 轴；QAOA 组件分别表示
单个成本哈密顿量演化层和混合哈密顿量演化层，并对传入 Hamiltonian 做防御性复制。
UCCSD 与 k-UpCCGSD 因当前 Core 尚缺少必要的费米算子和映射能力而暂不提供。

完整接口、线路结构、参数约定、理论来源和 Core 能力边界见
[Ansatz 与 QAOA 演化层组件](docs/ansatz.md)及
[UCC 类 Ansatz 的 Core 能力缺口](docs/ansatz-core-gap-report.md)。

## Measurement 组件

`PauliGroupingStrategy` 提供稳定的 none/QWC 分组，
`CommutingObservableGrouping` 提供全局对易关系的代数分组，
`PauliMeasurementPlanner` 根据 QWC 分组生成未测量的局域基变换线路。规划结果保留
原始 Pauli 项位置、相位和量子比特插入顺序；终端测量由 Sampler 在私有线路副本上
添加，避免重复测量并保持两层契约一致。

接口、分组规则、端序和 Sampler 组合方式见
[Pauli 测量规划与分组](docs/measurement.md)。

## Sampler 组件

`StatevectorSampler` 使用 Cqlib 态矢量模拟器执行有限-shot 本地采样；
`BackendSampler` 通过外部同步执行回调连接具体后端。两者均实现批量 `Sampler`
契约，在私有线路副本上添加测量，并保持请求中的量子比特顺序，不负责平台认证、
异步任务管理或测量误差缓解。

接口、结果语义、端序约定和边界条件见
[Sampler 实现](docs/sampler.md)。

## Estimator 组件

`StatevectorEstimator` 使用 Cqlib 态矢量精确计算 Hermitian Pauli 与 Hamiltonian
期望值；`SamplingEstimator` 注入任意符合公共契约的 `Sampler`，以有限 shots 的
局域 Pauli 基测量估计期望值及估计量方差。两者均实现批量 `Estimator` 契约，重新
核验可变请求对象，并保持参数绑定、线路量子比特插入顺序和结果顺序。

精确和采样路径的数学定义、shots 分配、方差语义、Core 兼容处理与边界条件见
[Estimator 实现](docs/estimator.md)。

## 线路纯态保真度

`ComputeUncomputeFidelity` 通过注入的 `Sampler` 执行
`U_A† U_B` 线路，以全零测量概率估计两个线路制备纯态的重叠保真度。
请求分别管理两个线路的精确参数绑定，并按量子比特插入位置建立对应关系；
实现会在单次批量采样前完整验证线路的静态可逆性与采样请求。

该组件仅面向由线路制备的纯态，不代替密度矩阵的 Uhlmann 保真度或
混合态 SWAP 测试。数学定义、统计方差、端序、线路边界与可运行示例见
[线路态保真度](docs/state_fidelity.md)。

## 梯度与量子几何组件

`cqlib_algorithms.operators.gradient` 提供参数移位、线性组合、有限差分和 SPSA
梯度，以及态矢量 QGT、QFI 与自然梯度方向。实现通过公共 `Estimator` 契约批量
评估，不绑定具体执行后端；QGT/QFI 使用 Cqlib 态矢量和生成元插入路径。

完整数学定义、支持门、边界行为和使用示例见
[梯度、量子几何张量与自然梯度组件](docs/gradient.md)。

## 经典优化器组件

`cqlib_algorithms.operators.optimizer` 提供 `GradientDescent`、`Adam` 和
`SPSAOptimizer`。前两者通过公共 `Gradient` 契约取得梯度，SPSA 每轮对每个活跃
请求执行两个目标函数评估。优化器保留未训练参数的绑定，并为每个请求独立记录
最终参数、目标值、更新次数和终止状态。

完整接口、更新公式、随机性和边界条件见
[经典优化器组件](docs/optimizer.md)。

## 布尔 Oracle 与反射算子

`BooleanFunction` 表示单输出布尔函数；`BooleanVectorFunction` 表示至少两个输出且
共用输入变量顺序的向量函数。输入量子比特 0 对应最低有效位。`BitFlipOracle` 计算
`|x>|y> → |x>|y XOR F(x)>`，每个输出位都有独立的输出量子比特；`PhaseOracle`
在单输出时施加 `(-1)**f(x)` 相位。在多输出时，必须显式给出非零 `output_mask`，
施加被选输出位的奇偶相位，不是逻辑 OR 标记。

```python
from cqlib_algorithms.operators.oracle import (
    BitFlipOracle,
    BooleanVectorFunction,
    PhaseOracle,
)

function = BooleanVectorFunction.from_truth_table(
    [(False, False), (True, False), (False, True), (True, True)]
)
bit_flip = BitFlipOracle(function).build_circuit()  # 2 inputs + 2 outputs
phase = PhaseOracle(function, output_mask=0b11).build_circuit()
```

`ZeroStateReflection`、`GeneralStateReflection` 和 `GroverDiffusion` 分别构建
`2|0><0|-I`、`2|ψ><ψ|-I` 与 `2|s><s|-I` 的精确反射线路。上述算子不执行
测量或后台任务。有限真值表综合需要指数规模的经典预处理，不提供高效黑盒 Oracle
或任意多输出 OR 标记。接口、边界、数学约定与可运行示例见
[布尔 Oracle 与反射算子](docs/oracle.md)。

## 复合算子组件

具体实现从 `cqlib_algorithms.operators` 或
`cqlib_algorithms.operators.composite` 导入：

- `GroverOperator`：相位 Oracle 与扩散/态反射组合的 Grover 迭代单元 `Q = D O`。
- `HadamardPhaseQuery`：组合 `H**n -> S_f -> H**n` 的未测量相位查询线路。

```python
from cqlib_algorithms.operators import GroverOperator, HadamardPhaseQuery
from cqlib_algorithms.operators.oracle import BooleanFunction, PhaseOracle

function = BooleanFunction.from_truth_table([False] * 7 + [True])
oracle = PhaseOracle(function).build_circuit()
iteration = GroverOperator(oracle).build_circuit()
query = HadamardPhaseQuery(function).build_circuit()
```

`GroverOperator` 接受可静态求逆的相位 Oracle 线路，默认在与 Oracle 相同量子比特
集合上构造标准 Grover 扩散。`state_preparation` 用自定义 `A` 代替默认
Hadamard 制备；全量子比特反射时这是关于 `A|0...0>` 的单态反射。
`reflection_qubits` 是真子集时，实际算子为 `A S_{0,R} A†`，关于
`A range(P_{0,R})` 子空间反射，不再是单态反射。线路时间顺序为 Oracle、
`A†`、零子空间反射、`A`，保持矩阵意义上的精确全局相位。它不选择
迭代次数、不执行测量，也不提供完整搜索算法。
组件只验证 Oracle 线路合法且可静态求逆，不做指数规模的矩阵分析来证明其
必然为对角相位 Oracle；Oracle 语义的正确性由调用方负责。

接口、数学约定与边界条件见
[复合算子组件](docs/composite.md)。

## 完整算法

`VQE` 将现有 `Ansatz`、`Estimator` 与 `Optimizer` 公共组件组合为变分量子
本征求解器。初值显式对应实际线路的活跃符号，算法不重复实现态矢量、期望值、
梯度或优化逻辑，也不绑定具体后端。数学定义、组件组合、结果语义、示例与首版边界
见 [VQE](docs/vqe.md)。

`QAOA` 在固定初态之后直接逐层组合现有的成本演化与混合演化组件，通过公共
Estimator/Optimizer 契约最小化计算基对角成本；保留显式的最大化入口。默认使用均匀叠加初态和横场
混合器，也支持同宽的固定初态、自定义有限 Hermitian 混合器与演化策略。参数按层采用
`gamma, beta` 交错顺序。数学约定、结果符号、复用边界和示例见
[QAOA](docs/qaoa.md)。两种算法共享 `VQAResult`，不嵌套另一种算法的结果；
`VQEResult` 和 `QAOAResult` 仅为该类型的兼容导出名称。

`DeutschJozsa` 的每条线路用一次相位 Oracle 调用判别布尔函数是常函数还是
平衡函数：`H**n -> S_f -> H**n` 后全零振幅的模在两种情形下恰为 1 与 0。
Oracle 由 `PhaseOracle` 从不可变布尔函数构建，执行注入任意 `Sampler` 组件。
算法层不读取真值表做经典承诺预判，但显式真值表 Oracle 的线路综合仍会读取该表；
理想无噪声承诺下单个 shot 即可确定判别，任意 Sampler 或噪声下结果只是观测判决。
多输出输入按 `output_mask` 奇偶归约为派生标量函数，承诺作用于该函数。
数学定义、查询模型、判别语义与边界见
[Deutsch-Jozsa 算法](docs/deutsch_jozsa.md)。

`BernsteinVazirani` 用一次相位 Oracle 调用恢复线性函数 `f(x) = s·x` 的隐藏
字符串：`H**n -> S_f -> H**n` 后末态恰好是 `|s>`。它与 `DeutschJozsa` 共享
`HadamardPhaseQuery` 复合算子和公共 Sampler 结果校验协议，执行注入任意
`Sampler` 组件。结果保留全部
并列的经验众数字符串，只有唯一众数时才提供 `hidden_string`；`support` 如实记录
每个众数的 shots 数。算法同样不做经典线性承诺预判，也不假定噪声模型或执行
错误纠正。数学定义、结果语义与边界见
[Bernstein-Vazirani 算法](docs/bernstein_vazirani.md)。

`Simon` 复用 `BitFlipOracle` 构建 `H**n -> U_F -> H**n` 查询线路，只测量输入
寄存器，并对观测结果执行 GF(2) 消元。在二对一 XOR 周期承诺下，测量方程均满足
`z·s=0`；秩为 `n-1` 时可唯一恢复非零周期，秩为 `n` 时对应一一函数的零周期。
结果同时保留规范独立方程和零空间基；样本不足时不虚构周期，噪声则可能改变秩，
本实现不提供鲁棒解码。算法接受单输出和多输出布尔函数，
不要求输出宽度等于输入宽度，也不经典检查承诺。数学定义、端序、结果语义与边界见
[Simon 算法](docs/simon.md)。

`CHSH` 复用 `GHZStatePreparation(2)` 制备默认 Bell 态，通过公共 `Estimator`
一次批量计算 `E00`、`E01`、`E10`、`E11` 四个 X–Z 平面乘积可观测量，并返回
`S = E00 + E01 + E10 - E11`、经典界违反判定及 Estimator 方差。也可传入固定的
二量子比特制备线路和自定义双方测量角度。数值容差只用于经典界比较，不代表统计
显著性；组件不处理实验漏洞、误差缓解或置信区间。数学定义、批处理语义和边界见
[CHSH 不等式验证](docs/chsh.md)。

`BB84` 将四态制备、共轭基测量、基筛选和抽样 QBER 检查组合为
显式可复现的协议实验。所有比特和基由调用方提供；量子线路复用
`BasisStateInitialization`，终端测量委托公共 `Sampler`。可选教材式
截获—重发路径使用 Eve 的实际测量结果构建重发信号。结果仅是筛选数据、
抽样误码率和候选原始密钥材料，不表示已完成认证、纠错、隐私放大或安全密钥
生成。完整协议语义与边界见 [BB84 协议实验](docs/bb84.md)。

## 量子傅里叶变换

`QFT` 构建新的 Cqlib `Circuit`。正向变换采用正指数约定，量子位 `0`
是计算基索引的最低有效位；默认加入输出位反转所需的 SWAP 门，因此线路矩阵可直接
对应标准离散傅里叶矩阵。

```python
from cqlib_algorithms import QFT

forward = QFT(3).build_circuit()
inverse = QFT(3, inverse=True).build_circuit()
approximate = QFT(
    4,
    approximation_degree=1,
    do_swaps=False,
).build_circuit()
```

`approximation_degree=0` 保留全部受控相位门。值增大时，会按量子位距离省略角度
最小的受控相位门；其取值必须满足
`0 <= approximation_degree < num_qubits`。当 `do_swaps=False` 时，输出量子位
保持位反转顺序。`inverse=True` 返回相同近似与 SWAP 配置下完整正向线路的伴随。
`QFT` 是冻结且使用 slots 的标准 dataclass，可使用 `dataclasses.replace` 派生新配置；
它是可复用线路变换，不属于可训练 `Ansatz` 类型层级。

## 开发

运行依赖为 `cqlib>=1.4.0b1,<1.5` 和 `numpy>=2.1.0`。首版协议已针对 Cqlib `develop`
分支提交 `c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85` 的 Python 接口核验。
SDK 尚未发布期间由开发者手动安装。
使用已有 `.venv` 时跳过创建环境的步骤：

```bash
uv venv --python 3.10
source .venv/bin/activate
```

将 Cqlib 安装到该环境后：

```bash
uv pip install --group dev
uv pip install --no-deps -e .
pytest
mypy src/cqlib_algorithms
ruff check .
ruff format --check .
python -m build
```

Windows PowerShell 使用 `.venv/Scripts/Activate.ps1` 激活环境。
不激活环境时，使用 `uv run --no-sync pytest`、`uv run --no-sync mypy`、
`uv run --no-sync ruff check .`、`uv run --no-sync ruff format --check .` 和
`uv run --no-sync python -m build`。
构建产物位于 `dist/`；隔离构建需要获取 Hatchling，但不需要安装运行时 SDK。

SDK 尚未发布期间不执行 `uv sync`。依赖可从包源获取后，再生成并提交 `uv.lock`。

开发规范见 [docs/development.md](docs/development.md)，多 Agent 开发流程见
[docs/cqlib-agent-development-workflow.md](docs/cqlib-agent-development-workflow.md)。许可证见
[LICENSE.txt](LICENSE.txt)。
