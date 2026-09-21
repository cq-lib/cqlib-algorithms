# cqlib-algorithms

面向 Cqlib 的量子算法组件库。支持 Python 3.10+，使用 uv 管理依赖、pytest 测试，
采用 Apache-2.0 开源协议。当前提供第一批公共组件协议、请求与结果数据结构，以及
首批状态制备、数据编码组件、变分拟设、QAOA 演化层、布尔 Oracle、反射算子、
Pauli 测量规划、同步采样器、期望值估计器、梯度、量子几何与经典优化组件，
以及可复用的量子傅里叶变换线路构造器。

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
tests/protocols/
tests/operators/state_preparation/
tests/operators/ansatz/
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
