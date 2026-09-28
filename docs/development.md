# 开发规范

## 结构

- 使用 `src/cqlib_algorithms/` 布局。稳定的公共协议集中放在 `protocols/`，可复用的线路变换放在 `transforms/`，其他具体组件按实际功能增加模块或子包。
- 完整算法放在 `algorithms/`，随实际实现按功能增加模块；不预建空分类目录。
- 不设置泛化的 `components` 或 `contracts` 层；`protocols` 只收纳已经冻结并被多个实现复用的公共契约。
- `StatePreparation`、`Ansatz`、`Sampler`、`Estimator`、`Gradient` 和 `Optimizer` 是首批公共抽象基类；请求和结果类型与对应契约放在同一模块。
- 稳定协议同时从 `cqlib_algorithms.protocols` 和包顶层导出，具体实现从所属功能模块导出。
- 受公共抽象基类约束的具体组件必须显式继承对应基类。`StatePreparation` 与 `Ansatz` 具有相同抽象操作，但属于不同的名义类型层级，并分别约束状态制备参数和可训练参数语义。线路变换等独立值类型不为满足名义类型而强行继承不匹配的抽象基类。
- 本库批量 `Estimator` 与 `cqlib.error_mitigation.Estimator` 回调类型属于不同契约；同时使用时必须显式别名，避免短名覆盖。
- 测试放在 `tests/` 并镜像源码功能结构；协议测试位于 `tests/protocols/`，不修改 `sys.path` 绕过包安装。

## 接口与依赖

- 复用 `cqlib` 的线路、门、算子和结果类型；调用前核对真实接口，不臆造 API。
- 执行实现由外部提供，运行时传入符合统一 Cqlib 接口与结果语义的对象。
- 本库不定义平台适配、登录、任务提交基础设施，也不依赖特定平台 SDK。仅使用相同名称的方法不足以保证兼容。
- 算法侧只定义 Sampler、Estimator、Gradient 和 Optimizer 所需的最小同步批处理协议，不定义后端提交、轮询、认证或传输协议；平台能力由外部适配器实现。
- 要求完整原始频数的算法使用 `protocols.sampler` 的公共结果校验函数，统一检查
  Sampler 返回容器、结果类型、执行状态、shots、qubit 顺序和完整计数；允许后选择
  或过滤频数的调用方不得误用该严格契约。
- 请求数据结构仅浅冻结；Cqlib 的线路和可观测量仍可变，调用方不得在请求生命周期内修改它们，执行实现必须在运行前重新验证可检测的不变量。
- 当前 Core 兼容基线为 Python 包 `cqlib>=1.4.0b1,<1.5`；扩大版本范围前必须通过兼容性测试。
- 保持依赖无环；新增公共接口或第三方依赖时说明必要性。

## QFT 约定

- `QFT` 是冻结且使用 slots 的标准 dataclass，通过 `__post_init__` 校验配置并保留 dataclass 的标准构造、比较、哈希和 `dataclasses.replace` 行为；它是线路变换，不继承 `Ansatz`。
- 正向 QFT 使用 `F[y, x] = exp(2π i x y / 2**n) / sqrt(2**n)`，量子位 `0` 是最低有效位。
- 精确受控相位使用 `StandardGate.Phase(theta).control(1)` 与 `Circuit.append_mc_gate`；`CRZ` 的相对相位不同，不能替代。
- `approximation_degree=d` 时仅保留满足 `distance < n - d` 的受控相位门。
- 默认 SWAP 将输出恢复为自然计算基顺序；关闭 SWAP 时必须明确记录输出位反转语义。
- 逆变换是相同近似和 SWAP 配置下完整正向线路的伴随。
- 正确性测试使用独立解析傅里叶矩阵和基态演化验证，不以实现自身作为参考。

## 代码与验证

- 支持 Python 3.10+；代码、标识符、注释、文档字符串和异常信息使用英文，README 仅使用中文。
- 使用类型标注；公共接口说明输入、输出、边界条件和异常。
- 算法实现明确数学定义、端序、相位和数值容差，使用解析结果或独立参考验证；不以放宽容差掩盖错误。
- 使用 pytest；发布的 `py.typed` 源码使用 mypy 严格检查。代码、必要测试和使用说明同步更新；不提前实现无需求的抽象。
- 使用 uv 管理环境；未发布 SDK 由开发者手动安装，不运行自动同步。验证使用 `uv run --no-sync`。
- 版本号维护在 `pyproject.toml`。打包使用 `python -m build`，采用 Apache-2.0，保留 `LICENSE.txt`。

## 状态制备实现约定

- 每个公共算子独占一个模块，显式继承 `StatePreparation`，并实现冻结契约的 `validate()`、
  `build_circuit(prefix)`、`num_parameters()` 与 `num_qubits()` 四个公共方法。
- 输入在构造时复制。固定数值态不公开可变配置，`build_circuit` 每次返回独立线路；
  数值组件不创建符号参数，因此 `num_parameters()` 返回 0。
- `validate_*` 辅助函数和组件 `validate()` 只检查现有值、成功时返回 `None`，不得复制、
  排序、归一化或丢弃转换结果；`copy_*`、`normalize_*`、`canonicalize_*` 只在构造阶段
  生成并返回内部标准快照。
- 基态索引和幅度索引遵循 Cqlib 小端约定：量子比特 0 是最低有效位；乘积态输入的
  第 `q` 项对应量子比特 `q`。数值输出测试必须比较 `Circuit.to_matrix()` 的第一列，
  逐幅度核验相位，不以保真度忽略全局相位。
- `ArbitraryStatePreparation` 要求输入已在相对、绝对误差 `1e-12` 内归一化；
  `AmplitudeEncoding` 自动归一化非零输入。二者使用 Möttönen 均匀受控 RY/RZ
  综合，并仅生成标准旋转、CX 及必要的多控门，不生成稠密 `UnitaryGate`。
- `SparseAmplitudeEncoding` 对显式支持集自动归一化，拒绝重复索引和显式零幅度。
  其 exact support-set two-level 综合的经典存储为 `O(s+n)`，逻辑多控操作为
  `O(sn)`，open-control 周边单比特操作为最坏 `O(sn²)`。当前 Core 无辅助量子比特
  的多控门分解使基础门上界为最坏 `O(sn³)`。该实现不作量子随机访问存储器
  （QRAM）能力或多项式以下的数据加载复杂度声明。
- `AngleEncoding` 仅接受未绑定的 `RX`、`RY`、`RZ` 或 `Phase`；
  `FeatureMapPreparation` 仅接受 Cqlib 的 `ZFeatureMap`、`IQPFeatureMap`、
  `ZZFeatureMap` 或 `PauliFeatureMap`。两者保存 Core 组件快照并委托 Core 构建。
- `UserCircuitPreparation` 保存用户线路快照，按输入线路的量子比特插入顺序映射到
  连续的 `0..n-1`，保留活跃符号和全局相位。线路必须能由 Core 静态求逆；测量、
  重置、动态控制及 Core 当前不能求逆的三量子比特以上数值酉门会被拒绝。
- 数值组件忽略合法字符串 `prefix`，Core 参数化编码及 Feature Map 使用它命名符号；
  User Circuit 保留用户原有符号名，不进行重命名。

## Ansatz 实现约定

- `Ansatz` 只定义公共抽象契约；`BaseAnsatz` 提供公共实现，统一 prefix、配置和 Core
  线路校验，并直接代理兼容的内部构建器。每个公共拟设或演化层独占一个模块并继承
  `BaseAnsatz`，仅在语义确有差异时覆盖受保护钩子。
- Core 中标记为 final 的 `TwoLocal`、`PauliEvolutionAnsatz` 等类型仅通过组合和委托
  复用，不通过继承绕过 Core 类型边界。
- `num_qubits` 必须为正整数，`reps` 必须为非负整数，且 `bool` 不作为整数接受；
  组件在构造时完成输入复制、随机配置冻结和验证，重复构建返回独立线路。
- TwoLocal 系列参数顺序固定为旋转层、量子比特、旋转门位置。旋转门仅接受未绑定
  `RX`、`RY`、`RZ`，纠缠门仅接受未绑定 `CX`、`CY`、`CZ`；拓扑使用 Core
  `EntanglementTopology` 并由 Core 继续验证自定义边。
- `PauliTwoDesignAnsatz` 在构造时抽样并冻结随机轴，使用 `RY(pi/4)` 初始层、
  `RX`/`RY`/`RZ` 参数层和 even/odd 最近邻 `CZ` brickwork。测试固定 seed 的
  操作快照，同时用独立矩阵验证，而不要求与其他框架共享随机数序列。
- QAOA 成本层和混合层分别使用 `{prefix}_gamma` 与 `{prefix}_beta`，防御性复制
  Hamiltonian，并委托 Core `PauliEvolutionAnsatz`。单层组件不隐式添加初态或组合
  为完整 QAOA。
- UCCSD 与 k-UpCCGSD 在具备经过评审的费米算子、参考态、激发枚举和映射器前不建
  占位实现；相关 Core 能力边界记录在 `docs/ansatz-core-gap-report.md`。

## 复合算子实现约定

- 复合算子独占 `operators/composite/` 功能子包，由已冻结的基础算子组合而成，
  不重复实现基础组件的线路构造；每个公共复合算子独占一个模块。
- `GroverOperator` 构建单次 Grover 迭代 `Q = D_R O`：Oracle 线路在前，
  `D_R = A S_{0,R} A†` 在后，其中 `S_{0,R} = 2P_{0,R}-I`，`P_{0,R}` 投影
  到 `R` 中所有量子比特为零的子空间。时间顺序为 Oracle、`A†`、
  零子空间反射、`A`；采用不带前置负号的相位约定。
- Oracle 与 `state_preparation` 必须是合法且可静态求逆的 Cqlib `Circuit`，
  构造时复制并归一化到连续量子比特 `0..n-1`；非酉或动态操作被拒绝。
- 默认扩散在 `reflection_qubits` 子集上作用 Hadamard 与零态反射；子集为全部
  搜索量子比特时必须与 `GroverDiffusion` 的矩阵一致，并且仅此时是关于
  单一制备态的秩一反射。真子集路径是关于 `A range(P_{0,R})` 的子空间
  反射。`reflection_qubits` 构造时校验、排序并冻结，不允许为空、
  重复、越界或布尔值；重复值被拒绝，不会被静默去重。
- 复合算子不选择迭代次数、不执行测量或采样，也不承诺查询复杂度。
- `HadamardPhaseQuery` 只组合 `H**n -> S_f -> H**n` 的未测量线路，函数与
  `output_mask` 语义完全委托 `PhaseOracle`；承诺判定、采样和经典后处理属于算法层。

## VQE 实现约定

- `VQE` 只组合公共 `Ansatz`、`Estimator` 和 `Optimizer` 契约；线路构建、期望值、
  梯度和参数更新分别委托已有组件，不在算法层复制实现，也不绑定具体执行后端。
- 初值必须显式、有限，并精确覆盖生成线路的活跃符号。训练顺序取
  `tuple(circuit.used_symbols)`；不根据 `Ansatz.num_parameters()` 补造退化线路中
  未激活的逻辑参数槽。
- `eigenvalue` 直接采用 `OptimizerResult.optimal_value`，不为获取终态方差重复执行
  Estimator。结果同时保留不可变最优参数映射、独立完全绑定线路和原始优化器结果。
- VQE 只保证在拟设可达子空间中最小化 Hermitian 算子期望值；结果不构成全局基态
  或经典优化全局收敛保证。

## QAOA 实现约定

- `QAOA` 只接受非空、有限、Hermitian 且计算基对角的成本 Hamiltonian。主入口
  `compute_minimum_eigenvalue` 最小化原始期望值；`compute_maximum` 显式最小化
  负成本，但演化线路仍使用原始成本。结果用 `objective_sense` 区分这两种语义。
- 默认初态复用 `UniformSuperposition`，每层依次复用 `QAOACostEvolution` 与
  `QAOAMixerEvolution`；算法层不重新实现演化、期望值、梯度、优化或参数绑定。
- 层参数顺序固定为 `{prefix}_{k}_gamma, {prefix}_{k}_beta`。自定义初态必须固定、
  无活跃符号且同宽；自定义 mixer 必须非空、有限、Hermitian 且同宽，再由底层
  演化组件验证线路语义。
- `QAOA.build_circuit` 直接拼装初态与演化层，不新建内部 Ansatz 包装；优化入口
  调用该可覆盖方法，并直接使用 Estimator/Optimizer 契约，不委托另一算法。
- VQE 与 QAOA 共享 `VQAResult`。`optimal_value` 表示原目标在最终迭代点的期望值，
  不是历史最佳值；最小化时与优化器值相同，最大化时与优化器值相反。结果构造必须
  校验该关系。不得虚构优化器未提供的 checkpoint、方差或收敛保证。
- 公共结果 dataclass 保持标准构造与 `dataclasses.replace`，简短 repr 只显示
  目标值、参数数量、方向、迭代次数与状态，不打印完整线路。

## Deutsch-Jozsa 实现约定

- `DeutschJozsa` 是注入 `Sampler` 的 Oracle 型算法：线路固定为未测量的
  `H**n -> S_f -> H**n`，末端测量由 Sampler 在私有执行线路上追加；
  Oracle 统一由 `PhaseOracle` 从布尔函数构建，函数与 `output_mask` 校验
  委托该底层组件，算法层不重复实现标记合成。
- 输入变量数至少为 1。向量函数通过 `output_mask` 奇偶定义派生标量函数，
  常/平衡承诺作用于该派生函数。算法层不读取真值表做经典承诺预判，但
  `PhaseOracle` 对显式真值表的线路综合仍然读取该表，查询复杂度结论不包含
  该指数规模的经典输入与综合成本。
- 判别只来自测量统计：仅当全零占满全部 shots 时判定"常"，否则判定"平衡"。
  该规则在理想无噪声承诺下为确定性结论且单个 shot 已足够；任意 Sampler 或
  噪声下仅表示观测判决。每条线路包含一次 Oracle 调用，多 shots 表示重复执行
  该线路，不能表述为整个采样任务总共只进行一次物理 Oracle 求值。违反常/平衡
  承诺的输入产生形式正确但语义未定义的结果。
- 整批请求先全部校验和变换，再用一次压平的 Sampler 调用执行；采样结果契约
  （返回列表、终态成功、shots 匹配、qubit 顺序、频数总数等于 shots）与库内
  其他采样算法一致，不完整频数被拒绝。
- 请求与结果是浅冻结 dataclass；结果只记录判别与全零统计，并强制判别和
  `zero_count == shots` 一致，不虚构后端未提供的置信度或误差信息。

## Bernstein-Vazirani 实现约定

- `BernsteinVazirani` 与 `DeutschJozsa` 同属注入 `Sampler` 的 Oracle 型算法，
  共享同一查询线路与采样契约：未测量的 `H**n -> S_f -> H**n` 由
  `operators/composite/HadamardPhaseQuery` 统一构建，采样结果由
  `protocols.sampler` 的公共校验函数统一核验；两种算法不各自复制这两段逻辑。
- 承诺为线性函数 `f(x) = s·x`（mod 2）时末态恰为 `|s>`；结果保留观测频数
  最高的全部经验众数字符串。只有唯一众数时 `hidden_string` 才返回候选值，平局
  返回 `None` 并由 `modal_strings` 完整表达歧义。理想执行下候选唯一且
  `support == shots`；一般采样结果不被解释为已经完成纠错或最大似然解码。
- 每个 `modal_strings` 候选的第 `q` 位对应量子比特 `q`（小端）；唯一候选时，
  `hidden_string` 和 `hidden_integer` 分别提供元组及整数视图。算法层不做经典线性
  承诺预判，违反承诺的输入结果语义未定义。
- 其余约定（多输出奇偶归约、请求/结果浅冻结、整批校验后一次压平调用、不虚构
  置信度）与 Deutsch-Jozsa 一致。

## Simon 实现约定

- `Simon` 接受 `BooleanFunction` 或 `BooleanVectorFunction`，复用 `BitFlipOracle`
  构建 `H**n -> U_F -> H**n` 未测量线路；仅输入寄存器交给公共 `Sampler` 测量，
  输出寄存器不测量。函数输出宽度 `m` 不要求等于输入宽度 `n`。
- 承诺是存在 `s`，且 `F(x) = F(y)` 当且仅当 `x = y` 或
  `x xor y = s`；`s = 0` 表示一一函数。算法层不读取真值表做经典承诺检查；
  违反承诺的输入只保证返回形式正确的方程结果，不具有周期恢复语义。
- Sampler 位串最右字符对应输入量子比特 0。经典后处理在 GF(2) 上生成规范 RREF
  独立方程和规范零空间基；零、重复与相关样本不增加秩。仅当秩为 `n - 1` 且
  零空间存在唯一非零方向时公开非零 `period`；满秩时公开全零周期，秩低于
  `n - 1` 时返回 `None`。噪声可能伪造秩，本实现不提供鲁棒解码。
- 实现不自动追加 shots、不进行噪声鲁棒解码或置信度估计。整批请求必须先验证与
  构建，再进行一次 Sampler 调用；结果使用公共完整频数契约验证。
- DJ/BV 的经典函数到查询线路能力已由
  `BooleanFunction/BooleanVectorFunction -> PhaseOracle -> HadamardPhaseQuery`
  覆盖，不增加同义构造器；显式 XOR 输出 Oracle 使用 `BitFlipOracle`。
