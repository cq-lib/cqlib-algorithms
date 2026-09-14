# 开发规范

## 结构

- 使用 `src/cqlib_algorithms/` 布局。稳定的公共协议集中放在 `protocols/`，可复用的线路变换放在 `transforms/`，其他具体组件按实际功能增加模块或子包。
- 完整算法放在 `algorithms/`，随实际实现按功能增加模块；不预建空分类目录。
- 不设置泛化的 `components` 或 `contracts` 层；`protocols` 只收纳已经冻结并被多个实现复用的公共契约。
- `StatePreparation`、`Ansatz`、`Sampler`、`Estimator` 和 `Gradient` 是首批公共结构协议；请求和结果类型与对应协议放在同一模块。
- 稳定协议同时从 `cqlib_algorithms.protocols` 和包顶层导出，具体实现从所属功能模块导出。
- 结构协议不启用运行时检查。`StatePreparation` 与 `Ansatz` 目前结构等价，其角色差异由调用位置和文档语义约束。
- 本库批量 `Estimator` 与 `cqlib.error_mitigation.Estimator` 回调类型属于不同契约；同时使用时必须显式别名，避免短名覆盖。
- 测试放在 `tests/` 并镜像源码功能结构；协议测试位于 `tests/protocols/`，不修改 `sys.path` 绕过包安装。

## 接口与依赖

- 复用 `cqlib` 的线路、门、算子和结果类型；调用前核对真实接口，不臆造 API。
- 执行实现由外部提供，运行时传入符合统一 Cqlib 接口与结果语义的对象。
- 本库不定义平台适配、登录、任务提交基础设施，也不依赖特定平台 SDK。仅使用相同名称的方法不足以保证兼容。
- 算法侧只定义 Sampler、Estimator 和 Gradient 所需的最小同步批处理协议，不定义后端提交、轮询、认证或传输协议；平台能力由外部适配器实现。
- 请求数据结构仅浅冻结；Cqlib 的线路和可观测量仍可变，调用方不得在请求生命周期内修改它们，执行实现必须在运行前重新验证可检测的不变量。
- 当前 Core 兼容基线为 Python 包 `cqlib>=1.4.0b1,<1.5`；扩大版本范围前必须通过兼容性测试。
- 保持依赖无环；新增公共接口或第三方依赖时说明必要性。

## QFT 约定

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

- 每个公共算子独占一个模块，并仅实现冻结协议的 `validate()`、
  `build_circuit(prefix)`、`num_parameters()` 与 `num_qubits()` 四个公共方法。
- 输入在构造时复制。固定数值态不公开可变配置，`build_circuit` 每次返回独立线路；
  数值组件不创建符号参数，因此 `num_parameters()` 返回 0。
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
