# 振幅估计

本库提供标准振幅估计、迭代振幅估计、最大似然振幅估计和快速振幅估计。四者共享
`AmplitudeEstimationProblem`，由调用方显式提供：

- 状态制备 `A`，可传入 `StatePreparation` 组件；
- Grover 迭代 `Q`，可传入 `GroverOperator` 组件；
- 非空且无重复的目标量子比特位置；
- 将目标寄存器测量字符串判为好态或坏态的谓词。

线路按量子比特插入位置映射到连续位置 $`0,\ldots,n-1`$。Sampler 返回的字符串右端
对应 `objective_qubits` 的第一项。问题对象保存两条线路的独立副本，要求它们同宽、静态、酉且所有参数已绑定。
构造 `Q` 时应使用同一初态制备，并让 Oracle 标记与好态谓词一致的子空间。
`problem.state_preparation` 保存固定制备组件，`problem.grover_operator` 保存独立的迭代组件副本；
`preparation_circuit` 和 `iteration_circuit` 是供执行使用的线路快照，应保持不变。
四种算法均可调用 `run(problem, ...)` 估计一个问题，也可调用 `run(requests)` 批量估计。
批量请求自行指定执行参数，不能同时传入单任务的关键字配置。
标准 AE 测量相位寄存器。其余三种方法对目标寄存器采样，再用谓词判断好态。FAE 对输入还有小振幅要求。

若

```math
A|0\rangle=\sqrt{1-a}\,|\psi_{\mathrm{bad}}\rangle+\sqrt{a}\,|\psi_{\mathrm{good}}\rangle,\qquad a=\sin^2\theta,\quad \theta\in\left[0,\frac{\pi}{2}\right]
```

则相关二维子空间上的 $`Q`$ 本征值为 $`e^{\pm2i\theta}`$，并且
$`Q^kA`$ 的好态概率为 $`\sin^2((2k+1)\theta)`$。

## 标准振幅估计

`StandardAmplitudeEstimation` 复用通用 `PhaseEstimationCircuit`，使用 $`m`$ 个估计
量子比特、完整逆 QFT，以及受控 $`Q^{2^j}`$。因此标准振幅估计与 QPE 共享同一套
全局相位、量子比特位序和内存限制处理。当前 Core 没有将任意线路提升为量子受控
线路的通用接口，因此底层使用完整稠密矩阵：

```math
U=\mathrm{matrix}(Q),\qquad U_p=U^{2^j},\qquad C_p=\mathrm{diag}(I,U_p)
```

`Circuit.to_matrix` 的顺序从最低有效位到最高有效位，而数值酉门追加列表的第一项是
门局部矩阵的最高有效张量因子。因此 $`C_p`$ 必须追加到
`[control, *reversed(system_targets)]`。实现保留 $`Q`$ 的全局相位，并在分配前执行
固定内存上限检查。该路径的矩阵存储和计算均为指数规模；生成的数值酉门也不保证
每个远端后端都能接受或分解。

测得估计寄存器整数 `y` 后，先归入规范振幅网格
$`x=\min(y,2^m-y)`$。有限 shots 下会把同一振幅的两个相位代表的频数相加，再选频数
最大的规范网格；规范网格并列时选择最小 $`x`$。返回值为

```math
\widehat{a}=\sin^2\left(\frac{\pi x}{2^m}\right)
```

`grid_point` 就是上述规范网格 $`x`$，其范围为 $`0,\ldots,2^{m-1}`$；
`evaluation_probabilities` 仍保存全部 $`2^m`$ 个原始相位经验概率，包括零概率网格点。
有限 shots 结果不额外声明严格置信区间。
$`\mathtt{num\_grover\_queries}=2^m-1`$ 表示线路中的 Grover 幂次数总和，
$`\mathtt{num\_oracle\_queries}=\mathtt{shots}(2^m-1)`$ 表示按 shots 加权的查询数。

下面估计单比特态中 `|1⟩` 的概率。将同一个 `preparation` 交给 Grover 迭代和问题对象，
前者用它构造关于初态的反射，后者用它初始化系统寄存器。

```python
import math

from cqlib_algorithms import AmplitudeEstimationProblem, StandardAmplitudeEstimation
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)

theta = math.pi / 8
marked = 1
preparation = AmplitudeEncoding((math.cos(theta), math.sin(theta)))
oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [marked]))
iteration = GroverOperator(oracle=oracle, state_preparation=preparation)
problem = AmplitudeEstimationProblem(
    state_preparation=preparation,
    grover_operator=iteration,
    objective_qubits=(0,),
    is_good_state=lambda bits: bits == str(marked),
)

ae = StandardAmplitudeEstimation(StatevectorSampler())
result = ae.run(problem, num_evaluation_qubits=3, shots=256)
print(result.estimation)  # 0.14644660940672624
```

相位寄存器使用 3 个量子比特。在理想仿真中，相位读数为 1 或 7，两者对应相同的概率 `sin²(π/8)`。
将 `marked` 改为 `0` 即可估计同一初态中 `|0⟩` 的概率，约为 `0.8535533906`。
一次运行多个问题时，将各自的 `StandardAmplitudeEstimationRequest` 放入列表，调用 `ae.run(requests)`，结果按请求顺序返回。

## 迭代振幅估计

IAE 复用标准 AE 的 `AmplitudeEstimationProblem`。每轮执行 $`Q^kA`$，
测量 `objective_qubits` 并用 `is_good_state` 统计好态。若初态的好态概率为 `a = sin²(θ)`，
这一轮观测到好态的概率为 $`\sin^2((2k+1)\theta)`$。

下面用振幅比 3∶1 制备单比特态，估计得到 `1` 的概率，解析值为 `1 / (3² + 1²) = 0.1`。
已有标准 AE 的 `problem` 时，可直接从创建 `iae` 的一行开始。

```python
from cqlib_algorithms import AmplitudeEstimationProblem, IterativeAmplitudeEstimation
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)

marked = 1
preparation = AmplitudeEncoding((3, 1))
oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [marked]))
iteration = GroverOperator(oracle, state_preparation=preparation)
problem = AmplitudeEstimationProblem(
    state_preparation=preparation,
    grover_operator=iteration,
    objective_qubits=(0,),
    is_good_state=lambda bits: bits == str(marked),
)

iae = IterativeAmplitudeEstimation(StatevectorSampler())
result = iae.run(problem, epsilon=0.01, alpha=0.05, shots=500)
print(result.estimation, result.confidence_interval)
for k, shots, interval in zip(
    result.powers, result.round_shots, result.round_intervals
):
    print(k, shots, interval)
```

**选择幂次。** 首轮取 `k=0`，直接采样初态。以后从大到小考察候选幂次，要求放大系数
`4k+2` 至少是上一轮的两倍，且当前角度区间经放大后落在同一半圆。这样观测概率可以在一个单调分支上反推出角度。
若没有符合条件的更大幂次，就在原幂次继续采样。

**更新区间。** 对当前幂次累计好态数和采样数，先为该幂次的好态概率建立 Chernoff–Hoeffding 区间，
再沿选定分支反解角度，与上一轮的角度区间取交集。交集为空时沿用旧区间。
将新区间两端映射为初态好态概率，按轮保存在 `round_intervals` 中。最终 `estimation` 是最后一个区间的中点。

论文 Algorithm 1 的理论轮数界

```math
T=\left\lceil\log_2\left(\frac{\pi}{8\epsilon}\right)\right\rceil
```

用于分配总失败概率预算 `alpha`。设 `N_total` 为当前幂次累计采样数，概率区间的半径为

```math
\epsilon_a=\sqrt{\frac{\log(2T/\alpha)}{2N_{\mathrm{total}}}}
```

区间端点截到 `[0, 1]`。为确定每轮实际采样数，先从基准 `shots=N_shots` 计算

```math
L_{\max}=\arcsin\left(\left(\frac{2\log(2T/\alpha)}{N_{\mathrm{shots}}}\right)^{1/4}\right)
```

当 $`K=4k+2>\lceil L_{\max}/\epsilon\rceil`$ 时，该轮实际 shots 降为
$`\lceil N_{\mathrm{shots}}L_{\max}/(10\epsilon K)\rceil`$；`round_shots` 与按 shots 加权的查询数均记录
实际值。binary64 实现要求
$`\pi\mathrm{ulp}(0.25)\leq\epsilon<\pi/8`$：除以 $`\pi`$ 后的规范角度目标至少覆盖
`[0,0.25]` 上最坏的相邻浮点间距。`max_rounds` 至少为 `T`，并且基准 shots 必须使
`Lmax` 的实数构造成立。基准 `shots` 还必须不超过 $`2^{53}-1`$，以保证参与
binary64 Chernoff--Hoeffding 几何计算的整数频数可精确表示，并避免超大 Python
整数隐式转换造成非契约异常。

幂次选择中的 `Kmax`、整数圈编号和上下半圆包含关系均以
`Fraction.from_float` 表示调用时的实际 binary64 区间端点，再做精确有理比较；不以
固定浮点容差把跨越半圆边界的区间视为合法。同样，区间更新的圈编号也由旧下界的
精确有理表示计算。

批量调用使用 `iae.run(requests)`，每个 `IterativeAmplitudeEstimationRequest` 自行设置精度、失败概率预算、采样数和轮数上限。
同一轮的全部活动问题合并为一次采样提交。已收敛的问题退出后续轮次，最终结果保持输入顺序。

以弧度计的角度区间宽度不超过 $`2\epsilon`$ 且初态好态概率区间宽度不超过
$`2\epsilon`$ 后才停止；构造结果前还会直接复核后一个条件。
`T` 是置信预算所覆盖的同幂次数据组数界，相同 `k` 的多轮采样合并为一组；
`max_rounds` 则是实际 Sampler 迭代次数的独立硬上限，不参与置信预算。因此
$`\mathtt{max\_rounds}\geq T`$ 只是合法性条件，不保证给定观测一定能在该上限内收敛。达到上限
仍未收敛时抛出 `RuntimeError`，不会把未收敛结果伪装成成功。结果保留逐轮幂次、
实际 shots 和振幅区间。

## 最大似然振幅估计

MLAE 使用与标准 AE、IAE 相同的 `AmplitudeEstimationProblem`。预先指定从 `0` 开始、严格递增的幂次元组 `powers`，
算法为每个幂次构造 $`Q^kA`$，一次提交全部线路，每条线路独立采样 `shots` 次。
下面估计同一振幅比为 3∶1 的初态中测到 `1` 的概率。若已创建上一节的 `problem`，直接从创建 `mlae` 的一行开始。

```python
from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    MaximumLikelihoodAmplitudeEstimation,
)
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)

marked = 1
preparation = AmplitudeEncoding((3, 1))
oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [marked]))
iteration = GroverOperator(oracle, state_preparation=preparation)
problem = AmplitudeEstimationProblem(
    state_preparation=preparation,
    grover_operator=iteration,
    objective_qubits=(0,),
    is_good_state=lambda bits: bits == str(marked),
)

mlae = MaximumLikelihoodAmplitudeEstimation(StatevectorSampler())
result = mlae.run(problem, powers=(0, 1, 2, 4), shots=500)
print(list(zip(result.powers, result.good_counts)))
print(result.estimation, result.asymptotic_interval)
```

设初态好态概率为 `a = sin²(θ)`。对于给定的候选角度，各幂次分别预测好态概率
$`\sin^2((2k+1)\theta)`$。
例如在解析值 `a=0.1` 处，幂次 `0、1、2、4` 依次预测 `0.1、0.676、0.99856、0.059228416`。
对每个幂次，算法用 `is_good_state` 统计目标比特上的好态数 $`h_k`$，
再把各组二项计数的对数似然相加，在 $`\theta\in[0,\pi/2]`$ 上求最大值，其中 `N=shots` 是每个幂次的采样数。

```math
\sum_k\left[h_k\log\left(\sin^2((2k+1)\theta)\right)+(N-h_k)\log\left(\cos^2((2k+1)\theta)\right)\right]
```

最终 `estimation` 为最优角度的正弦平方。示例共采样 2,000 次，`good_counts` 与 `powers` 逐项对应。
批量运行时，每个问题使用 `MaximumLikelihoodAmplitudeEstimationRequest(problem, powers, shots)` 指定实验配置。
`mlae.run(requests)` 将所有线路合并提交，各问题单独拟合，结果按请求顺序返回。

公共请求要求 $`\mathtt{shots}\leq2^{53}-1`$。Core 的结果对象在 64 位主机上以 Rust `usize`
承载 shots，表示范围更大；这里采用更严格的算法边界，是因为候选生成的一阶导数
几何仍使用 binary64，只有不超过该上限的整数频数才能保证精确表示。超限请求会在
调用 Sampler 前失败。内部高精度数学搜索可独立验证更大计数，但不扩大公共执行契约。

似然通常多峰。实现用所有正弦和余弦奇点划分完整物理区间；对数似然在每个所得开
区间内严格凹，因此其一阶导数单调递减。每个区间通过稳定 score 的符号二分定位唯一
内部根；若无根则由单边导数极限选择合法端点。边界以 $`\theta/\pi`$ 的有理数表示，从
代数上识别概率恰为 $`0`$ 或 $`1`$ 的项；开区间内部直接累计稳定的
$`2\log\vert \sin(\cdot)\vert `$ 和 $`2\log\vert \cos(\cdot)\vert `$，不以宽松裁剪替代零概率语义。
只有 score 根的最终包围区间能证明它距离某个精确边界不超过 `search_precision` 时，
才把结果吸附到该边界；不会因为巨大总对数似然在浮点中形成平台而统一偏向边界。
跨区间的全域候选改用标准库 `Decimal` 重新评分：内部候选从其 binary64 `theta` 的
所在的精确 Fraction 奇点区间开始，用 Decimal score 独立细化根，而不是直接评分
用户精度下的 binary64 中点；随后按高精度 $`\pi`$ 做模约化，并用高精度 Taylor 正弦、
余弦与 Decimal 对数计算似然。整个过程不会先计算双精度三角函数再转换。精确有理
边界继续代数处理概率 $`0/1`$。工作精度至少覆盖 shots 与最大频率的十进制位数并保留
guard digits，而且用两档精度复算候选排序；若排序改变，再提高一档，仍不稳定则显式
报错。全域比较不使用按似然绝对值或 shots 放大的相对容差。公开
`log_likelihood` 仍是选中候选对应的有限 float。低于
$`[0,\pi/2]`$ 浮点分辨率的 `search_precision` 会在执行前被拒绝，搜索还包含有限迭代
与浮点停滞保护。`search_precision` 控制角度搜索的数值容差，单位为弧度。
`asymptotic_interval` 与 `confidence_interval` 给出同一 95% 渐近区间，由 Fisher 信息构造并映射回好态概率。其覆盖率采用大样本正态近似。

## 快速振幅估计

`FasterAmplitudeEstimation` 复用标准 AE 的 `AmplitudeEstimationProblem`。每轮采样普通线路 $`Q^kA`$。
第一阶段依次取幂次 1、2、4 等，从每轮好态频率估计余弦并反解角度区间。达到转换条件后，
第二阶段每轮取两个幂次，利用两组计数区分同一余弦值对应的不同相位，再结合上一轮区间确定当前角度范围。
达到 `max_iterations` 后，将最终角度区间的中点取正弦平方，得到输入问题的好态概率估计。

输入问题须满足 $`\sqrt{a_{\mathrm{problem}}}\leq0.25`$，即好态概率至多为 `1/16`。
`amplitude_rescaling_factor` 声明输入问题已经采用的振幅缩放系数。

- `1.0` 是默认值，直接估计当前问题的好态概率。
- `0.25` 适用于已把目标振幅缩小到四分之一的问题。调用方须提供配套的初态、Grover 迭代和好态判据。
  FAE 将估计概率及区间端点除以 $`0.25^2`$，恢复原问题的概率尺度。

总失败概率预算 $`\alpha`$ 在至多 $`2\mathtt{max\_iterations}`$ 次余弦估计之间均分。单次预算为
$`\delta_c=\alpha/(2\mathtt{max\_iterations})`$，第一、第二阶段每条线路的 shots 分别为

```math
\left\lceil1944\log\left(\frac{2}{\delta_c}\right)\right\rceil,\qquad\left\lceil972\log\left(\frac{2}{\delta_c}\right)\right\rceil
```

结果中的 $`\mathtt{success\_probability\_lower\_bound}=1-\alpha`$ 是联合界给出的理论下界，不是从
本次样本估计出的经验成功概率。`powers`、`circuit_shots`、`good_counts` 和逐轮
`theta_intervals` 记录完整执行轨迹。查询统计沿用本库口径：
`num_grover_queries` 是所有线路中 Grover 指数之和，`num_oracle_queries` 是按每条
线路 shots 加权后的指数之和。因为当前线路构造显式重复 $`Q`$，请求会在构造前拒绝
单条线路超过一百万次 Grover 重复的规模。

下面用振幅比 4∶1 制备单比特态，标记 `1` 的概率为 `1/17`，可以直接估计。

```python
from cqlib_algorithms import AmplitudeEstimationProblem, FasterAmplitudeEstimation
from cqlib_algorithms.operators import (
    AmplitudeEncoding,
    BooleanFunction,
    GroverOperator,
    PhaseOracle,
    StatevectorSampler,
)

marked = 1
preparation = AmplitudeEncoding((4, 1))
oracle = PhaseOracle(BooleanFunction.from_marked_states(1, [marked]))
iteration = GroverOperator(oracle, state_preparation=preparation)
problem = AmplitudeEstimationProblem(
    state_preparation=preparation,
    grover_operator=iteration,
    objective_qubits=(0,),
    is_good_state=lambda bits: bits == str(marked),
)

fae = FasterAmplitudeEstimation(StatevectorSampler())
result = fae.run(problem, alpha=0.05, max_iterations=5)
print(result.estimation, result.confidence_interval)
print(result.powers)
```

每条线路的采样数由 `alpha` 和 `max_iterations` 决定。批量运行时，每个问题通过
`FasterAmplitudeEstimationRequest(problem, alpha, max_iterations)` 指定自己的配置。
`fae.run(requests)` 在每轮把仍需执行的线路合并为一次采样提交，最终按请求顺序返回各自的结果。

## 理论来源

- G. Brassard, P. Høyer, M. Mosca, A. Tapp，*Quantum Amplitude
  Amplification and Estimation*，arXiv:quant-ph/0005055，Algorithm `Est_Amp`
  与 Theorem 12。
- D. Grinko, J. Gacon, C. Zoufal, S. Woerner，*Iterative Quantum Amplitude
  Estimation*，arXiv:1912.05559v3，Algorithms 1–2 与 Theorem 1。
- Y. Suzuki 等，*Amplitude Estimation without Phase Estimation*，
  arXiv:1904.10246v2，Eqs. (4)–(13)。
- K. Nakaji，*Faster Amplitude Estimation*，Quantum Information &
  Computation 20(13&14), 1109–1123, 2020，arXiv:2003.02417，Algorithm 1。

