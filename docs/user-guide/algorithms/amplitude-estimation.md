# 振幅估计

本库提供标准振幅估计、迭代振幅估计、最大似然振幅估计和快速振幅估计。四者共享
`AmplitudeEstimationProblem`，由调用方显式提供：

- 固定、完全绑定且静态酉的状态制备线路 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;">；
- 与 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 同宽的固定、完全绑定且静态酉 Grover 迭代线路 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;">；
- 非空且无重复的目标量子比特位置；
- 将目标寄存器测量字符串判为好态或坏态的谓词。

线路按量子比特插入位置映射到连续位置 <img src="../../assets/formulas/inline/inline-80df05cab2d1ce42.svg" alt="0,\ldots,n-1" style="vertical-align: -3.000pt;">。Sampler 返回的字符串右端
对应 `objective_qubits` 的第一项，谓词直接接收这一字符串。组件验证线路机械约束，
但不通过指数规模分析证明调用方给出的 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;">、<img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 和好态谓词在数学上相互一致。

若

<p align="center">
  <img src="../../assets/formulas/ae-state-decomposition.svg" alt="A|0⟩ = √(1-a)|ψ_bad⟩ + √a|ψ_good⟩, where a = sin²θ and θ is in [0, π/2].">
</p>

则相关二维子空间上的 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;"> 本征相位为 <img src="../../assets/formulas/inline/inline-224c55075b62d340.svg" alt="e^{\pm2i\theta}" style="vertical-align: -1.000pt;">，并且
<img src="../../assets/formulas/inline/inline-cbcc0a5a5fb6988b.svg" alt="Q^kA" style="vertical-align: -3.000pt;"> 的好态概率为 <img src="../../assets/formulas/inline/inline-982c3cd3fdbf23fa.svg" alt="\sin^2((2k+1)\theta)" style="vertical-align: -4.000pt;">。

## 标准振幅估计

`StandardAmplitudeEstimation` 复用通用 `PhaseEstimationCircuit`，使用 <img src="../../assets/formulas/inline/inline-62c66a7a5dd70c31.svg" alt="m" style="vertical-align: -1.000pt;"> 个估计
量子比特、完整逆 QFT，以及受控 <img src="../../assets/formulas/inline/inline-0938b6ecbd864e65.svg" alt="Q^{2^j}" style="vertical-align: -3.000pt;">。因此标准振幅估计与 QPE 共享同一套
全局相位、量子比特位序和内存限制处理。当前 Core 没有将任意线路提升为量子受控
线路的通用接口，因此底层使用完整稠密矩阵：

<p align="center">
  <img src="../../assets/formulas/ae-controlled-power.svg" alt="U is the matrix of Q, U_p = U^(2^j), and C_p = diag(I, U_p).">
</p>

`Circuit.to_matrix` 的顺序从最低有效位到最高有效位，而数值酉门追加列表的第一项是
门局部矩阵的最高有效张量因子。因此 <img src="../../assets/formulas/inline/inline-374d4fe4488ce15e.svg" alt="C_p" style="vertical-align: -5.000pt;"> 必须追加到
`[control, *reversed(system_targets)]`。实现保留 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;"> 的全局相位，并在分配前执行
固定内存上限检查。该路径的矩阵存储和计算均为指数规模；生成的数值酉门也不保证
每个远端后端都能接受或分解。

测得估计寄存器整数 `y` 后，先归入规范振幅网格
<img src="../../assets/formulas/inline/inline-306a9f11bec98612.svg" alt="x=\min(y,2^m-y)" style="vertical-align: -4.000pt;">。有限 shots 下会把同一振幅的两个相位代表的频数相加，再选频数
最大的规范网格；规范网格并列时选择最小 <img src="../../assets/formulas/inline/inline-2d711642b726b044.svg" alt="x" style="vertical-align: -1.000pt;">。返回值为

<p align="center">
  <img src="../../assets/formulas/ae-grid-estimate.svg" alt="The amplitude estimate is sin²(πx/2^m).">
</p>

`grid_point` 就是上述规范网格 <img src="../../assets/formulas/inline/inline-2d711642b726b044.svg" alt="x" style="vertical-align: -1.000pt;">，其范围为 <img src="../../assets/formulas/inline/inline-35578a9f3f3943f4.svg" alt="0,\ldots,2^{m-1}" style="vertical-align: -3.000pt;">；
`evaluation_probabilities` 仍保存全部 <img src="../../assets/formulas/inline/inline-da9165994272339c.svg" alt="2^m" style="vertical-align: -0.000pt;"> 个原始相位经验概率，包括零概率网格点。
有限 shots 结果不额外声明严格置信区间。
<img src="../../assets/formulas/inline/inline-74d4f5dd95885b5e.svg" alt="\mathtt{num\_grover\_queries}=2^m-1" style="vertical-align: -4.000pt;"> 表示线路中的 Grover 幂次数总和，
<img src="../../assets/formulas/inline/inline-3704f4331ef2205e.svg" alt="\mathtt{num\_oracle\_queries}=\mathtt{shots}(2^m-1)" style="vertical-align: -4.000pt;"> 表示按 shots 加权的查询数。

## 迭代振幅估计

`IterativeAmplitudeEstimation` 每轮采样普通线路 <img src="../../assets/formulas/inline/inline-cbcc0a5a5fb6988b.svg" alt="Q^kA" style="vertical-align: -3.000pt;">，使用
Chernoff--Hoeffding 区间约束 <img src="../../assets/formulas/inline/inline-982c3cd3fdbf23fa.svg" alt="\sin^2((2k+1)\theta)" style="vertical-align: -4.000pt;">，再按同半圆规则选择下一轮幂次。
论文 Algorithm 1 的理论轮数界

<p align="center">
  <img src="../../assets/formulas/iae-round-bound.svg" alt="T equals the ceiling of log base 2 of π divided by 8ε.">
</p>

用于置信预算；同一幂次 `k` 再次出现时，把它的好态数和 shots 与此前数据合并，再用

<p align="center">
  <img src="../../assets/formulas/iae-amplitude-radius.svg" alt="ε_a equals the square root of log(2T/α) divided by 2N_total.">
</p>

更新区间。为遵守 no-overshooting 规则，先从基准 `shots=N_shots` 计算

<p align="center">
  <img src="../../assets/formulas/iae-angle-bound.svg" alt="L_max is the arcsine of the fourth root of 2 log(2T/α) divided by N_shots.">
</p>

当 <img src="../../assets/formulas/inline/inline-10f2ac23532f9c87.svg" alt="K=4k+2&gt;\lceil L_{\max}/\epsilon\rceil" style="vertical-align: -4.000pt;"> 时，该轮实际 shots 降为
<img src="../../assets/formulas/inline/inline-86ebb7909a09fc9a.svg" alt="\lceil N_{\mathrm{shots}}L_{\max}/(10\epsilon K)\rceil" style="vertical-align: -4.000pt;">；`round_shots` 与按 shots 加权的查询数均记录
实际值。binary64 实现要求
<img src="../../assets/formulas/inline/inline-093e67fe4dcb947d.svg" alt="\pi\operatorname{ulp}(0.25)\leq\epsilon&lt;\pi/8" style="vertical-align: -4.000pt;">：除以 <img src="../../assets/formulas/inline/inline-f1cf04093d08b6dc.svg" alt="\pi" style="vertical-align: -1.000pt;"> 后的规范角度目标至少覆盖
`[0,0.25]` 上最坏的相邻浮点间距。`max_rounds` 至少为 `T`，并且基准 shots 必须使
`Lmax` 的实数构造成立。基准 `shots` 还必须不超过 <img src="../../assets/formulas/inline/inline-9abc7e9372f4edbb.svg" alt="2^{53}-1" style="vertical-align: -0.000pt;">，以保证参与
binary64 Chernoff--Hoeffding 几何计算的整数频数可精确表示，并避免超大 Python
整数隐式转换造成非契约异常。

幂次选择中的 `Kmax`、整数圈编号和上下半圆包含关系均以
`Fraction.from_float` 表示调用时的实际 binary64 区间端点，再做精确有理比较；不以
固定浮点容差把跨越半圆边界的区间视为合法。同样，区间更新的圈编号也由旧下界的
精确有理表示计算。

同一轮所有尚未收敛的问题合并为一次 Sampler 批调用；下一轮依赖当前观测，所以
不能把全部轮次预先合并。

角度区间宽度不超过 <img src="../../assets/formulas/inline/inline-9cbc7f6855e810d2.svg" alt="2\epsilon" style="vertical-align: -1.000pt;"> 且实际 binary64 振幅区间宽度不超过
<img src="../../assets/formulas/inline/inline-9cbc7f6855e810d2.svg" alt="2\epsilon" style="vertical-align: -1.000pt;"> 后才停止；构造结果前还会直接复核后一个条件。
`T` 是置信预算所覆盖的同幂次数据组数界，相同 `k` 的多轮采样合并为一组；
`max_rounds` 则是实际 Sampler 迭代次数的独立硬上限，不参与置信预算。因此
<img src="../../assets/formulas/inline/inline-04f2c52f495ac817.svg" alt="\mathtt{max\_rounds}\geq T" style="vertical-align: -2.000pt;"> 只是合法性条件，不保证给定观测一定能在该上限内收敛。达到上限
仍未收敛时抛出 `RuntimeError`，不会把未收敛结果伪装成成功。结果保留逐轮幂次、
实际 shots 和振幅区间。

## 最大似然振幅估计

`MaximumLikelihoodAmplitudeEstimation` 对固定、严格递增且从 `0` 开始的幂次序列，
一次批量执行全部 <img src="../../assets/formulas/inline/inline-cbcc0a5a5fb6988b.svg" alt="Q^kA" style="vertical-align: -3.000pt;"> 线路。若第 <img src="../../assets/formulas/inline/inline-8254c329a92850f6.svg" alt="k" style="vertical-align: -1.000pt;"> 个实验观察到 <img src="../../assets/formulas/inline/inline-8a5a5b48b1793c8b.svg" alt="h_k" style="vertical-align: -3.000pt;"> 次好态、总 shots 为
<img src="../../assets/formulas/inline/inline-8ce86a6ae65d3692.svg" alt="N" style="vertical-align: -0.000pt;">，则在 <img src="../../assets/formulas/inline/inline-316345810c2bcc09.svg" alt="\theta\in[0,\pi/2]" style="vertical-align: -4.000pt;"> 最大化

<p align="center">
  <img src="../../assets/formulas/mlae-log-likelihood.svg" alt="Maximum-likelihood amplitude-estimation log-likelihood over experiment powers k.">
</p>

公共请求要求 <img src="../../assets/formulas/inline/inline-4ba1c74d5dbe64e9.svg" alt="\mathtt{shots}\leq2^{53}-1" style="vertical-align: -2.000pt;">。Core 的结果对象在 64 位主机上以 Rust `usize`
承载 shots，表示范围更大；这里采用更严格的算法边界，是因为候选生成的一阶导数
几何仍使用 binary64，只有不超过该上限的整数频数才能保证精确表示。超限请求会在
调用 Sampler 前失败。内部高精度数学搜索可独立验证更大计数，但不扩大公共执行契约。

似然通常多峰。实现用所有正弦和余弦奇点划分完整物理区间；对数似然在每个所得开
区间内严格凹，因此其一阶导数单调递减。每个区间通过稳定 score 的符号二分定位唯一
内部根；若无根则由单边导数极限选择合法端点。边界以 <img src="../../assets/formulas/inline/inline-3704c0ca35f6f7ec.svg" alt="\theta/\pi" style="vertical-align: -4.000pt;"> 的有理数表示，从
代数上识别概率恰为 <img src="../../assets/formulas/inline/inline-5feceb66ffc86f38.svg" alt="0" style="vertical-align: -1.000pt;"> 或 <img src="../../assets/formulas/inline/inline-6b86b273ff34fce1.svg" alt="1" style="vertical-align: -0.000pt;"> 的项；开区间内部直接累计稳定的
<img src="../../assets/formulas/inline/inline-bfb92f1b78c2ef7a.svg" alt="2\log|\sin(\cdot)|" style="vertical-align: -4.000pt;"> 和 <img src="../../assets/formulas/inline/inline-d2b59db3b74b0e9c.svg" alt="2\log|\cos(\cdot)|" style="vertical-align: -4.000pt;">，不以宽松裁剪替代零概率语义。
只有 score 根的最终包围区间能证明它距离某个精确边界不超过 `search_precision` 时，
才把结果吸附到该边界；不会因为巨大总对数似然在浮点中形成平台而统一偏向边界。
跨区间的全域候选改用标准库 `Decimal` 重新评分：内部候选从其 binary64 `theta` 的
所在的精确 Fraction 奇点区间开始，用 Decimal score 独立细化根，而不是直接评分
用户精度下的 binary64 中点；随后按高精度 <img src="../../assets/formulas/inline/inline-f1cf04093d08b6dc.svg" alt="\pi" style="vertical-align: -1.000pt;"> 做模约化，并用高精度 Taylor 正弦、
余弦与 Decimal 对数计算似然。整个过程不会先计算双精度三角函数再转换。精确有理
边界继续代数处理概率 <img src="../../assets/formulas/inline/inline-a93875fe509ac2fa.svg" alt="0/1" style="vertical-align: -4.000pt;">。工作精度至少覆盖 shots 与最大频率的十进制位数并保留
guard digits，而且用两档精度复算候选排序；若排序改变，再提高一档，仍不稳定则显式
报错。全域比较不使用按似然绝对值或 shots 放大的相对容差。公开
`log_likelihood` 仍是选中候选对应的有限 float。低于
<img src="../../assets/formulas/inline/inline-e9742eff9c4a7c8d.svg" alt="[0,\pi/2]" style="vertical-align: -4.000pt;"> 浮点分辨率的 `search_precision` 会在执行前被拒绝，搜索还包含有限迭代
与浮点停滞保护。`asymptotic_interval` 是
根据 Fisher 信息构造并映射回振幅的 95% 渐近区间，同时作为共享
`confidence_interval` 字段返回；它不是有限样本覆盖保证。

## 快速振幅估计

`FasterAmplitudeEstimation` 实现 Nakaji 的两阶段快速振幅估计，不使用相位寄存器、
受控 Grover 幂或逆 QFT。每轮执行普通线路 <img src="../../assets/formulas/inline/inline-cbcc0a5a5fb6988b.svg" alt="Q^kA" style="vertical-align: -3.000pt;">；第一阶段用一个余弦估计逐轮
缩小 <img src="../../assets/formulas/inline/inline-2b4af44cb17349bb.svg" alt="\theta" style="vertical-align: -1.000pt;"> 区间，达到相位消歧条件后，第二阶段用两个不同 Grover 幂的余弦估计
恢复正弦和余弦，从而继续倍增查询深度。同一轮所有请求合并为一次 Sampler 批调用，
后续轮次仍依赖当前观测。

算法要求实际送入的估计问题已经满足 <img src="../../assets/formulas/inline/inline-56396a782bef738c.svg" alt="\sqrt{a_{\mathrm{problem}}}\leq0.25" style="vertical-align: -5.000pt;">。请求中的
`amplitude_rescaling_factor` 只声明调用方已经施加的幅度缩放，不会改写线路：

- `1.0` 表示直接估计这个小振幅问题；
- `0.25` 表示调用方已经构造了配套的缩放后 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;">、<img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;"> 和好态谓词，返回值再除以
  <img src="../../assets/formulas/inline/inline-19626a5e79b04c3f.svg" alt="0.25^2" style="vertical-align: -1.000pt;"> 恢复原目标。

经典 `is_good_state` 谓词不足以自动生成量子 Oracle，复用未缩放问题的旧 Grover
迭代线路也不能代表缩放后的问题。因此实现不会在内部静默执行自动缩放。

总失败概率预算 <img src="../../assets/formulas/inline/inline-4893e9df8b5496eb.svg" alt="\alpha" style="vertical-align: -1.000pt;"> 在至多 <img src="../../assets/formulas/inline/inline-40d294e752c9bd80.svg" alt="2\mathtt{max\_iterations}" style="vertical-align: -2.000pt;"> 次余弦估计之间均分。单次预算为
<img src="../../assets/formulas/inline/inline-9d177c825e748e45.svg" alt="\delta_c=\alpha/(2\mathtt{max\_iterations})" style="vertical-align: -4.000pt;">，第一、第二阶段每条线路的 shots 分别为

<p align="center">
  <img src="../../assets/formulas/fae-shot-bounds.svg" alt="The two FAE shot bounds are ceil(1944 log(2/δ_c)) and ceil(972 log(2/δ_c)).">
</p>

结果中的 <img src="../../assets/formulas/inline/inline-fbb430999b68d22f.svg" alt="\mathtt{success\_probability\_lower\_bound}=1-\alpha" style="vertical-align: -4.000pt;"> 是联合界给出的理论下界，不是从
本次样本估计出的经验成功概率。`powers`、`circuit_shots`、`good_counts` 和逐轮
`theta_intervals` 记录完整执行轨迹。查询统计沿用本库口径：
`num_grover_queries` 是所有线路中 Grover 指数之和，`num_oracle_queries` 是按每条
线路 shots 加权后的指数之和。因为当前线路构造显式重复 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;">，请求会在构造前拒绝
单条线路超过一百万次 Grover 重复的规模。

## 示例

```python
import math

from cqlib import Circuit
from cqlib_algorithms import (
    AmplitudeEstimationProblem,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationRequest,
)
from cqlib_algorithms.operators import StatevectorSampler

theta = math.pi / 8
preparation = Circuit(1)
preparation.ry(0, 2 * theta)
grover = Circuit(1)
grover.ry(0, 4 * theta)
problem = AmplitudeEstimationProblem(
    preparation,
    grover,
    (0,),
    lambda bits: bits == "1",
)

algorithm = MaximumLikelihoodAmplitudeEstimation(StatevectorSampler())
result = algorithm.run(
    [MaximumLikelihoodAmplitudeEstimationRequest(problem, (0, 1, 2, 4), 1000)]
)[0]
print(result.estimation, result.asymptotic_interval)
```

快速振幅估计必须使用满足小振幅承诺的问题。例如：

```python
from cqlib_algorithms import (
    FasterAmplitudeEstimation,
    FasterAmplitudeEstimationRequest,
)

theta = 0.1
preparation = Circuit(1)
preparation.ry(0, 2 * theta)
grover = Circuit(1)
grover.ry(0, 4 * theta)
problem = AmplitudeEstimationProblem(
    preparation,
    grover,
    (0,),
    lambda bits: bits == "1",
)

algorithm = FasterAmplitudeEstimation(StatevectorSampler())
result = algorithm.run(
    [FasterAmplitudeEstimationRequest(problem, alpha=0.05, max_iterations=5)]
)[0]
print(result.estimation, result.confidence_interval)
```

若目标振幅大于 `0.25`，调用方应先构造数学上一致的缩放后问题，并在请求中设置
`amplitude_rescaling_factor=0.25`；不能只修改该参数而继续传入原来的 <img src="../../assets/formulas/inline/inline-559aead08264d579.svg" alt="A" style="vertical-align: -0.000pt;"> 和 <img src="../../assets/formulas/inline/inline-4ae81572f06e1b88.svg" alt="Q" style="vertical-align: -3.000pt;">。

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

