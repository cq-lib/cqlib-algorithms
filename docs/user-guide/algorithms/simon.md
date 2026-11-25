# Simon 算法

本文档说明 `cqlib_algorithms.algorithms.Simon` 的数学约定、公共接口、组件复用、
结果语义与边界。该实现通过公共 `Sampler` 契约执行测量，不绑定具体模拟器或后端。

## 问题与承诺

给定布尔函数

<p align="center">
  <img src="../../assets/formulas/simon-function-domain.svg" alt="F:{0,1}n→{0,1}m.">
</p>

Simon 问题承诺存在一个字符串 `s`，使得

<p align="center">
  <img src="../../assets/formulas/simon-promise.svg" alt="F(x)=F(y)⇔x=y 或 x⊕y=s.">
</p>

当 <img src="../../assets/formulas/inline/inline-579581daa15822bb.svg" alt="s\ne0" style="vertical-align: -3.000pt;"> 时，每个函数值恰好对应一对输入 <img src="../../assets/formulas/inline/inline-2d711642b726b044.svg" alt="x" style="vertical-align: -1.000pt;"> 与 <img src="../../assets/formulas/inline/inline-82b61d37fa4b40ba.svg" alt="x\oplus s" style="vertical-align: -2.000pt;">；当 <img src="../../assets/formulas/inline/inline-b10f271a76545007.svg" alt="s=0" style="vertical-align: -1.000pt;">
时，该条件退化为 <img src="../../assets/formulas/inline/inline-c8288c5ec67de4f6.svg" alt="F(x)=F(y)" style="vertical-align: -4.000pt;"> 当且仅当 <img src="../../assets/formulas/inline/inline-14bf50d80768f3bc.svg" alt="x=y" style="vertical-align: -3.000pt;">，即 <img src="../../assets/formulas/inline/inline-f67ab10ad4e4c531.svg" alt="F" style="vertical-align: -0.000pt;"> 是一一函数。实现同时接受单输出
`BooleanFunction` 和多输出 `BooleanVectorFunction`，不要求 <img src="../../assets/formulas/inline/inline-201a08c10e842b76.svg" alt="m=n" style="vertical-align: -1.000pt;">。
显式真值表只适合规模较小的精确构造；其指数规模的经典输入和线路综合成本不属于
黑盒查询复杂度结论。

## 量子线路

`Simon.build_circuit(function)` 构建 <img src="../../assets/formulas/inline/inline-6a766438d1de2cc9.svg" alt="n+m" style="vertical-align: -2.000pt;"> 量子比特的未测量线路：

```text
输入 |0>^n ── H^n ──┐ U_F ┌── H^n
输出 |0>^m ─────────┘     └──────
```

其中 <img src="../../assets/formulas/inline/inline-393db2f05e282178.svg" alt="U_F" style="vertical-align: -3.000pt;"> 由 `BitFlipOracle` 构建，满足

<p align="center">
  <img src="../../assets/formulas/simon-oracle-action.svg" alt="UF|x⟩|y⟩=|x⟩|y⊕F(x)⟩.">
</p>

算法仅测量输入寄存器。每个测量结果 `z` 满足 GF(2) 方程

<p align="center">
  <img src="../../assets/formulas/simon-linear-equation.svg" alt="z·s=0 mod 2.">
</p>

量子比特 0、布尔变量 0 和结果元组下标 0 都是最低有效位。Sampler 返回的字符串
采用显示顺序，因此算法把最右字符解释为量子比特 0，再转为小端布尔元组。

## 公共接口

```python
from cqlib_algorithms import Simon
from cqlib_algorithms.operators import (
    BitFlipOracle,
    BooleanVectorFunction,
    StatevectorSampler,
)

period = 0b101
rows = []
for value in range(8):
    representative = min(value, value ^ period)
    rows.append(tuple(bool((representative >> bit) & 1) for bit in range(3)))
oracle = BitFlipOracle(BooleanVectorFunction.from_truth_table(rows))
result = Simon(StatevectorSampler()).run(oracle, shots=128)
print(result.period_integer, result.is_complete)  # 5 True
```

`run(oracle, *, shots)` 接收 `BitFlipOracle` 并返回一个结果。兼容实现可继承该类并重写 `build_circuit()`，其线路应计算同一个 `function`，且保留输入在前、输出在后的寄存器布局。
批量请求 `SimonRequest(function, shots)` 中 `shots` 必须为正整数。`run(requests)` 支持批量
执行，空批次返回空列表。

`SimonResult` 保留：

- `num_qubits`：输入寄存器宽度 <img src="../../assets/formulas/inline/inline-1b16b1df538ba12d.svg" alt="n" style="vertical-align: -1.000pt;">；
- `independent_equations`：从观测结果得到的规范 GF(2) 行最简独立方程；
- `nullspace_basis`：方程组零空间的规范基；
- `shots`：执行 shots 数；
- `rank`：独立方程秩；
- `is_complete`：秩是否为 <img src="../../assets/formulas/inline/inline-51aeea8ffa05d262.svg" alt="n-1" style="vertical-align: -1.000pt;"> 或 <img src="../../assets/formulas/inline/inline-1b16b1df538ba12d.svg" alt="n" style="vertical-align: -1.000pt;">，从而唯一确定承诺周期；
- `period` / `period_integer`：秩为 <img src="../../assets/formulas/inline/inline-51aeea8ffa05d262.svg" alt="n-1" style="vertical-align: -1.000pt;"> 时给出唯一非零零空间方向，满秩时
  分别给出全 `False` 元组和整数 `0`。

零结果和重复、线性相关结果不会增加秩。若样本不足，`nullspace_basis` 会如实保留
多个候选方向，`period` 为 `None`。在理想承诺下，满秩表示一一函数并恢复零周期；
噪声或承诺外输入也可能伪造满秩，此时组件仍按形式方程返回零候选，不将其解释为
经过统计验证的结论。组件不自动追加 shots、不估计置信度，也不实现噪声鲁棒解码。

## 承诺与复杂度边界

算法不读取真值表来验证二对一 XOR 承诺。违反承诺的输入仍产生合法线路和形式正确
的 GF(2) 结果，但这些结果不具有 Simon 周期恢复语义。每条构造线路包含一次
`BitFlipOracle`；多个 shots 表示重复执行该线路，不能表述为整个采样任务仅进行一次
物理 Oracle 求值。经典后处理对观测方程执行 GF(2) 消元。

实现依据 Daniel R. Simon, *On the Power of Quantum Computation*, Proceedings of
the 35th Annual Symposium on Foundations of Computer Science, 116--123 (1994),
DOI `10.1109/SFCS.1994.365701`。

## DJ/BV Oracle 构造能力

Deutsch-Jozsa 与 Bernstein-Vazirani 所需的“经典函数到量子 Oracle”能力已经由现有
组件完整覆盖，不另设同义构造器：`BooleanFunction` / `BooleanVectorFunction`
保存经典函数，`PhaseOracle` 构造相位 Oracle，`HadamardPhaseQuery` 组合
<img src="../../assets/formulas/inline/inline-678b057a76c6a024.svg" alt="H^{\otimes n}\rightarrow S_f\rightarrow H^{\otimes n}" style="vertical-align: -5.000pt;">。需要标准 XOR 输出形式时使用 `BitFlipOracle`。Simon 复用
后者，因为算法必须保留函数输出寄存器，而不是把输出按奇偶归约为单一相位。
