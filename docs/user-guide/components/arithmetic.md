# 量子算术组件

`cqlib_algorithms.operators.arithmetic` 提供可嵌入算法线路的可逆算术和函数旋转组件。
组件只构建线路，不负责初始化输入寄存器、执行后端任务或解释应用结果。除各类明确说明
的辅助位外，输入寄存器按小端整数编码：局部量子比特 0 是最低有效位。

## 公共组件

| 类别 | 组件 | 语义 |
| --- | --- | --- |
| 加法 | `HalfAdder`、`FullAdder`、`ModularAdder` | 两个等宽量子整数的半加、带输入进位的全加，以及模 <img src="../../assets/formulas/inline/inline-e8a3fed33f2f1fe0.svg" alt="2^n" style="vertical-align: -0.000pt;"> 加法 |
| 常数加法 | `DraperConstantAdder` | 将寄存器值原地更新为 <img src="../../assets/formulas/inline/inline-e8ff3692f34b2f6c.svg" alt="(b+\mathtt{value})\bmod 2^n" style="vertical-align: -4.000pt;"> |
| 比较 | `IntegerComparator` | 将 <img src="../../assets/formulas/inline/inline-d336d8cb50f20858.svg" alt="x\geq\mathtt{value}" style="vertical-align: -2.000pt;"> 或 <img src="../../assets/formulas/inline/inline-940e6749b7048589.svg" alt="x&lt;\mathtt{value}" style="vertical-align: -1.000pt;"> 写入标志位 |
| 乘法 | `Multiplier` | 保留两个输入寄存器，并把乘积写入 <img src="../../assets/formulas/inline/inline-8438925a75f710ec.svg" alt="2n" style="vertical-align: -1.000pt;"> 位结果寄存器 |
| 加权和 | `WeightedSum` | 把多个量子整数的非负整数加权和写入独立结果寄存器 |
| 二次型 | `QuadraticForm`、`build_quadratic_form` | 计算 <img src="../../assets/formulas/inline/inline-a438564559e9eead.svg" alt="x^\mathsf{T}Ax+b^\mathsf{T}x+c\pmod{2^m}" style="vertical-align: -4.000pt;"> |
| 均匀受控旋转 | `UCRX`、`UCRY`、`UCRZ` | 按控制寄存器基态选择目标量子比特的 X/Y/Z 旋转角 |
| 函数与倒数旋转 | `ArbitraryFunction`、`ExactReciprocal` | 根据显式函数表编码角度或振幅；为 HHL 相位寄存器编码倒数振幅 |
| 线性与多项式旋转 | `LinearAmplitudeRotation`、`LinearPauliRotation`、`PolynomialPauliRotation` | 将线性函数编码为目标振幅，或将线性/多项式编码为 Pauli 旋转角 |
| 分段函数旋转 | `PiecewiseLinearPauliRotation`、`PiecewisePolynomialPauliRotation`、`PiecewiseChebyshev` | 通过比较器和分段多项式近似构造函数控制旋转 |

## 综合策略与寄存器

- `HalfAdder` 和 `ModularAdder` 支持 ripple-carry 与 Draper QFT 综合；`FullAdder`
  当前只支持 ripple-carry。半加器额外输出进位，模加器只保留低 `n` 位。
- `Multiplier` 支持 HRS cumulative 与 Draper QFT 综合。前者使用一个辅助位，后者
  可以设置 QFT 近似度。
- `WeightedSum` 支持 QFT 和 carry 两种综合；权重必须是非负整数，结果寄存器宽度由
  可达最大和确定。
- `IntegerComparator` 的输出标志位通过 XOR 写入，调用方应按组件合同初始化并管理
  标志位和辅助位。
- 函数旋转类均把控制寄存器解释为无符号小端整数。角度编码与振幅编码不是同一合同：
  振幅模式会把函数值转换为 <img src="../../assets/formulas/inline/inline-9c4cc7cf87d630fa.svg" alt="2\arcsin(f(x))" style="vertical-align: -4.000pt;">，并要求值位于 <img src="../../assets/formulas/inline/inline-471610700dbccc74.svg" alt="[-1,1]" style="vertical-align: -4.000pt;">。

## 使用示例

```python
from cqlib_algorithms.operators.arithmetic import (
    DraperConstantAdder,
    IntegerComparator,
)

increment = DraperConstantAdder(3, 2).build_circuit()
at_least_five = IntegerComparator(3, 5, relation="geq").build_circuit()

print(increment.num_qubits)
print(at_least_five.num_qubits)
```

`build_circuit()` 每次返回新的线路，可以将其组合到更大的算法线路中。组合时必须按照
各组件公开的 `*_register_indices`、`flag_index`、`helper_index` 等属性映射局部量子比特，
不能仅根据总宽度猜测寄存器顺序。

## 选择建议与限制

- 小规模精确线路或需要直接读出计算基结果时，可优先使用 ripple/carry 综合；需要
  减少辅助位且可以接受 QFT 旋转结构时，可选择 Draper QFT 路径。
- `LinearAmplitudeRotation` 使用截断的 `arcsin` 多项式，公开
  `amplitude_error_bound`；需要逐控制态精确振幅时，应使用
  `ArbitraryFunction(mode="amplitude")`。
- 高次多项式和分段多项式会增加多控旋转与比较器数量；`ArbitraryFunction` 的显式表
  长度为 <img src="../../assets/formulas/inline/inline-e8a3fed33f2f1fe0.svg" alt="2^n" style="vertical-align: -0.000pt;">。这些入口适合明确的小至中等规模函数编码，不隐含高效的通用函数
  求值器。
- QFT 近似度会改变算术线路的数值精度。调用方应针对目标输入范围验证近似误差，而
  不能只依据线路深度选择参数。
