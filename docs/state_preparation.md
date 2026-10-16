# 状态制备与数据编码组件

本文档说明 `cqlib_algorithms.operators.state_preparation` 的首版公共接口。
所有组件显式继承 `StatePreparation` 抽象基类；稳定公共面统一为：

- `validate() -> None`：核验构造配置；
- `build_circuit(prefix: str) -> Circuit`：每次返回独立线路；
- `num_parameters() -> int`：返回逻辑输入或特征参数槽数量；
- `num_qubits() -> int`：返回线路量子比特数。

## 组件一览

| 组件 | 构造函数 | 参数槽 | 实现与主要边界 |
| --- | --- | ---: | --- |
| `BasisStateInitialization` | `(num_qubits, basis_index)` | 0 | 准备指定计算基态；索引必须能由给定宽度表示 |
| `UniformSuperposition` | `(num_qubits)` | 0 | 对每个量子比特施加 H，得到等幅正相位叠加态 |
| `ArbitraryStatePreparation` | `(amplitudes)` | 0 | 输入长度至少为 2 且为 2 的幂；范数须在 `1e-12` 内为 1 |
| `ProductStatePreparation` | `(qubit_states)` | 0 | 每项为一个已归一化的二元复幅度，且第 `q` 项作用于量子比特 `q` |
| `GHZStatePreparation` | `(num_qubits)` | 0 | 使用 H(0) 和相邻 CX 链准备 GHZ 态 |
| `AmplitudeEncoding` | `(features)` | 0 | 复制并自动归一化非零稠密复数特征向量 |
| `AngleEncoding` | `(num_qubits, rotation_gate)` | `num_qubits` | 仅接受未绑定的 Core `RX`、`RY`、`RZ` 或 `Phase` |
| `BasisEncoding` | `(bits)` | 0 | 仅接受非空布尔序列，并委托 Core 的 Basis Encoding |
| `ImageAngleEncoding` | `(image_shape, channels, *, value_range=(0.0, 255.0))` | 0 | 每个像素通道标量映射为一个 RY 角度；通道优先、行优先布局 |
| `ImageBasisEncoding` / `BRQI` | `(image_shape, channels)` | 0 | 固定 8 位单通道灰度 BRQI；高、宽均须为 2 的非负整数次幂（含 1） |
| `SparseAmplitudeEncoding` | `(num_qubits, entries)` | 0 | 输入为非空 `(index, amplitude)` 序列；拒绝重复索引和显式零幅度，自动归一化 |
| `UserCircuitPreparation` | `(circuit)` | 活跃符号数 | 保存线路快照并要求 Core 能静态求逆 |
| `FeatureMapPreparation` | `(feature_map)` | 由 Core 决定 | 接受 Core 的 `ZFeatureMap`、`IQPFeatureMap`、`ZZFeatureMap` 或 `PauliFeatureMap` |

`num_qubits` 和所有索引必须是真正的 Python `int`，不接受 `bool` 代替整数。
数值输入必须可转换为有限复数。配置或数值非法时抛出 `ValueError`，输入类型不匹配时
抛出 `TypeError`；有效配置进入 Core 后产生的 `CircuitError` 保留 Core 语义。

## 端序、归一化与相位

Cqlib 使用小端基态索引：量子比特 0 对应索引最低有效位。例如，
`BasisStateInitialization(3, 5)` 准备 `|101>`，其中输入位 `bits[0]` 对应量子比特 0。
幅度向量的第 `i` 项就是计算基态索引 `i` 的复幅度。

`ArbitraryStatePreparation` 不替调用方掩盖明显的归一化错误，只接受相对误差和绝对
误差均为 `1e-12` 范围内的单位向量，然后修正可接受的浮点舍入；
`AmplitudeEncoding` 和 `SparseAmplitudeEncoding` 则自动归一化非零数据。
实现只把精确的复数零视为零，不用阈值删除小幅度。

数值态综合保留相对相位和全局相位。验证线路输出时应比较
`Circuit.to_matrix()[:, 0]` 与目标复幅度向量；Core 的 Statevector 演化不包含线路的
`global_phase`，因此不能单独用于精确全局相位验收。

## 稠密态综合

`ArbitraryStatePreparation` 与 `AmplitudeEncoding` 使用 Möttönen 的均匀受控旋转
方法：先以 RY 树综合幅度绝对值，再以 RZ 树和线路全局相位综合复相位。通常 UCR
通过循环 Gray code 和 Walsh-Hadamard 角变换降为单比特旋转与 CX；当极端动态范围
会使 Walsh 角变换丢失可表示的非零小角度时，使用互斥控制模式的多控旋转回退。
实现不构造稠密 `UnitaryGate`。

算法来源：M. Möttönen 等，
[Transformation of quantum states using uniformly controlled rotations](https://arxiv.org/abs/quant-ph/0407010)。

## 稀疏态综合

`SparseAmplitudeEncoding` v1 采用本项目实现的、可逐步证明的 exact support-set
two-level Givens 综合。调研阶段同时对照了 Gleinig 与 Hoefler 的
[An Efficient Algorithm for Sparse Quantum State Preparation](https://spcl.inf.ethz.ch/Publications/.pdf/quantum_dac.pdf)
（[DOI 10.1109/DAC18074.2021.9586240](https://doi.org/10.1109/DAC18074.2021.9586240)）。
当前实现采用 two-level Givens 路径和下述独立复杂度界：线路从支持集中
绝对值最大的参考基态开始，以 Gray 路径把参考态和目标态映射为相邻基态，再用满足
`phi + lambda = 0 (mod 2*pi)` 的受控 SU(2) Givens 旋转分配一个目标幅度，最后逆向
恢复 Gray 路径。已经分配的其他支持态不属于该二能级子空间，因此不会被后续步骤
改动。

令量子比特数为 `n`、支持大小为 `s`：经典存储为 `O(s+n)`，线路包含 `O(sn)`
个逻辑多控操作及最坏 `O(sn²)` 个 open-control 周边单比特操作。按当前 Core 的
无辅助量子比特多控门分解规则，基础门上界为最坏 `O(sn³)`。数据加载采用显式支持集
线路综合，并保持 `O(s+n)` 经典存储；该过程无需创建长度为 `2**n` 的稠密向量或矩阵。

## Core 适配边界

`BasisEncoding`、`AngleEncoding` 和 `FeatureMapPreparation` 保存输入快照并直接委托
兼容版本的 Cqlib Core 构建线路，不复制 Core 已有线路逻辑。

`UserCircuitPreparation` 将源线路的量子比特按插入位置映射到连续 `0..n-1`，保留
活跃符号和线路全局相位，且 `prefix` 不重命名用户符号。它要求线路可由 Core 静态
求逆，因此拒绝测量、重置、动态控制等非幺正操作。当前 Core 也不能对三量子比特及
以上的矩阵型 `UnitaryGate` 求逆，即使矩阵本身是幺正的；该限制会在构造时明确暴露，
不能用稠密酉门绕过。

## 图像编码

两个图像组件都接受 `(height, width)` 和通道优先的扁平行优先数据。若
`N = height * width`，`ImageAngleEncoding` 将通道 `c` 的第 `p` 个标量放在
量子比特 `q = c * N + p`，并按
`theta = pi * (value - lower) / (upper - lower)` 施加 `RY(theta)`。输入值和范围
必须有限，且由调用方在编码前完成所需的裁剪。每个标量使用一个量子比特和一个局部
旋转，形成直积态编码。

`ImageBasisEncoding`（精确别名 `BRQI`）实现固定 8 位单通道灰度 BRQI。对
`H=2**h`、`W=2**w`，目标态为

```text
1/sqrt(8*H*W) * sum_(l,y,x) |g(y,x,l)> |x> |y> |l>
```

其中 `g(y,x,l)=(pixel[y,x] >> l) & 1`，`l=0` 为最低有效位平面。Cqlib 小端物理
布局固定为：量子比特 0 是值位，其后依次为 `w` 个列位、`h` 个行位和 3 个平面位，
各寄存器内部均最低有效位在前。对应基态索引为
`g + 2*x + 2*W*y + 2*W*H*l`。

线路先对全部地址与平面量子比特施加 H，再对每个值为 1 的 `(x,y,l)` 地址使用完整
开放/闭合控制写入值位，返回未测量的状态制备线路。只接受恰好一个通道，像素必须是真正的 Python
整数且位于 `[0,255]`；高和宽必须分别为 2 的非负整数次幂（含 1）。逻辑量子比特数为
`h+w+4`；经典构造显式枚举 `8*H*W` 个像素位，线路规模和构造时间随输入图像的数据量
增长。两个图像组件都在构造时复制输入，并忽略合法字符串 prefix。

## 使用示例

```python
from cqlib_algorithms.operators.state_preparation import AmplitudeEncoding

preparation = AmplitudeEncoding([1.0, 1.0j, 0.0, 0.0])
preparation.validate()
circuit = preparation.build_circuit("data")

assert preparation.num_qubits() == 2
assert preparation.num_parameters() == 0
assert circuit.num_qubits == 2
```

```python
from cqlib_algorithms.operators.state_preparation import ImageAngleEncoding

image = ImageAngleEncoding((1, 2), [[0.0, 255.0]])
circuit = image.build_circuit("ignored")
assert image.num_qubits() == 2
```

```python
from cqlib_algorithms.operators.state_preparation import BRQI

image = BRQI((2, 2), [[0, 64, 128, 255]])
circuit = image.build_circuit("ignored")
assert image.num_qubits() == 6
assert image.num_parameters() == 0
```

开发环境验证使用 `uv run --no-sync`，避免同步尚未发布的 Cqlib SDK。例如：

```powershell
uv run --no-sync pytest tests/operators/state_preparation -q
```
