# 线路态保真度

`ComputeUncomputeFidelity` 使用公共 `Sampler` 协议，估计两条 Cqlib 线路所制备
纯态之间的保真度。构造参数、返回类型与异常以源码 docstring 为准。

## 数学定义

设

```text
|psi_a> = U_a |0...0>
|psi_b> = U_b |0...0>，
```

则纯态保真度为

```text
F(|psi_a>, |psi_b>) = |<psi_a|psi_b>|^2
                     = |<0...0| U_a† U_b |0...0>|^2。
```

组件构建 `U_a† U_b` 的 compute-uncompute 线路，并对全部量子比特采样。全零比特串
的频率即保真度估计值 `p`。当 `shots > 1` 时，返回的估计方差为

```text
p * (1 - p) / (shots - 1)。
```

单次采样无法估计统计方差，因此 `variance` 为 `None`。

## 请求与结果

- `CircuitFidelityRequest`：保存两条制备线路、正整数 shots，以及两组独立的参数绑定；
- `CircuitFidelityResult`：返回 `[0, 1]` 内的保真度和可用时的估计方差；
- `ComputeUncomputeFidelity`：接收一个 `Sampler`，批量执行保真度请求。

两条线路必须具有相同的正宽度，并且是可完整绑定、可分解、可求逆的静态酉线路。
测量、重置、动态控制和不可求逆操作会在采样前被拒绝。参数绑定必须恰好覆盖线路当前
使用的全部符号，绑定值必须是有限实数。

两条线路按各自的量子比特插入位置对应，而不是按量子比特数值标识对应，因此支持
非连续且彼此不同的源量子比特标识。调用不会修改输入线路。

## 使用示例

```python
from cqlib import Circuit
from cqlib_algorithms.operators import (
    CircuitFidelityRequest,
    ComputeUncomputeFidelity,
    StatevectorSampler,
)

first = Circuit(1)
first.h(0)

same_state = Circuit(1)
same_state.h(0)
same_state.set_global_phase(0.4)

request = CircuitFidelityRequest(first, same_state, shots=256)
result = ComputeUncomputeFidelity(StatevectorSampler()).run([request])[0]

assert result.fidelity == 1.0
assert result.variance == 0.0
```

## 适用范围

该组件面向由线路制备的纯态，不计算密度矩阵的 Uhlmann 保真度、子系统保真度或混合态
SWAP 测试，也不执行测量误差缓解。数值态矢量与密度矩阵的直接保真度计算属于
Cqlib Core 的能力范围。

compute-uncompute 构造可用于量子特征映射核。理论参考：Havlíček et al.,
*Supervised learning with quantum-enhanced feature spaces*, Nature 567,
209–212 (2019), [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)。
