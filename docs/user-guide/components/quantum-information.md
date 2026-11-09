# 量子信息度量

量子互信息衡量一个二分量子态的两个子系统之间包含的总相关性，包括经典相关和量子
相关。对状态 $\rho_{AB}$，本库采用

$$
I(A:B)=S(\rho_A)+S(\rho_B)-S(\rho_{AB}).
$$

其中 $S$ 是 von Neumann 熵，$\rho_A$ 和 $\rho_B$ 由偏迹得到。默认以 2 为对数底，
结果单位为 bit。

## 计算互信息

下面制备 Bell 态，并计算量子比特 0 与其补系统之间的互信息：

```python
from cqlib import Circuit
from cqlib.qis.state import Statevector

from cqlib_algorithms.operators.information import mutual_information

circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
state = Statevector.from_circuit(circuit)

information = mutual_information(state, [0])
print(information)  # approximately 2.0
```

`subsystem_a` 使用 Cqlib 的量子比特位置，量子比特 0 是计算基索引的最低有效位；未
列出的量子比特自动组成子系统 B。子系统必须是非空真子集，列表中的顺序不改变结果。
函数也接受 `DensityMatrix`，因此可以处理经典相关混态和一般混态。需要 nat 或其他
单位时，可设置大于 1 的 `base`。

## 结果解释与边界

- 乘积态的互信息为 0；互信息大于 0 表示两个子系统存在相关性，但不能单独据此判定
  纠缠。
- Bell 纯态的两个单量子比特子系统在以 2 为底时互信息为 2；经典等概率相关态的互
  信息为 1。
- 这是基于完整量子态的精确经典后处理，不是从有限-shot 测量直接估计互信息的协议。
- `Statevector` 输入会先转换成密度矩阵。密度矩阵占用 $O(4^n)$ 空间，因此该接口
  适合能够在内存中显式表示完整状态的小规模精确计算，不适合直接处理大规模硬件态。

实现复用 Cqlib Core 的偏迹和 von Neumann 熵，并只在浮点舍入范围内把极小负结果
归零。输入态不会被修改。

## 理论参考

[1] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
Information*, 10th Anniversary Edition, Cambridge University Press, 2010.
[DOI: 10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)。
