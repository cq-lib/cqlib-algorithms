# 快速入门

## 安装

项目要求 Python 3.10 及以上、`cqlib>=2.0.0b5,<2.1` 和 NumPy 2.1 及以上。

```bash
uv venv --python 3.10
source .venv/bin/activate  # Windows PowerShell: .venv/Scripts/Activate.ps1
uv pip install --no-deps -e .
```

## 运行第一个算法

Deutsch–Jozsa 算法在“函数保证为常函数或平衡函数”的前提下，用一次相位 Oracle
查询完成判别：

```python
from cqlib_algorithms import DeutschJozsa, DeutschJozsaRequest
from cqlib_algorithms.operators.oracle import BooleanFunction
from cqlib_algorithms.operators.sampler import StatevectorSampler

balanced = BooleanFunction.from_truth_table([False, True, True, False])
algorithm = DeutschJozsa(StatevectorSampler())
result = algorithm.run([DeutschJozsaRequest(balanced, shots=32)])[0]

print(result.is_constant)  # False
print(result.zero_count)   # 0 on the ideal statevector sampler
```

算法接受请求序列并返回等长结果序列。这里 `zero_count` 是全零测量结果的次数；在
理想承诺下，常函数产生全零结果，平衡函数不会产生全零结果。

## 替换执行组件

完整算法依赖 `Sampler` 或 `Estimator` 协议，而不依赖某个具体后端。例如 CHSH
可以在精确期望值和有限-shot 估计之间切换：

```python
from cqlib_algorithms import CHSH, CHSHRequest
from cqlib_algorithms.operators.estimator import SamplingEstimator
from cqlib_algorithms.operators.sampler import StatevectorSampler

algorithm = CHSH(SamplingEstimator(StatevectorSampler()))
result = algorithm.run([CHSHRequest(shots=4096)])[0]

print(result.chsh_value)
print(result.violates_classical_bound)
```

采样结果会随机波动；`violates_classical_bound` 是对观测值与经典界 `2` 的比较，
不是自动完成的统计显著性检验。

下一步可查看[能力清单](capabilities.md)，再按任务进入相应的算法或组件指南。
