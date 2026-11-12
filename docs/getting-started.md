# 快速入门

## 安装

项目要求 Python 3.11 及以上、`cqlib>=2.0.0b7,<2.1` 和 NumPy 2.1 及以上。
当前代码已在 Python 3.11.15、
[Cqlib 2.0.0b7](https://pypi.org/project/cqlib/2.0.0b7/)、NumPy 2.4.6、
SciPy 1.17.1 和 Pytest 9.1.1 的组合下完成全量功能测试，并使用 Mypy 1.20.2
完成严格类型检查。

当前开发代码维护在 `dev` 分支。以下命令假定系统已经安装
[uv](https://docs.astral.sh/uv/getting-started/installation/)。

### Windows PowerShell

```powershell
git clone --branch dev --single-branch https://gitee.com/zdxlz/cqlib-algorithms.git
cd cqlib-algorithms
uv venv --python 3.11
& .\.venv\Scripts\Activate.ps1
uv pip install -e .
```

### Linux/macOS

```bash
git clone --branch dev --single-branch https://gitee.com/zdxlz/cqlib-algorithms.git
cd cqlib-algorithms
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

`uv pip install -e .` 安装当前仓库中的 `cqlib-algorithms`，并根据
`pyproject.toml` 从 PyPI 自动安装兼容的 Cqlib、NumPy 和 SciPy；Cqlib 为主流
Windows、Linux 和 macOS 平台提供二进制 wheel。需要执行仓库测试时，再安装
`pytest==9.1.1`。

量子机器学习功能使用 `uv pip install -e ".[machine-learning]"` 安装
scikit-learn；量子化学参考功能使用 `uv pip install -e ".[chemistry]"` 安装
PySCF。两个可选依赖组可以通过
`uv pip install -e ".[chemistry,machine-learning]"` 同时安装。

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

## 替换 Ansatz

本库的完整算法通过统一组件接口组织计算流程，因此可以根据问题结构和线路资源替换
Ansatz，而无需重新实现算法主体。VQE 通过参数化 Ansatz 制备试探态，再由 Estimator
计算能量、Optimizer 更新参数。
下面求解两量子比特 Heisenberg Hamiltonian
<img src="assets/formulas/inline/inline-3158500ed308fe9d.svg" alt="H=X\otimes X+Y\otimes Y+Z\otimes Z" style="vertical-align: -2.000pt;">。其基态是纠缠态
<img src="assets/formulas/inline/inline-9c26ee1cf8bd0388.svg" alt="\left(|01\rangle-|10\rangle\right)/\sqrt{2}" style="vertical-align: -4.000pt;">，最小本征值为 <img src="assets/formulas/inline/inline-615bdd17c2556f82.svg" alt="-3" style="vertical-align: -1.000pt;">。

```python
from cqlib.qis import Hamiltonian, PauliString

from cqlib_algorithms import VQE
from cqlib_algorithms.operators import (
    COBYLAOptimizer,
    EfficientSU2Ansatz,
    RealAmplitudesAnsatz,
    StatevectorEstimator,
)

hamiltonian = Hamiltonian.from_list(
    [
        (PauliString.from_str("XX"), 1.0),
        (PauliString.from_str("YY"), 1.0),
        (PauliString.from_str("ZZ"), 1.0),
    ]
)


def solve(ansatz):
    circuit = ansatz.build_circuit("theta")
    initial_point = {
        name: 0.1 * (index + 1)
        for index, name in enumerate(circuit.used_symbols)
    }
    algorithm = VQE(
        StatevectorEstimator(),
        ansatz,
        COBYLAOptimizer(maxiter=300),
    )
    return algorithm.compute_minimum_eigenvalue(
        hamiltonian,
        initial_point,
    )


real_result = solve(RealAmplitudesAnsatz(2, reps=1))
su2_result = solve(EfficientSU2Ansatz(2, reps=1))

print(real_result.eigenvalue)
print(su2_result.eigenvalue)
```

两次求解使用相同的 Hamiltonian、Estimator、Optimizer 和 VQE 流程，只替换 Ansatz。
`RealAmplitudesAnsatz` 使用 RY 旋转和 CNOT 纠缠层，本例有 4 个参数；
`EfficientSU2Ansatz` 使用 RY/RZ 旋转和 CNOT 纠缠层，本例有 8 个参数。两条线路都能
表示目标纠缠态，因此优化结果应接近 <img src="assets/formulas/inline/inline-615bdd17c2556f82.svg" alt="-3" style="vertical-align: -1.000pt;">。替换 Ansatz 后必须根据新线路重新生成
初始参数，不能沿用参数数量或名称不匹配的旧参数映射。

下一步可查看[能力清单](capabilities.md)，再按任务进入相应的算法或组件指南。
