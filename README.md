# cqlib-algorithms

`cqlib-algorithms` 是构建在 Cqlib Core 之上的量子算法与可复用组件库。项目通过
`StatePreparation`、`Ansatz`、`Sampler`、`Estimator`、`Gradient` 和
`Optimizer` 六类公共协议解耦算法结构与执行后端，使状态制备、变分线路、测量、
梯度和经典优化组件能够独立替换与组合。

当前版本处于早期开发阶段。已实现能力及其稳定边界以
[能力清单](docs/capabilities.md)为准；公共类和方法的参数、返回值与异常以源码
docstring 为准。

## 安装

环境要求以 `pyproject.toml` 为准：Python 3.10 及以上，
`cqlib>=2.0.0b5,<2.1`，`numpy>=2.1.0`。

```bash
uv venv --python 3.10
source .venv/bin/activate  # Windows PowerShell: .venv/Scripts/Activate.ps1
uv pip install --no-deps -e .
```

Cqlib Core 2.0 beta 尚未正式发布时，应先在同一环境中安装受支持版本；不要通过
`uv sync` 将未发布依赖替换成其他来源。

## 快速开始

下面使用 Deutsch–Jozsa 算法区分常函数与平衡函数：

```python
from cqlib_algorithms import DeutschJozsa, DeutschJozsaRequest
from cqlib_algorithms.operators.oracle import BooleanFunction
from cqlib_algorithms.operators.sampler import StatevectorSampler

function = BooleanFunction.from_truth_table([False] * 4)
algorithm = DeutschJozsa(StatevectorSampler())
result = algorithm.run([DeutschJozsaRequest(function, shots=16)])[0]

print(result.is_constant)  # True
```

有限 shots 的输出是测量统计；它不自动提供噪声鲁棒性或统计显著性保证。

## 文档

- [文档首页](docs/index.md)：按学习、使用和开发任务进入文档。
- [快速入门](docs/getting-started.md)：安装、首个算法和组件替换。
- [能力清单](docs/capabilities.md)：当前已经实现的协议、组件和算法。
- [用户指南](docs/user-guide/)：算法与组件的使用方法、结果解释和适用边界。
- [开发规范](docs/development.md)：扩展规则、辅助函数归属、测试和文档要求。
- [Agent 协作流程](docs/agent-workflow.md)：多 Agent 任务的职责与交付要求。

## 开发检查

```bash
uv run --no-sync pytest
uv run --no-sync mypy src/cqlib_algorithms
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m build
```

贡献前请阅读[开发规范](docs/development.md)。项目采用 Apache-2.0 许可证，见
[LICENSE.txt](LICENSE.txt)。
