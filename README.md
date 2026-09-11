# cqlib-algorithms

面向 Cqlib 的量子算法组件库。支持 Python 3.10+，使用 uv 管理依赖、pytest 测试，
采用 Apache-2.0 开源协议。当前提供第一批公共组件协议及其请求、结果数据结构，
尚未提供具体算法实现。

## 目录

```text
src/cqlib_algorithms/
├── __init__.py
├── py.typed
├── protocols/
│   ├── state_preparation.py
│   ├── ansatz.py
│   ├── sampler.py
│   ├── estimator.py
│   └── gradient.py
└── algorithms/
tests/protocols/
docs/development.md
AGENTS.md
pyproject.toml
LICENSE.txt
```

公共协议及其请求、结果类型集中放在 `protocols/`；具体组件按实际功能增加模块或
子包，完整算法放在 `algorithms/`。不预建没有实现的分类目录。

## 公共组件协议

- `StatePreparation`：构建状态制备或数据编码线路。
- `Ansatz`：构建参数化变分线路。
- `Sampler`：同步、批量的线路采样协议。
- `Estimator`：同步、批量的期望值估计协议。
- `Gradient`：基于 Estimator 期望值目标的同步、批量梯度协议。

协议采用结构类型，具体组件无须继承协议即可兼容。`StatePreparation` 与 `Ansatz`
目前具有相同的方法结构，二者的区别是组合位置和参数语义，而不是运行时类型判断。
请求对象只浅冻结字段；其中的 Cqlib 线路和可观测量仍然可变，执行实现必须在运行前
重新核验请求约束，调用方不得在请求生命周期内修改这些 Core 对象。

本库的 `Estimator` 是批量估计协议，与 `cqlib.error_mitigation.Estimator` 的
单请求回调类型处于不同命名空间；同一模块需要同时使用时，应为后者设置明确别名。

```python
from cqlib_algorithms import (
    Ansatz,
    Estimator,
    Gradient,
    Sampler,
    StatePreparation,
)
```

## 开发

运行依赖为 `cqlib>=1.4.0b1,<1.5`。首版协议已针对 Cqlib `develop`
分支提交 `c3abe91a4adf804b6fb3304fab6ec3dc24a4ad85` 的 Python 接口核验。
SDK 尚未发布期间由开发者手动安装。
使用已有 `.venv` 时跳过创建环境的步骤：

```bash
uv venv --python 3.10
source .venv/bin/activate
```

将 Cqlib 安装到该环境后：

```bash
uv pip install --group dev
uv pip install --no-deps -e .
pytest
mypy src/cqlib_algorithms
python -m build
```

Windows PowerShell 使用 `.venv/Scripts/Activate.ps1` 激活环境。
不激活环境时，使用 `uv run --no-sync pytest`、`uv run --no-sync mypy` 和
`uv run --no-sync python -m build`。
构建产物位于 `dist/`；隔离构建需要获取 Hatchling，但不需要安装运行时 SDK。

SDK 尚未发布期间不执行 `uv sync`。依赖可从包源获取后，再生成并提交 `uv.lock`。

开发规范见 [docs/development.md](docs/development.md)，多 Agent 开发流程见
[docs/cqlib-agent-development-workflow.md](docs/cqlib-agent-development-workflow.md)。许可证见
[LICENSE.txt](LICENSE.txt)。
