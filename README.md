# cqlib-algorithms

面向 Cqlib 的量子算法组件库。支持 Python 3.10+，使用 uv 管理依赖、pytest 测试，
采用 Apache-2.0 开源协议。当前仅提供基础工程骨架。

## 目录

```text
src/cqlib_algorithms/
├── __init__.py
├── py.typed
└── algorithms/
tests/
docs/development.md
AGENTS.md
pyproject.toml
LICENSE.txt
```

组件按功能直接放在 `cqlib_algorithms/` 下，完整算法放在 `algorithms/` 下。
有实际实现时再增加模块或子包。

## 开发

运行依赖为 `cqlib>=0.1.0`，尚未发布期间由开发者手动安装。
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
python -m build
```

Windows PowerShell 使用 `.venv/Scripts/Activate.ps1` 激活环境。
不激活环境时，使用 `uv run --no-sync pytest` 和 `uv run --no-sync python -m build`。
构建产物位于 `dist/`；隔离构建需要获取 Hatchling，但不需要安装运行时 SDK。

SDK 尚未发布期间不执行 `uv sync`。依赖可从包源获取后，再生成并提交 `uv.lock`。

开发规范见 [docs/development.md](docs/development.md)。许可证见 [LICENSE.txt](LICENSE.txt)。
