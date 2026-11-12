# cqlib-algorithms

`cqlib-algorithms` 是构建在 Cqlib Core 之上的量子算法组件库，面向量子算法的快速
构建、组合复用和持续扩展，统一组织从问题表示、量子态与线路构造、测量估计、微分
优化到结果处理的完整计算过程。项目采用“基础算子—复合组件—完整算法”的分层结构，
通过公共协议分离算法流程、线路实现与执行后端，使组件能够独立验证；在接口契约、
线路宽度和物理语义相容时，同类组件可组合复用或灵活替换，并为科学计算、量子机器
学习、误差缓解及领域问题求解提供一致的开发和执行基础。

当前版本仍在持续迭代。已经实现的公共能力及其适用边界以
[能力清单](docs/capabilities.md)为准；参数、返回值和异常以源码 docstring 为准，
具体使用方法见[用户指南](docs/index.md#用户指南)。

## 安装

环境要求以 `pyproject.toml` 为准：Python 3.11 及以上，
`cqlib>=2.0.0b7,<2.1`，`numpy>=2.1.0`，`scipy>=1.14,<2`。当前代码已在
Python 3.11.15、[Cqlib 2.0.0b7](https://pypi.org/project/cqlib/2.0.0b7/)、
NumPy 2.4.6、SciPy 1.17.1 和 Pytest 9.1.1 的组合下完成全量功能测试，并使用
Mypy 1.20.2 完成严格类型检查。

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

`uv pip install -e .` 以可编辑方式安装当前仓库中的 `cqlib-algorithms`，并根据
`pyproject.toml` 从 PyPI 自动安装兼容的 Cqlib、NumPy 和 SciPy；Cqlib 为主流
Windows、Linux 和 macOS 平台提供二进制 wheel。

如需运行测试、类型检查、代码检查和构建，再安装开发工具：

```bash
uv pip install "build>=1.2.2,<2" "mypy==1.20.2" "pytest==9.1.1" "ruff>=0.16.8,<1"
```

量子机器学习和量子化学参考功能使用可选依赖组安装：

```bash
# 量子机器学习：安装 scikit-learn
uv pip install -e ".[machine-learning]"

# 量子化学参考计算：安装 PySCF
uv pip install -e ".[chemistry]"

# 同时安装两个可选依赖组
uv pip install -e ".[chemistry,machine-learning]"
```

运行完整测试套件前，至少需要安装 `machine-learning` 依赖组，否则依赖
scikit-learn 的 QSVR 和量子核训练测试无法执行。若还需覆盖 PySCF 电子结构参考
测试，请同时安装 `chemistry` 依赖组；推荐完整开发环境直接安装上述两个依赖组。

## 快速开始

下面使用 Deutsch–Jozsa 算法在一次批量调用中区分常函数与平衡函数：

```python
from cqlib_algorithms import DeutschJozsa, DeutschJozsaRequest
from cqlib_algorithms.operators.oracle import BooleanFunction
from cqlib_algorithms.operators.sampler import StatevectorSampler

constant = BooleanFunction.from_truth_table([False] * 4)
balanced = BooleanFunction.from_truth_table([False, True, True, False])
algorithm = DeutschJozsa(StatevectorSampler())
results = algorithm.run(
    [
        DeutschJozsaRequest(constant, shots=16),
        DeutschJozsaRequest(balanced, shots=16),
    ]
)

print([(result.is_constant, result.zero_count) for result in results])
# [(True, 16), (False, 0)] on the ideal statevector sampler
```

有限 shots 的输出是测量统计；它不自动提供噪声鲁棒性或统计显著性保证。

## 文档

- [文档首页](docs/index.md)：按学习、使用和开发任务进入文档。
- [快速入门](docs/getting-started.md)：安装、首个算法和组件替换。
- [能力清单](docs/capabilities.md)：当前已经实现的协议、组件和算法。
- [用户指南](docs/user-guide/)：算法与组件的使用方法、结果解释和适用边界。
- [开发规范](docs/development.md)：扩展规则、辅助函数归属、测试和文档要求。

## 开发检查

以下全量测试命令假定已经按照安装章节配置完整开发环境及可选依赖。

```bash
uv run --no-sync pytest
uv run --no-sync mypy src/cqlib_algorithms
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m build
```

贡献前请阅读[开发规范](docs/development.md)。项目采用 Apache-2.0 许可证，见
[LICENSE.txt](LICENSE.txt)。
