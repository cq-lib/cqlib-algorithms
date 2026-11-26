# 文档与 Docstring 规范

本文规定 Python 源文件、公共 API docstring、私有辅助函数说明和代码行注释的统一
格式。新增或修改代码时应遵循本规范；数学实现细节以经过独立验证的代码和测试为准。

## 文件头与文献引用

每个 Python 文件依次包含：版权声明、模块 docstring、`from __future__`（如需）和
导入。版权声明必须保持以下文本，不在其中加入作者、日期、论文或修改记录：

```python
# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
```

模块 docstring 第一段用一句话说明模块提供的能力。只有当公式、算法或边界直接来源
于论文、标准或权威资料时才加入引用；模块整体共享的引用放在模块 docstring，而不
放在版权注释、普通行注释或每个函数中重复。格式统一为：

```python
"""Quantum mutual information of a qubit-state bipartition.

References
----------
[1] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
    Information*, Cambridge University Press, 2010.
"""
```

引用必须能直接支撑当前实现，优先给出作者、题名、期刊或出版社、年份和 DOI/arXiv。
一个模块包含相互独立的多种方法时，可在公共 API 的正文中用 `[1]` 指向模块引用；
不得为了显得完整加入与实现无直接关系的文献。实现限制、尚未支持的扩展和与其他库的
比较属于用户指南，不写入版权声明或参考文献段。

## 公共类、函数和方法

公共 docstring 使用英文和 `Args`、`Returns`、`Raises` 结构，不使用 NumPy 风格的
`Parameters / ----------`。类型信息由签名表达，docstring 说明语义、约束和单位。

公共类说明以下内容：

1. 类代表的数学对象或执行角色；
2. 关键约定，例如量子比特顺序、端序、相位、精确或有限-shot 语义；
3. 构造参数及其约束；
4. 构造时可能抛出的异常；
5. 只有确实影响使用者时才说明资源增长、近似假设或可变性。

```python
class ExampleOperator:
    """Build a parameterized operator on a fixed qubit register.

    Qubit zero is the least-significant basis-state bit. Repeated builds return
    independent circuits and do not mutate the configuration.

    Args:
        num_qubits: Positive number of operator qubits.
        angle: Finite rotation angle in radians.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its valid range.
    """
```

公共函数和非平凡公共方法说明行为、必要数学语义、输入约束、返回语义和调用者可见的
异常。返回 `None` 时省略 `Returns`；只含一句话即可说清的属性和简单委托方法不扩写
模板段落。普通公共 docstring 通常保持 10–25 行，复杂公式或资源边界确有必要时再
扩展。

```python
def build_distribution(values: Sequence[float]) -> tuple[float, ...]:
    """Return a normalized immutable probability distribution.

    Args:
        values: Non-empty finite non-negative weights with positive sum.

    Returns:
        Normalized weights in the original order.

    Raises:
        TypeError: If an entry is not a real number.
        ValueError: If the sequence is empty, non-finite, or has zero sum.
    """
```

不要在多个方法中复制整段类说明、论文背景或实现步骤。构造参数写在类 docstring；
`build_circuit()`、`run()` 等方法只补充该次调用特有的输入、输出和异常。

## 私有辅助函数与命名

私有辅助函数只说明非显然语义，通常使用一句命令式说明；不为显而易见的局部步骤复制
完整 `Args / Returns / Raises`。名称必须反映函数是否转换数据：

- `_validate_*`：只检查已有值，成功时返回 `None`，不修改输入、不复制、不排序、
  不归一化。统一写作 `"""Validate ... ."""` 或
  `"""Raise if ... ."""`。
- `_normalize_*`：检查输入并返回语义等价的规范化新值，例如数值类型转换、尺度归一化
  或标准范围映射；不得原地修改调用方对象。统一写作
  `"""Validate and return ... in normalized form."""`。
- `_canonicalize_*`：检查输入并返回唯一表示，可包含排序、合并重复项、固定相位或稳定
  顺序。统一写作 `"""Validate and return the canonical ... ."""`。
- `_snapshot_*` 或 `_copy_*`：只创建防御性快照或副本，不改变数学语义。统一写作
  `"""Return a defensive ... snapshot."""`。
- `_require_*`：检查调用位置的前置条件，成功时返回 `None`；仅在“要求某能力存在”比
  “验证一个参数”更准确时使用。

禁止 `_validate_*` 返回转换后的值，也禁止调用 `_normalize_*` 或
`_canonicalize_*` 后丢弃其返回值：

```python
def _validate_num_qubits(num_qubits: int) -> None:
    """Validate a positive qubit count."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")


def _normalize_angle(angle: float) -> float:
    """Validate and return an angle normalized to ``[0, 2*pi)``."""
    ...
```

如果辅助函数同时做复制和规范化，应按主要可观察语义命名，并在一句话中明确返回的是
新对象。多个模块复用前必须先统一异常类型、输入边界和返回语义。

## 代码行注释

行注释解释代码本身无法表达的信息，包括：

- 采用某条公式或分解路线的原因；
- 小端顺序、寄存器角色、全局相位和符号约定；
- 为规避底层接口行为而做的兼容处理；
- 数值容差、复杂度上界和提前拒绝规模的原因；
- 看似可以简化但必须保留的算法不变量。

不要写逐行翻译，例如 `# increment counter`、`# build circuit`；不要把参数说明、异常
列表、论文摘要或测试结论放在行注释中。临时审计记录、PR 历史和“以后可能修改”的
说明应进入问题记录或用户指南，而不是生产源码。

## 用户指南与检查

源码 docstring 是接口契约；`docs/user-guide/` 说明如何选择组件、如何调用、如何解释
结果以及当前适用边界。用户指南不复制完整字段列表、私有调用链或测试清单。

Markdown 文档使用 GitHub 原生支持的 LaTeX 数学语法。公式直接维护在对应的文档中，
参数名、类型名和短程序表达式使用行内代码。

行内公式使用美元符号与反引号组合，避免下划线等字符被当作 Markdown 标记。例如：

```markdown
初态为 $`\lvert 0^n\rangle`$。
```

独立公式使用 `math` 围栏代码块，并与相邻段落之间保留空行。例如：

````markdown
```math
Q=A(2\lvert 0^n\rangle\langle 0^n\rvert-I)A^\dagger S_f
```
````

表格中的行内公式使用 `\lvert`、`\rvert`、`\vert` 或 `\Vert` 表示竖线，防止
竖线被识别为列分隔符。较长的推导放在表格外，按需要使用 `aligned` 对齐分行。
公式中的运算符、上下标和宏必须能被 GitHub 的 MathJax 渲染器识别。

修改公式后检查数学含义、行内与独立公式的分隔、表格列数和页面显示；同时确认
代码示例、文档链接和参考文献没有受到排版调整的影响。语法说明见
[GitHub 数学表达式文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。

修改完成后至少检查：

1. 所有 Python 文件保留统一版权声明和模块 docstring；
2. 公共 API 使用 `Args / Returns / Raises`，异常与实现一致；
3. `_validate_*` 均返回 `None`，转换函数名称与返回语义一致；
4. 引用位于模块 docstring，且与实现直接相关；
5. 示例可运行，文档只声明已实现能力；
6. pytest、mypy、Ruff、格式和构建结果如实记录。
