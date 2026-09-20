# 采样器实现

## StatevectorSampler

`StatevectorSampler` 是现有 `Sampler` 抽象基类的本地、同步、批量实现。从
`cqlib_algorithms.operators.sampler` 导入；本阶段不改动共享的包顶层导出。

```python
from cqlib import Circuit, Qubit
from cqlib_algorithms.operators.sampler import StatevectorSampler
from cqlib_algorithms.protocols.sampler import SamplerRequest

circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
request = SamplerRequest(circuit, (Qubit(0), Qubit(1)), 1000)
result = StatevectorSampler().run([request])[0]
print(result.counts, result.probabilities)
```

输入要求是 `SamplerRequest` 序列；空序列返回空列表。每项线路必须已绑定全部活跃符号，
且只能包含静态纯酉操作。Barrier 与 Delay 可保留，它们不改变理想态；测量、重置、
经典控制和其他非酉操作会被拒绝。运行前重新验证每项请求。输出按请求顺序返回
`cqlib.device.ExecutionResult`，状态为成功且终止，`shots` 等于请求值，`qubits` 保持
请求的测量顺序。`counts` 是有限次抽样得到的整数频数，`probabilities` 是其归一化频率，
不是态矢量的解析概率，也不是拟概率分布。Core 当前的 `Statevector.sample` 不提供种子
参数，因此随机频数不保证可复现；计算基确定态的结果可精确验证。

量子比特映射遵从源线路的插入顺序。运行时在独立线路上将原始 ID 映射到连续
`0..n-1`，演化态矢量后才在这份私有线路上生成测量 receipt。输出的
`ExecutionResult.qubits` 再恢复为请求中的原始 `Qubit` ID。测量结果的小端位串约定：
请求中的第一个量子比特是最低有效测量位，位串最右字符对应它；例如请求顺序为
`(Qubit(2), Qubit(7))`、仅 `Qubit(7)` 为 1 时，计数键为 `"10"`。

本组件覆盖态矢量有限-shot 采样及既有 `run` 的批处理。Core `ExecutionResult` 已提供
`counts` 和频率 `probabilities`，故不另造统计结果容器。测量校准、拟概率和误差缓解
需要校准数据与另行明确的契约，不属于本组件；这里不会伪造缓解能力。

## BackendSampler

`BackendSampler` 将冻结的 `Sampler` 批处理契约连接到外部同步后端适配器。组件会逐项
重新验证请求，复制源线路，再按照 `request.qubits` 的顺序追加末端 Core 测量。适配器
同时收到私有线路、该次目标测量的 `Measurement` receipt 和 shots，因此即使源线路已有
其他动态操作，也无需猜测哪个经典值是目标读出；适配器内部的编译或修改不会污染源线路。

```python
from cqlib import Circuit, Qubit
from cqlib.circuit import Measurement
from cqlib.device import ExecutionResult
from cqlib_algorithms.operators.sampler import BackendSampler
from cqlib_algorithms.protocols.sampler import SamplerRequest


def execute(
    circuit: Circuit,
    measurement: Measurement,
    shots: int,
) -> ExecutionResult:
    # 示例适配器返回全 0 频数；真实适配器在这里同步执行私有线路，
    # 并读取指定的 measurement。
    qubits = list(measurement.qubits)
    return ExecutionResult.from_counts(
        "demo-task",
        qubits,
        shots,
        len(qubits),
        {"0" * len(qubits): shots},
        "demo-backend",
    )


sampler = BackendSampler(execute)
circuit = Circuit(1)
request = SamplerRequest(circuit, (Qubit(0),), 100)
result = sampler.run([request])[0]
```

适配器每次必须同步返回终态 `ExecutionResult`；`run` 仍按输入顺序提供批处理语义。
成功结果的 shots、测量宽度和 qubit 顺序会再次核验；失败或取消的终态结果原样保留，
由上层决定如何处理。Core 允许过滤或后选择后的 counts 总数小于 shots，所以该适配层
不会把这种结果误判为无效；需要原始完整频数的上层组件必须自行收紧约束。认证、异步
任务提交、轮询、取消、远程原生批任务和网络传输仍属于平台适配层，不进入算法库，也
不在这里虚构一个 Core 尚未提供的统一后端对象。
