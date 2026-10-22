# cqlib-algorithms 文档

本项目提供可组合的量子算法、算子组件和统一执行协议。公共接口的精确定义以源码
docstring 为准；本目录负责说明当前能力、使用方法和开发规则。

## 开始使用

- [快速入门](getting-started.md)：安装并运行第一个算法。
- [能力清单](capabilities.md)：确认某项协议、组件或算法是否已经实现。

## 用户指南

### 算法

- [振幅估计](user-guide/algorithms/amplitude-estimation.md)
- [相位估计](user-guide/algorithms/phase-estimation.md)
- [VQE](user-guide/algorithms/vqe.md)
- [QAOA](user-guide/algorithms/qaoa.md)
- [Deutsch–Jozsa](user-guide/algorithms/deutsch-jozsa.md)
- [Bernstein–Vazirani](user-guide/algorithms/bernstein-vazirani.md)
- [Simon](user-guide/algorithms/simon.md)
- [CHSH](user-guide/algorithms/chsh.md)
- [BB84](user-guide/algorithms/bb84.md)
- [量子隐形传态](user-guide/algorithms/quantum-teleportation.md)
- [超密编码](user-guide/algorithms/superdense-coding.md)

### 组件

- [状态制备与数据编码](user-guide/components/state-preparation.md)
- [Ansatz 与 QAOA 演化层](user-guide/components/ansatz.md)
- [Mixer](user-guide/components/mixer.md)
- [Oracle 与反射](user-guide/components/oracle.md)
- [线路组合](user-guide/components/circuit-composition.md)
- [测量规划](user-guide/components/measurement.md)
- [执行组件](user-guide/components/execution.md)
- [微分、量子几何与优化](user-guide/components/differentiation.md)
- [线路态保真度](user-guide/components/state-fidelity.md)

## 开发者

- [开发规范](development.md)
- [Agent 协作流程](agent-workflow.md)
