# 经典优化器组件

`cqlib_algorithms.operators.optimizer` 提供面向 Estimator 期望值目标的三个
无第三方优化依赖组件：

- `GradientDescent`：固定学习率的一阶梯度下降；
- `Adam`：带一阶、二阶矩估计和偏差修正的自适应优化；
- `SPSAOptimizer`：每个活跃请求、每轮仅执行两个目标函数评估的同时扰动优化。

SciPy 已提供的 COBYLA、L-BFGS-B 和 Nelder-Mead 不在本库重复实现，也不因此
增加 SciPy 依赖。

## 公共协议

`OptimizerRequest(estimator_request, parameter_names)` 把一个已完整绑定的
`EstimatorRequest` 作为初始目标点。`parameter_names` 是有序、唯一的训练参数子集；
未列出的绑定在整个优化过程中保持不变。空元组表示只执行一次最终目标评估。

`Optimizer.run(estimator, requests)` 保持请求顺序，返回 `OptimizerResult`：

- `optimal_parameters` 是最终迭代点，不是运行过程中见到的 best-so-far 点；
- `optimal_value` 来自优化结束后对最终点的一次独立 Estimator 评估；
- `iterations` 只计算已经实际应用的更新；
- `status` 为 `STEP_TOLERANCE`、`MAX_ITERATIONS` 或 `NO_PARAMETERS`；
- `converged` 仅对 `STEP_TOLERANCE` 和 `NO_PARAMETERS` 返回 `True`。

三个优化器都先重新校验整批请求，再进行任何梯度或目标评估。空请求批次直接返回
空列表且不调用依赖。`maxiter=0` 对非空训练参数返回初始点和
`MAX_ITERATIONS`；空参数请求返回 `NO_PARAMETERS`。只要请求批次非空，所有最终点
会合并为一次 Estimator 调用。Estimator 或 Gradient 返回数量不一致时显式报错。

## GradientDescent

`GradientDescent(gradient, learning_rate=0.1, *, maxiter=100,
tolerance=None)` 仅依赖公共 `Gradient` 抽象接口。更新为

`θ_(k+1) = θ_k - learning_rate · g_k`。

每轮把所有仍活跃的请求合并为一次 `gradient.run`。更新应用后计算步长的 L2 范数；
当 `||Δθ||₂ <= tolerance` 时该请求独立停止并计入本次迭代。`tolerance=None`
完全关闭步长早停。

## Adam

`Adam` 使用 Kingma 与 Ba（arXiv:1412.6980）的标准偏差修正形式：

`m_k = β₁m_(k-1) + (1-β₁)g_k`

`v_k = β₂v_(k-1) + (1-β₂)g_k²`

`θ_(k+1) = θ_k - learning_rate · m̂_k / (sqrt(v̂_k) + epsilon)`。

默认参数为 `learning_rate=0.001`、`beta1=0.9`、`beta2=0.999`、
`epsilon=1e-8`、`maxiter=100`、`tolerance=None`。每个请求拥有独立的 `m/v`
状态，每次 `run` 都从零状态重新开始。批处理和早停语义与 GradientDescent 相同。

## SPSAOptimizer

SPSA 采用 Spall（1992）的双边 Rademacher 同时扰动。请求局部的零基迭代 `k`
使用

`a_k = learning_rate / (stability_constant + k + 1)^alpha`

`c_k = perturbation / (k + 1)^gamma`

并抽取 `Δ_k ∈ {−1,+1}^d`：

`ĝ_(k,i) = [E(θ_k+c_kΔ_k)-E(θ_k-c_kΔ_k)]Δ_(k,i)/(2c_k)`，

`θ_(k+1) = θ_k - a_k ĝ_k`。

默认 `alpha=0.602`、`gamma=0.101`、`stability_constant=0`、`maxiter=100`。
每轮所有活跃请求的正负扰动点合并为一次 Estimator 调用，因此每个请求的评估数固定
为二，与参数维数无关。整数 seed 可复现；传入 NumPy `Generator` 时共享随机流；
同一优化器实例的随机状态跨 `run` 调用继续推进。

## 参数边界

- learning rate、perturbation、Adam epsilon、alpha 和 gamma 必须有限且大于零；
- `beta1`、`beta2` 必须有限且位于 `[0,1)`；
- `stability_constant` 必须有限且非负；
- `maxiter` 必须是非布尔整数且非负；
- tolerance 可为 `None`，否则必须有限且非负；
- 所有参数和最终目标值必须为有限实数，布尔值不按数值接受。

## 使用示例

```python
from cqlib import Circuit, Parameter
from cqlib.qis import PauliString

from cqlib_algorithms.operators.gradient import ParameterShiftGradient
from cqlib_algorithms.operators.optimizer import Adam, GradientDescent, SPSAOptimizer
from cqlib_algorithms.protocols import Estimator, EstimatorRequest, OptimizerRequest

theta = Parameter("theta")
circuit = Circuit(1)
circuit.ry(0, theta)
request = OptimizerRequest(
    EstimatorRequest(circuit, PauliString.from_str("Z"), {"theta": 0.4}),
    ("theta",),
)


def optimize(estimator: Estimator):
    gradient = ParameterShiftGradient()
    gd = GradientDescent(gradient, learning_rate=0.1, maxiter=50)
    adam = Adam(gradient, maxiter=50)
    spsa = SPSAOptimizer(0.1, 0.05, maxiter=50, seed=7)
    return (
        gd.run(estimator, [request])[0],
        adam.run(estimator, [request])[0],
        spsa.run(estimator, [request])[0],
    )
```

Estimator 是否支持精确或 shots 模式由具体执行实现决定；优化器保持原请求的
`shots` 配置，不定义平台提交、回调、历史记录或函数评估次数接口。

## 理论依据

- Kingma and Ba, *Adam: A Method for Stochastic Optimization*,
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)：Adam 矩估计与偏差修正。
- Spall, *Multivariate stochastic approximation using a simultaneous
  perturbation gradient approximation*, IEEE Transactions on Automatic
  Control 37(3), 1992：SPSA 双边同时扰动梯度近似及增益序列。

