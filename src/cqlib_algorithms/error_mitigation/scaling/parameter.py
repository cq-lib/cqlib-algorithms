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

r"""Noise scaling by parameter perturbation.

Parameter-noise scaling does not add gates. It assumes the dominant error
source is a stochastic shift of every rotation angle, drawn once per gate
from a zero-mean Gaussian whose variance grows linearly with the scale
factor:

.. math::

    \theta_i \mapsto \theta_i + \varepsilon_i, \qquad
    \varepsilon_i \sim \mathcal{N}\!\left(0, (\lambda - 1)\, \sigma^2\right)

where :math:`\sigma^2` is the per-gate ``base_variance``. At
:math:`\lambda = 1` the injected noise vanishes and the circuit is returned
unchanged; a device whose noise is parameter dominated therefore sees a
genuinely rescaled error rate without any gate-count overhead.

The mechanism is device-agnostic: the perturbed angles are rebound onto the
same standard gates, so the scaled circuit is executable as-is. Non-parametric
gates, directives, and the measurement tail are copied through untouched;
measurements are stripped first and re-appended last.
"""

from __future__ import annotations

import numpy as np

from cqlib import Circuit, Parameter
from cqlib.circuit import StandardGate, ValueOperation

from .._circuits import rebuild, strip_measurements
from .folding import _normalize_scale_factor, _restore_measurements


def scale_parameters(
    circuit: Circuit,
    scale_factor: float,
    /,
    *,
    base_variance: float,
    seed: int,
) -> Circuit:
    r"""Scale the parameter noise of ``circuit`` to ``scale_factor``.

    Every standard gate parameter is shifted by an independent Gaussian
    draw with standard deviation :math:`\sqrt{(\lambda - 1) \sigma^2}`, where
    :math:`\sigma^2` is ``base_variance``. The draws are made in operation
    order from a single seeded generator, so the result is a deterministic
    function of ``seed``.

    Parameters
    ----------
    circuit
        Circuit to scale. Measurements are stripped and re-appended.
    scale_factor
        Noise scale factor :math:`\lambda \ge 1`. ``1`` injects no noise.
    base_variance
        Base parameter variance :math:`\sigma^2 \ge 0` assumed for every
        gate, i.e. the variance the gate sees at :math:`\lambda = 2`.
    seed
        Seed for the Gaussian draw sequence.

    Returns
    -------
    Circuit
        Circuit with rescaled parameters, carrying the source global phase.

    Raises
    ------
    ValueError
        If ``scale_factor < 1`` or ``base_variance < 0``, if the body holds a
        parametric operation that is not a standard gate (nothing to rebind),
        or if a parameter is symbolic and therefore has no numeric value to
        perturb.

    Notes
    -----
    ``base_variance`` and ``seed`` are required keyword-only arguments so both
    the assumed noise level and the random stream are always explicit.

    References
    ----------
    .. [1] K. Temme, S. Bravyi, and J. M. Gambetta,
       "Error mitigation for short-depth quantum circuits,"
       Physical Review Letters 119, 180509, 2017. arXiv:1612.02058.
    """
    scale = _normalize_scale_factor(scale_factor)
    if base_variance < 0.0:
        raise ValueError(
            f"base_variance must be >= 0 but was {base_variance!r}; a variance "
            "cannot be negative."
        )
    body, measured = strip_measurements(circuit)
    sigma = float(np.sqrt((scale - 1.0) * base_variance))
    rng = np.random.RandomState(seed)
    operations = [_perturb_operation(op, rng, sigma) for op in body.operations]
    return _restore_measurements(rebuild(body, operations), measured)


def _perturb_operation(
    operation: ValueOperation,
    rng: np.random.RandomState,
    sigma: float,
    /,
) -> ValueOperation:
    r"""Shift each parameter of ``operation`` by one Gaussian draw.

    Raises
    ------
    ValueError
        If the operation carries parameters but is not a standard gate, or if
        any of its parameters is a non-constant symbolic expression.
    """
    # Operations without parameters pass through untouched; a parametric
    # operation is rebound onto a fresh standard gate built from its own name,
    # which keeps the gate type and qubit order while replacing the angles.
    if operation.num_params == 0:
        return operation
    if not operation.is_standard:
        raise ValueError(
            f"Operation {operation.name!r} carries parameters but is not a "
            "standard gate, so its parameters cannot be rebound."
        )
    values = [_parameter_to_float(value) for value in operation.params]
    definition = StandardGate.from_name(operation.name)
    perturbed = [value + rng.normal(loc=0.0, scale=sigma) for value in values]
    return ValueOperation.from_standard_gate(
        definition(*perturbed),
        list(operation.qubits),
        label=operation.label,
    )


def _parameter_to_float(value: float | Parameter, /) -> float:
    r"""Return the numeric value of a constant gate parameter.

    Raises
    ------
    ValueError
        If ``value`` is a symbolic expression with free symbols.
    """
    if isinstance(value, Parameter):
        if not value.is_constant():
            raise ValueError(
                f"Parameter {value!s} is symbolic, so it has no numeric value "
                "to perturb."
            )
        return float(value.evaluate({}))
    return float(value)
