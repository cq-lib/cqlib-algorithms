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

"""Abstract interface for synchronous state-fidelity estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class StateFidelity(ABC, Generic[RequestT, ResultT]):
    """Define synchronous batched state-fidelity estimation.

    Implementations own their request and result value types. They validate a
    complete input batch before execution and return one result per request in
    the original order. Empty input produces an empty list.
    """

    __slots__ = ()

    @abstractmethod
    def run(self, requests: Sequence[RequestT], /) -> list[ResultT]:
        """Estimate a batch and return results in input order.

        Args:
            requests: Fidelity requests in output order.

        Returns:
            One fidelity result per request.
        """
        raise NotImplementedError
