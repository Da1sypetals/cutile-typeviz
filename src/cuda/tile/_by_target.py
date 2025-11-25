# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from typing import TypeVar, Generic


class _UnspecifiedType():
    def __repr__(self):
        return "UNSPECIFIED"


UNSPECIFIED = _UnspecifiedType()

T = TypeVar("T")


class ByTarget(Generic[T]):
    """
    Type used to specify a value that depends on the target GPU architecture.

    Args:
        default: The fallback value to use when the target GPU architecture is not explicitly
            listed in ``value_by_target``.
        value_by_target: Mapping from GPU architecture name to value. Keys must be strings of
            the form ``"sm_<major><minor>"``, such as ``"sm_100"`` or ``"sm_120"``.

    Examples
    --------
    Use one ``num_ctas`` value for all architectures:

    .. code-block:: python

        from cuda.tile import kernel, ByTarget

        @kernel(num_ctas=8)
        def kernel_fn(x):
            ...

    Use different ``num_ctas`` values for specific architectures, and a
    fallback value for all others:

    .. code-block:: python

        from cuda.tile import kernel, ByTarget

        @kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
        def kernel_fn(x):
            ...
    """
    def __init__(self, *, default=UNSPECIFIED, **value_by_target):
        for sm, value in value_by_target.items():
            if not _is_valid_sm_string(sm):
                raise ValueError(f"Invalid GPU architecture name: {sm}, expected sm_<major><minor>")
        self._default = default
        self._by_target = value_by_target

    def __repr__(self):
        entries = [f"{sm}={repr(value)}" for sm, value in self._by_target.items()]
        if self._default is not UNSPECIFIED:
            entries.append(f"default={repr(self._default)}")
        return f"ByTarget({', '.join(entries)})"

    def __eq__(self, other):
        if not isinstance(other, ByTarget):
            return False
        return self._default == other._default and self._by_target == other._by_target


def _is_valid_sm_string(s: str) -> bool:
    return s.startswith("sm_") and s[3:].isdigit()
