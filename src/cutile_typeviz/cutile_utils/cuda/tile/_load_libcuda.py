# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from ctypes import CDLL


if os.name == 'nt':
    _dll = CDLL("nvcuda.dll")
else:
    _dll = CDLL("libcuda.so.1")
_cuGetProcAddress_v2 = _dll["cuGetProcAddress_v2"]

cuGetProcAddress_v2_ptrptr = ctypes.addressof(_cuGetProcAddress_v2)
