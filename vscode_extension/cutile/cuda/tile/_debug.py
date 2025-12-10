# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os


# Internal environment variables for debugging

CUDA_TILE_DUMP_TILEIR = os.environ.get('CUDA_TILE_DUMP_TILEIR', None)
CUDA_TILE_DUMP_BYTECODE = os.environ.get('CUDA_TILE_DUMP_BYTECODE', None)
CUDA_TILE_TESTING_DISABLE_DIV = (
    os.environ.get("CUDA_TILE_TESTING_DISABLE_DIV", "0") == "1")
CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER = (
    os.environ.get("CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER", "0") == "1")
