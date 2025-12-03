.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

cuTile Python
=============

cuTile is a parallel programming model for NVIDIA GPUs and a Python-based :abbr:`DSL (Domain-Specific Language)`.
It automatically leverages advanced hardware capabilities, such as tensor cores and tensor memory accelerators,
while providing portability across different NVIDIA GPU architectures.
cuTile enables the latest hardware features without requiring code changes.

cuTile |kernels| are GPU programs that are executed in parallel on a logical |grid| of |blocks|.
The :py:class:`@ct.kernel <cuda.tile.kernel>` decorator marks a Python function as a kernel's entry point.
Kernels cannot be called directly from the host code; the host must queue kernels for execution on GPU
using the :py:func:`ct.launch() <cuda.tile.launch>` function:

.. literalinclude:: ../../test/test_frontpage_example.py
    :language: python
    :dedent:
    :start-after: example-begin
    :end-before: example-end

|Kernels| move data between |arrays| and |tiles| using functions like
:py:func:`ct.load() <cuda.tile.load>` and :py:func:`ct.store() <cuda.tile.store>`.
Both arrays and tiles are tensor-like data structures: each has a specific shape
(i.e., the number of elements along each axis) and a |dtype| (i.e., the data type of elements).
However, there are important differences:

- |Arrays| are stored in the global memory. They are mutable and have physical, strided
  memory layouts. Within the kernel code, they support only a limited set of operations,
  mostly related to |loading and storing| data to/from tiles. Various Python objects,
  including PyTorch tensors and CuPy arrays, can be passed as arrays from the host code
  to the kernel via kernel arguments.

- |Tiles| are immutable values without defined storage that only exist in the kernel code.
  Tile dimensions must be compile-time constants that are powers of two.
  Tiles support a multitude of |operations|, including elementwise arithmetic,
  matrix multiplication, reduction, shape manipulation, etc.

Proceed to the :ref:`quickstart` page for installation instructions and a complete working example.


.. toctree::
   :maxdepth: 2
   :hidden:

   release_notes
   quickstart
   execution
   data
   memory_model
   interoperability
   performance
   operations
   debugging
