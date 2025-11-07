.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.tile

Data Model
==========

cuTile is an array-based programming model.
The fundamental data structure is multidimensional arrays with elements of a single homogeneous type.
cuTile Python does not expose pointers, only arrays.

An array-based model was chosen because:

- Arrays know their bounds, so accesses can be checked to ensure safety and correctness.
- Array-based load/store operations can be efficiently lowered to speed-of-light hardware mechanisms.
- Python programmers are used to array-based programming frameworks such as NumPy.
- Pointers are not a natural choice for Python.

Within |tile code|, only the types described in this section are supported.

Global Arrays
-------------

.. autoclass:: Array
   :no-members:
   :no-index:

   .. seealso::
      :ref:`Complete cuda.tile.Array class documentation <data/array:cuda.tile.Array>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/array

Tiles
-----

.. autoclass:: Tile
   :no-members:
   :no-index:

   .. seealso::
      :ref:`Complete cuda.tile.Tile class documentation <data/tile:cuda.tile.Tile>`

.. toctree::
   :maxdepth: 2
   :hidden:

   data/tile

Element & Tile Space
--------------------

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__dark_background.svg
   :class: only-dark

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_2x4__tile_grid_6x4__light_background.svg
   :class: only-light

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__dark_background.svg
   :class: only-dark

.. image:: /_static/images/cutile__indexing__array_shape_12x16__tile_shape_4x2__tile_grid_3x8__light_background.svg
   :class: only-light

The *element space* of an array is the multidimensional space of elements contained in that array,
stored in memory according to a certain layout (row major, column major, etc).

The *tile space* of an array is the multidimensional space of tiles into that array of a certain
tile shape.
A tile index ``(i, j, ...)`` with shape ``S`` refers to the elements of the array that belong to the
``(i+1)``-th, ``(j+1)``-th, ... tile.

When accessing the elements of an array using tile indices, the multidimensional memory layout of the array is used.
To access the tile space with a different memory layout, use the `order` parameter of load/store operations.

Shape Broadcasting
------------------

*Shape broadcasting* allows |tiles| with different shapes to be combined in arithmetic operations.
When performing operations between |tiles| of different shapes, smaller |tile| is automatically
extended to match the shape of the larger one, following these rules:

- |Tiles| are aligned by their trailing dimensions.
- If the corresponding dimensions have the same size or one of them is 1, they are compatible.
- If one |tile| has fewer dimensions, its shape is padded with 1s on the left.

Broadcasting follows the same semantics as |NumPy|, which makes code more concise and readable
while maintaining computational efficiency.

Data Types
----------

.. autoclass:: cuda.tile.DType()
   :members:

.. include:: generated/includes/numeric_dtypes.rst

Numeric & Arithmetic Data Types
-------------------------------
A *numeric* data type represents numbers. An *arithmetic* data type is a numeric data type
that supports general arithmetic operations such as addition, subtraction, multiplication,
and division.

Arithmetic Promotion
--------------------

Tile-Tile & Scalar-Scalar Promotion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When performing a binary operation on either two |tiles| or two |scalars|, if the objects have different
|numeric dtypes|, both shall be *promoted* to the |dtype| specified in the table below.

Such promotions follow these principles:

- If one |dtype| is an integer type and the other is a floating-point type, the result is a floating-point type.
- If both |dtypes| are integer types, the result is the |dtype| with the larger number of bits.
- If both |dtypes| are floating-point types, the result is the |dtype| with the larger number of bits.
- If one |dtype| is a signed integer and the other is an unsigned integer, an error occurs.

.. rst-class:: compact-table

.. include:: generated/includes/dtype_promotion_table.rst

Tile-Scalar Promotion
~~~~~~~~~~~~~~~~~~~~~

When performing a binary operation between a |tile| and a |scalar| with different |numeric dtypes|,
promotion is determined by comparing their respective |dtype| categories:

- When the category of the |scalar| is less than or equal to the category of the |tile|, the
  |scalar| is promoted or demoted to match the |tile|'s |dtype|.
- When the category of the |scalar| is greater than the category of the |tile|, the |tile| is
  promoted to match the |scalar|'s |dtype|.

These rules ensure that operations between tiles and scalars follow a consistent pattern of type
promotion that preserves numeric precision while allowing for efficient computation.

Scalars
-------

A *scalar* is a single immutable value of a specific |data type|.

|Scalars| can be used in |host code| and |tile code|.
They can be |kernel| parameters.

.. toctree::
   :maxdepth: 2
   :hidden:

   data/scalar

Tuples
------

Tuples can be used in |tile code|. They cannot be |kernel| parameters.

Rounding Modes
--------------

.. autoclass:: cuda.tile.RoundingMode
   :members:
   :undoc-members:
   :member-order: bysource

Padding Modes
-------------

.. autoclass:: cuda.tile.PaddingMode
   :members:
   :undoc-members:
   :member-order: bysource