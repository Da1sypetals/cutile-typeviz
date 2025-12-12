## Type Annotation Rules

To enable type checking and visualization for cuTile kernels, you need to annotate your kernel function's parameters using the `<typecheck>` tag in the docstring.

### Syntax

```python
@ct.kernel
def your_kernel(param1, param2, param3, const_param: ConstInt):
    """
    <typecheck>
    MockTensor((shape_tuple), dtype="dtype_string")
    MockTensor((shape_tuple), dtype="dtype_string")
    MockTensor((shape_tuple), dtype="dtype_string")
    scalar_value
    </typecheck>
    
    Your docstring description here...
    """
    # kernel implementation
```

### Rules

1. **Placement**: The `<typecheck>` block must be placed at the **beginning** of the docstring
2. **Order**: Each line corresponds to a function parameter **in the same order** as they appear in the function signature
3. **One per line**: Each parameter annotation must be on its own line

### Parameter Types

| Parameter Type | Annotation Format | Example |
|----------------|-------------------|---------|
| Tensor | `MockTensor((...shape), dtype="dtype")` | `MockTensor((1024, 2048), dtype="float16")` |
| Integer scalar | Numeric literal | `32`, `64`, `128` |
| Float scalar | Numeric literal | `1e-5`, `0.1` |

### Supported Data Types (dtype)

The following dtype strings are supported for `MockTensor`:

| Category | Supported Types |
|----------|-----------------|
| Boolean | `bool` |
| Unsigned Integer | `uint8`, `uint16`, `uint32`, `uint64` |
| Signed Integer | `int8`, `int16`, `int32`, `int64` |
| Floating Point | `float16`, `float32`, `float64`, `bfloat16`, `tfloat32` |
| FP8 | `float8_e4m3fn`, `float8_e5m2` |

### Complete Example

```python
@ct.kernel
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    <typecheck>
    MockTensor((64, 2048), dtype="float32")
    MockTensor((64, 2048), dtype="float32")
    MockTensor((2048,), dtype="float16")
    MockTensor((2048,), dtype="float16")
    32
    1024
    </typecheck>
    Backward pass part 2: Final reduction for dW and dB.

    Args:
        DW: Partial gradient with respect to W (TILE_M, N).
        DB: Partial gradient with respect to B (TILE_M, N).
        FINAL_DW: Final gradient with respect to W (N,).
        FINAL_DB: Final gradient with respect to B (N,).
        TILE_M: Number of partial gradients to reduce.
        TILE_N: Tile size along N dimension.
    """
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))

    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        # Sum partial gradients
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)

    ct.store(FINAL_DW, index=(bid_n,), tile=sum_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(bid_n,), tile=sum_db.astype(FINAL_DB.dtype))
```


## Task

Now, add type annotation for this cuTile kernel. You should closely follow the annotation rules above.

```python
<FILL IN YOUR CODE>
```