"""
Test transpiler for various operations:
- rsqrt (reciprocal square root)
- argmax/argmin (arg reduce operations)
- bitshift (left shift, right shift)
- arange (tile factory)
- cumsum/cumprod (scan operations)
- bitcast (type reinterpretation)

Uses large tensors (4096+) with small tiles (64/128) to test multi-block execution.
"""

import cuda.tile as ct
import numpy as np


# Tile dimensions (small)
TileSize = 64
TileRows = 32
TileCols = 64

# Tensor dimensions (large, all >1024)
TensorSize = 4096
TensorRows = 2048
TensorCols = 1024


@ct.kernel
def test_unary_ops(
    x_in: ct.Array,  # [TensorSize]
    rsqrt_out: ct.Array,  # [TensorSize]
):
    """
    Test rsqrt operation.

    <typecheck>
    MockTensor((TensorSize,), dtype="float32")
    MockTensor((TensorSize,), dtype="float32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load input tile (index is block index, not offset)
    x = ct.load(x_in, index=(bid,), shape=(TileSize,))

    # Test rsqrt: 1 / sqrt(x)
    y = ct.rsqrt(x)

    ct.store(rsqrt_out, index=(bid,), tile=y)


@ct.kernel
def test_arg_reduce_ops(
    x_in: ct.Array,  # [TensorRows, TensorCols]
    argmax_out: ct.Array,  # [TensorRows]
    argmin_out: ct.Array,  # [TensorRows]
):
    """
    Test argmax and argmin operations.

    <typecheck>
    MockTensor((TensorRows, TensorCols), dtype="float32")
    MockTensor((TensorRows,), dtype="int32")
    MockTensor((TensorRows,), dtype="int32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load input tile (index is block index)
    x = ct.load(x_in, index=(bid, 0), shape=(TileRows, TileCols))

    # Test argmax along axis 1
    idx_max = ct.argmax(x, axis=1)

    # Test argmin along axis 1
    idx_min = ct.argmin(x, axis=1)

    ct.store(argmax_out, index=(bid,), tile=idx_max)
    ct.store(argmin_out, index=(bid,), tile=idx_min)


@ct.kernel
def test_bitshift_ops(
    x_in: ct.Array,  # [TensorSize]
    shift_amt: ct.Array,  # [TensorSize]
    lshift_out: ct.Array,  # [TensorSize]
    rshift_out: ct.Array,  # [TensorSize]
):
    """
    Test bit shift operations (left shift and right shift).

    <typecheck>
    MockTensor((TensorSize,), dtype="int32")
    MockTensor((TensorSize,), dtype="int32")
    MockTensor((TensorSize,), dtype="int32")
    MockTensor((TensorSize,), dtype="int32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load inputs (index is block index)
    x = ct.load(x_in, index=(bid,), shape=(TileSize,))
    shift = ct.load(shift_amt, index=(bid,), shape=(TileSize,))

    # Test left shift: x << shift
    y_lshift = ct.bitwise_lshift(x, shift)

    # Test right shift: x >> shift
    y_rshift = ct.bitwise_rshift(x, shift)

    ct.store(lshift_out, index=(bid,), tile=y_lshift)
    ct.store(rshift_out, index=(bid,), tile=y_rshift)


@ct.kernel
def test_arange_op(
    out: ct.Array,  # [TensorSize]
):
    """
    Test arange factory operation.

    <typecheck>
    MockTensor((TensorSize,), dtype="int32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Create arange tile: [0, 1, 2, ..., TileSize-1]
    indices = ct.arange(TileSize, dtype=ct.int32)

    # Offset by bid to get global indices
    indices = indices + bid * TileSize

    ct.store(out, index=(bid,), tile=indices)


@ct.kernel
def test_scan_ops(
    x_in: ct.Array,  # [TensorRows, TensorCols]
    cumsum_out: ct.Array,  # [TensorRows, TensorCols]
    cumprod_out: ct.Array,  # [TensorRows, TensorCols]
):
    """
    Test scan operations: cumsum and cumprod.

    <typecheck>
    MockTensor((TensorRows, TensorCols), dtype="float32")
    MockTensor((TensorRows, TensorCols), dtype="float32")
    MockTensor((TensorRows, TensorCols), dtype="float32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load input tile (index is block index)
    x = ct.load(x_in, index=(bid, 0), shape=(TileRows, TileCols))

    # Test cumsum along axis 1
    y_cumsum = ct.cumsum(x, axis=1)

    # Test cumprod along axis 1
    y_cumprod = ct.cumprod(x, axis=1)

    ct.store(cumsum_out, index=(bid, 0), tile=y_cumsum)
    ct.store(cumprod_out, index=(bid, 0), tile=y_cumprod)


@ct.kernel
def test_bitcast_op(
    x_in: ct.Array,  # [TensorSize] float32
    out: ct.Array,  # [TensorSize] int32
):
    """
    Test bitcast operation (reinterpret float32 as int32).

    <typecheck>
    MockTensor((TensorSize,), dtype="float32")
    MockTensor((TensorSize,), dtype="int32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load input (float32) (index is block index)
    x = ct.load(x_in, index=(bid,), shape=(TileSize,))

    # Bitcast float32 to int32 (view the bits as integer)
    y = ct.bitcast(x, dtype=ct.int32)

    ct.store(out, index=(bid,), tile=y)


@ct.kernel
def test_assert_op(
    x_in: ct.Array,  # [TensorSize] float32 (positive values)
    out: ct.Array,  # [TensorSize] float32
):
    """
    Test assert_ operation.

    <typecheck>
    MockTensor((TensorSize,), dtype="float32")
    MockTensor((TensorSize,), dtype="float32")
    </typecheck>
    """
    bid = ct.bid(0)

    # Load input (must be positive for assert test)
    x = ct.load(x_in, index=(bid,), shape=(TileSize,))

    # Test assert without message - all elements should be > 0
    ct.assert_(x > 0.0)

    # Test assert with message
    ct.assert_(x < 100.0, "All elements should be less than 100")

    # Output the input (if asserts pass)
    ct.store(out, index=(bid,), tile=x)


# cutile-typeviz: end

from cutile_typeviz.transpiler import launch_numpy
from pathlib import Path


def run_test_unary_ops():
    """Test rsqrt operation."""
    print("\n=== Testing rsqrt ===")
    print(f"  Tensor size: {TensorSize}, Tile size: {TileSize}")

    # Create input (must be positive for sqrt)
    x_in = np.random.uniform(low=0.5, high=4.0, size=(TensorSize,)).astype(np.float32)
    rsqrt_out = np.zeros((TensorSize,), dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "test_unary"

    # Number of blocks needed
    num_blocks = TensorSize // TileSize

    launch_numpy(
        test_unary_ops,
        [x_in, rsqrt_out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify rsqrt
    expected_rsqrt = 1.0 / np.sqrt(x_in)
    mae_rsqrt = np.abs(rsqrt_out - expected_rsqrt).mean()
    print(f"rsqrt MAE: {mae_rsqrt}")
    print(f"  Input[:5]:    {x_in[:5]}")
    print(f"  Output[:5]:   {rsqrt_out[:5]}")
    print(f"  Expected[:5]: {expected_rsqrt[:5]}")

    assert mae_rsqrt < 1e-5, f"rsqrt test failed! MAE={mae_rsqrt}"
    print("rsqrt test PASSED!")


def run_test_arg_reduce():
    """Test argmax and argmin operations."""
    print("\n=== Testing argmax/argmin ===")
    print(f"  Tensor: ({TensorRows}, {TensorCols}), Tile: ({TileRows}, {TileCols})")

    # Note: We only process first TileCols columns per row since tile width is TileCols
    x_in = np.random.uniform(low=-10.0, high=10.0, size=(TensorRows, TensorCols)).astype(np.float32)
    argmax_out = np.zeros((TensorRows,), dtype=np.int32)
    argmin_out = np.zeros((TensorRows,), dtype=np.int32)

    tmp_dir = Path("ir_artifacts") / "test_arg_reduce"

    # Number of blocks (processing TileRows rows per block)
    num_blocks = TensorRows // TileRows

    launch_numpy(
        test_arg_reduce_ops,
        [x_in, argmax_out, argmin_out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify argmax/argmin (only on the first TileCols columns)
    expected_argmax = np.argmax(x_in[:, :TileCols], axis=1).astype(np.int32)
    expected_argmin = np.argmin(x_in[:, :TileCols], axis=1).astype(np.int32)

    print(f"argmax output[:8]:   {argmax_out[:8]}")
    print(f"argmax expected[:8]: {expected_argmax[:8]}")
    print(f"argmin output[:8]:   {argmin_out[:8]}")
    print(f"argmin expected[:8]: {expected_argmin[:8]}")

    assert np.array_equal(argmax_out, expected_argmax), "argmax test failed!"
    assert np.array_equal(argmin_out, expected_argmin), "argmin test failed!"
    print("argmax/argmin test PASSED!")


def run_test_bitshift():
    """Test bitshift operations."""
    print("\n=== Testing bitshift ===")
    print(f"  Tensor size: {TensorSize}, Tile size: {TileSize}")

    # Create larger arrays
    base_x = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)
    base_shift = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.int32)

    # Tile the arrays to TensorSize
    x_in = np.tile(base_x, TensorSize // len(base_x) + 1)[:TensorSize].astype(np.int32)
    shift_amt = np.tile(base_shift, TensorSize // len(base_shift) + 1)[:TensorSize].astype(np.int32)

    lshift_out = np.zeros((TensorSize,), dtype=np.int32)
    rshift_out = np.zeros((TensorSize,), dtype=np.int32)

    tmp_dir = Path("ir_artifacts") / "test_bitshift"

    num_blocks = TensorSize // TileSize

    launch_numpy(
        test_bitshift_ops,
        [x_in, shift_amt, lshift_out, rshift_out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify left shift
    expected_lshift = (x_in << shift_amt).astype(np.int32)
    # Verify right shift
    expected_rshift = (x_in >> shift_amt).astype(np.int32)

    print(f"x_in[:8]:           {x_in[:8]}")
    print(f"shift_amt[:8]:      {shift_amt[:8]}")
    print(f"lshift output[:8]:  {lshift_out[:8]}")
    print(f"lshift expected[:8]:{expected_lshift[:8]}")
    print(f"rshift output[:8]:  {rshift_out[:8]}")
    print(f"rshift expected[:8]:{expected_rshift[:8]}")

    assert np.array_equal(lshift_out, expected_lshift), "lshift test failed!"
    assert np.array_equal(rshift_out, expected_rshift), "rshift test failed!"
    print("bitshift test PASSED!")


def run_test_arange():
    """Test arange factory operation."""
    print("\n=== Testing arange ===")
    print(f"  Tensor size: {TensorSize}, Tile size: {TileSize}")

    out = np.zeros((TensorSize,), dtype=np.int32)

    tmp_dir = Path("ir_artifacts") / "test_arange"

    num_blocks = TensorSize // TileSize

    launch_numpy(
        test_arange_op,
        [out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify arange - each block produces [bid*TileSize, bid*TileSize+1, ..., bid*TileSize+TileSize-1]
    expected = np.arange(TensorSize, dtype=np.int32)

    print(f"arange output[:8]:     {out[:8]}")
    print(f"arange expected[:8]:   {expected[:8]}")
    print(f"arange output[-8:]:    {out[-8:]}")
    print(f"arange expected[-8:]:  {expected[-8:]}")

    assert np.array_equal(out, expected), "arange test failed!"
    print("arange test PASSED!")


def run_test_scan():
    """Test cumsum and cumprod operations."""
    print("\n=== Testing cumsum/cumprod ===")
    print(f"  Tensor: ({TensorRows}, {TensorCols}), Tile: ({TileRows}, {TileCols})")

    # Use small values to avoid overflow in cumprod
    # Only the first TileCols columns are processed
    x_in = np.random.uniform(low=0.9, high=1.1, size=(TensorRows, TensorCols)).astype(np.float32)
    cumsum_out = np.zeros((TensorRows, TensorCols), dtype=np.float32)
    cumprod_out = np.zeros((TensorRows, TensorCols), dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "test_scan"

    num_blocks = TensorRows // TileRows

    launch_numpy(
        test_scan_ops,
        [x_in, cumsum_out, cumprod_out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify cumsum/cumprod (only on first TileCols columns)
    expected_cumsum = np.cumsum(x_in[:, :TileCols], axis=1)
    expected_cumprod = np.cumprod(x_in[:, :TileCols], axis=1)

    # Compare only the first TileCols columns
    actual_cumsum = cumsum_out[:, :TileCols]
    actual_cumprod = cumprod_out[:, :TileCols]

    mae_cumsum = np.abs(actual_cumsum - expected_cumsum).mean()
    mae_cumprod = np.abs(actual_cumprod - expected_cumprod).mean()

    print(f"cumsum MAE: {mae_cumsum}")
    print(f"  Input[0,:5]:    {x_in[0, :5]}")
    print(f"  Output[0,:5]:   {actual_cumsum[0, :5]}")
    print(f"  Expected[0,:5]: {expected_cumsum[0, :5]}")

    print(f"cumprod MAE: {mae_cumprod}")
    print(f"  Output[0,:5]:   {actual_cumprod[0, :5]}")
    print(f"  Expected[0,:5]: {expected_cumprod[0, :5]}")

    assert mae_cumsum < 1e-4, f"cumsum test failed! MAE={mae_cumsum}"
    assert mae_cumprod < 1e-4, f"cumprod test failed! MAE={mae_cumprod}"
    print("cumsum/cumprod test PASSED!")


def run_test_bitcast():
    """Test bitcast operation."""
    print("\n=== Testing bitcast ===")
    print(f"  Tensor size: {TensorSize}, Tile size: {TileSize}")

    # Create random float32 array
    x_in = np.random.uniform(low=-10.0, high=10.0, size=(TensorSize,)).astype(np.float32)
    out = np.zeros((TensorSize,), dtype=np.int32)

    tmp_dir = Path("ir_artifacts") / "test_bitcast"

    num_blocks = TensorSize // TileSize

    launch_numpy(
        test_bitcast_op,
        [x_in, out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify bitcast (view float32 bits as int32)
    expected = x_in.view(np.int32)

    print(f"bitcast input (float32)[:5]:  {x_in[:5]}")
    print(f"bitcast output (int32)[:5]:   {out[:5]}")
    print(f"bitcast expected (int32)[:5]: {expected[:5]}")

    assert np.array_equal(out, expected), "bitcast test failed!"
    print("bitcast test PASSED!")


def run_test_assert():
    """Test assert_ operation."""
    print("\n=== Testing assert_ ===")
    print(f"  Tensor size: {TensorSize}, Tile size: {TileSize}")

    # Create positive float32 array (must satisfy x > 0 and x < 100)
    x_in = np.random.uniform(low=0.1, high=50.0, size=(TensorSize,)).astype(np.float32)
    out = np.zeros((TensorSize,), dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "test_assert"

    num_blocks = TensorSize // TileSize

    launch_numpy(
        test_assert_op,
        [x_in, out],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Verify output equals input (asserts should pass)
    mae = np.abs(out - x_in).mean()

    print(f"assert_ MAE: {mae}")
    print(f"  Input[:5]:  {x_in[:5]}")
    print(f"  Output[:5]: {out[:5]}")

    assert mae < 1e-6, f"assert test failed! MAE={mae}"
    print("assert_ test PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Transpiler Operations")
    print(f"Tile dimensions: TileSize={TileSize}, TileRows={TileRows}, TileCols={TileCols}")
    print(f"Tensor dimensions: TensorSize={TensorSize}, TensorRows={TensorRows}, TensorCols={TensorCols}")
    print("=" * 60)

    # Run all tests
    run_test_unary_ops()  # rsqrt
    run_test_arg_reduce()  # argmax/argmin
    run_test_bitshift()  # lshift/rshift
    run_test_arange()  # arange
    run_test_scan()  # cumsum/cumprod
    run_test_bitcast()  # bitcast
    run_test_assert()  # assert_

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
