import cuda.tile as ct


ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


def zfunc(a, b):
    sum = a + b
    res = ct.cos(sum)
    return res


def apply_mod(mod, c_tile, i_m, i_n, tm, tn):
    mod_tile = ct.load(mod, index=(i_m, i_n), shape=(tm, tn), padding_mode=PAD_ZERO)
    zval = zfunc(mod_tile, c_tile)
    return ct.sin(zval)


@ct.kernel
def my_kernel(a, b, c, mod, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    i_m = ct.bid(0)
    i_n = ct.bid(1)
    acc = ct.zeros((tm, tn), dtype=ct.float32)

    for i_k in range(tk):
        t_a = ct.load(a, index=(i_m, i_k), shape=(tm, tk), padding_mode=PAD_ZERO)
        t_b = ct.load(b, index=(i_k, i_n), shape=(tk, tn), padding_mode=PAD_ZERO)
        acc = ct.mma(t_a, t_b, acc)
        # c_tile = apply_mod(mod, acc, i_m, i_n, tm, tn).astype(ct.float16)

    tile1 = ct.full((32, 16), 0.0, ct.float32)
    tile2 = ct.full((32, 16), 0.0, ct.float32)
    tile3 = zfunc(tile1, tile2)

    tile4 = ct.full((16, 64), 2.0, ct.bfloat16)
    tile5 = ct.full((16, 64), 2.0, ct.bfloat16)
    tile6 = zfunc(tile4, tile5)

    c_tile = apply_mod(mod, acc, i_m, i_n, tm, tn).astype(ct.float16)
    ct.store(c, index=(i_m, i_n), tile=c_tile)


if __name__ == "__main__":
    from ir_dump.mock_tensor import MockTensor
    from ir_dump.dumper import CutileIrDump
    from ir_dump.shape_check import get_kernel_shapes_info
    import json

    dumper = CutileIrDump(
        output_dir="./ir_artifacts",
        dump_cutileir=True,
        dump_bytecode=True,
        dump_mlir=False,
    )

    # 定义参数
    M = 1024
    N = 2048
    K = 512
    TILE_M = 32
    TILE_N = 64
    TILE_K = 128

    # Forward pass tensors
    a = MockTensor((M, K), dtype="float16")
    b = MockTensor((K, N), dtype="float16")
    c = MockTensor((M, N), dtype="float16")
    mod = MockTensor((M, N), dtype="float16")

    # dumper.dump_typechecked_ir(
    #     kernel_func=my_kernel,
    #     args=[a, b, c, mod, TILE_M, TILE_N, TILE_K],
    #     output_file="my_func.tcir",
    # )

    # Get shape info for backward kernel part 2
    ops = get_kernel_shapes_info(
        kernel_func=my_kernel,
        args=[a, b, c, mod, TILE_M, TILE_N, TILE_K],
    )

    # ops_str = json.dumps(ops)
    # print(ops_str)
