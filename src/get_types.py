import cuda.tile as ct
from ir_dump import CutileIrDump
from cuda.tile._ir.ir import Function, Block, Operation
from cuda.tile._ir.ops import IfElse, Loop
from cuda.tile._ir.ops import Assign
from copy import deepcopy
from icecream import ic


@ct.kernel
def flash_attention_forward_v2(
    q,
    k,
    v,
    out,
    hidden_size: ct.Constant,
    br: ct.Constant,
    bc: ct.Constant,
):
    ib = ct.bid(0)
    ih = ct.bid(1)
    z = ih * ct.float64(244.1)
    Tc = k.shape[0] // bc
    qi = ct.load(q, index=(ib, ih, ct.bid(2), 0), shape=(1, 1, br, hidden_size))

    qi = ct.reshape(qi, (br, hidden_size))

    oi = ct.full((br, hidden_size), 0.0, dtype=q.dtype)
    li = ct.full((br, 1), 0.0, dtype=q.dtype)
    mi = ct.full((br, 1), -1e10, dtype=q.dtype)

    for j in range(0, Tc):
        kjt = ct.load(k, index=(ib, ih, j, 0), shape=(1, 1, bc, hidden_size))
        vj = ct.load(v, index=(ib, ih, j, 0), shape=(1, 1, bc, hidden_size))
        kjt = ct.reshape(kjt, (bc, hidden_size))
        vj = ct.reshape(vj, (bc, hidden_size))
        kj = kjt.transpose()
        sij = ct.matmul(qi, kj) / hidden_size**0.5
        mij = ct.max(sij, axis=-1, keepdims=True)
        mi_mij = ct.cat((mi, mij), axis=-1)
        mi_new = ct.max(mi_mij, axis=-1, keepdims=True)
        pij = ct.exp(sij - mi_new)
        lij = ct.sum(pij, axis=-1, keepdims=True)
        exp_mi = ct.exp(mi - mi_new)
        li_new = li * exp_mi + lij
        oi = ct.mma(pij, vj, oi * exp_mi)
        li = li_new
        mi = mi_new

    oi = oi / li
    ct.store(out, index=(ib, ih, ct.bid(2), 0), tile=oi)


def flatten_operations(operations: list[Operation]) -> list[Operation]:
    flattened = []
    for i in range(len(operations)):
        op = operations[i]
        if isinstance(op, Loop):
            body = flatten_operations(op.body._operations)
            flattened.extend(body)

        elif isinstance(op, IfElse):
            then_block = flatten_operations(op.then_block._operations)
            else_block = flatten_operations(op.else_block._operations)
            flattened.extend(then_block)
            flattened.extend(else_block)

        else:
            flattened.append(op)
    return flattened


def list_all_operations(func: Function) -> list[Operation]:
    operations = func.root_block._operations
    return flatten_operations(operations)


def main():
    """主函数"""
    print("=" * 60)
    print("测试 dump_typechecked_ir - Flash Attention Kernel")
    print("=" * 60)

    # 创建 IR dumper
    dumper = CutileIrDump(output_dir="./ir_artifacts")

    # 定义参数
    batch_size = 8
    num_head = 16
    seq_len = 1024
    hidden_size = 64
    Br = 32  # block row size
    Bc = 32  # block col size

    # 创建 mock tensors
    q = dumper.create_mock_tensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    k = dumper.create_mock_tensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    v = dumper.create_mock_tensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    out = dumper.create_mock_tensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")

    print("\n参数配置：")
    print(f"  - batch_size: {batch_size}")
    print(f"  - num_head: {num_head}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - block size: Br={Br}, Bc={Bc}")

    dumper.dump_typechecked_ir(
        flash_attention_forward_v2,
        [q, k, v, out, hidden_size, Br, Bc],
        output_file="flash_attention_typechecked.cutileir",
    )

    # Manipulate function
    func_repr = dumper.get_function_repr(
        kernel_func=flash_attention_forward_v2,
        args=[q, k, v, out, hidden_size, Br, Bc],
    )
    ic(func_repr)

    flattened_ops = list_all_operations(func_repr)
    ic(len(flattened_ops))

    for op in flattened_ops:
        if isinstance(op, Assign):
            # ic(op)
            if not str(op.result_var).startswith("$"):
                print(f"{op.loc}--{op.loc.end_col} | {op}")


if __name__ == "__main__":
    main()
