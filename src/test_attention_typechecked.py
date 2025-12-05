#!/usr/bin/env python3
"""
测试 dump_typechecked_ir 功能 - 使用 Flash Attention Kernel

使用方法：
    cd src && python test_attention_typechecked.py

生成的 IR 文件将保存在 ./ir_artifacts 目录下：
    - flash_attention_typechecked.cutileir: Type-checked IR (infer_types_pass 后的 IR)
"""

import cuda.tile as ct
from ir_dump import CutileIrDump


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

    # 测试 dump_typechecked_ir
    print("\n正在生成 type-checked IR...")
    ir_string = dumper.dump_typechecked_ir(
        flash_attention_forward_v2,
        [q, k, v, out, hidden_size, Br, Bc],
        output_file="flash_attention_typechecked.cutileir",
    )

    print("✓ 成功生成 type-checked IR")
    print("  文件: ./ir_artifacts/flash_attention_typechecked.cutileir")
    print(f"  大小: {len(ir_string)} 字符")


if __name__ == "__main__":
    main()
