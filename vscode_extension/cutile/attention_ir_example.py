#!/usr/bin/env python3
"""
Flash Attention Kernel IR Dump 示例

使用 ir_dump 库编译 flash attention kernel 并导出 IR
注意：这是无 batch 和无 multihead 的简化版本

使用方法：
    python attention_ir_example.py

生成的 IR 文件将保存在 ./ir_artifacts 目录下：
    - flash_attention.cutileir: CuTile IR (高级 IR)
    - flash_attention.cutile: Bytecode (序列化的 IR)
"""

import sys
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
    print("Flash Attention Kernel IR Dump 示例")
    print("=" * 60)

    # 创建 IR dumper
    dumper = CutileIrDump(
        output_dir="./ir_artifacts",
        dump_cutileir=True,
        dump_bytecode=True,
        dump_mlir=False,
    )

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

    # 编译并导出 IR
    print("\n正在编译 flash attention kernel...")
    print(f"  - seq_len: {seq_len}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - block size: Br={Br}, Bc={Bc}")

    files = dumper.compile_kernel(
        flash_attention_forward_v2,
        args=[q, k, v, out, hidden_size, Br, Bc],
        kernel_name="flash_attention",
    )

    print("\n✓ 编译成功！生成的文件：")
    for ir_type, path in files.items():
        print(f"  - {ir_type}: {path}")


if __name__ == "__main__":
    main()
