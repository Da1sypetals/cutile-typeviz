#!/usr/bin/env python3
"""
Flash Attention Kernel IR Dump Example

Use ir_dump library to compile flash attention kernel and dump IR.
Note: This is a simplified version without batch and multihead support.

Usage:
    python attention_ir_example.py

Generated IR files will be saved in ./ir_artifacts directory:
    - flash_attention.cutileir: CuTile IR (high-level IR)
    - flash_attention.cutile: Bytecode (serialized IR)
"""

import sys
import os
import cuda.tile as ct


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
    from ir_dump import (
        dump_cutileir,
        dump_bytecode,
        dump_typechecked_ir,
        get_function_repr,
        MockTensor,
    )

    """Main function"""
    print("=" * 60)
    print("Flash Attention Kernel IR Dump Example")
    print("=" * 60)

    # Define output directory
    output_dir = "./ir_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters
    batch_size = 8
    num_head = 16
    seq_len = 1024
    hidden_size = 64
    Br = 32  # block row size
    Bc = 32  # block col size

    # Create mock tensors
    q = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    k = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    v = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")
    out = MockTensor((batch_size, num_head, seq_len, hidden_size), dtype="float32")

    # Compile and dump IR
    print("\nCompiling flash attention kernel...")
    print(f"  - seq_len: {seq_len}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - block size: Br={Br}, Bc={Bc}")

    args = [q, k, v, out, hidden_size, Br, Bc]
    kernel_name = "flash_attention"

    # Dump CuTile IR
    cutileir = dump_cutileir(flash_attention_forward_v2, args)
    cutileir_path = os.path.join(output_dir, f"{kernel_name}.cutileir")
    with open(cutileir_path, "w") as f:
        f.write(cutileir)

    # Dump bytecode
    bytecode = dump_bytecode(flash_attention_forward_v2, args)
    bytecode_path = os.path.join(output_dir, f"{kernel_name}.cutile")
    with open(bytecode_path, "wb") as f:
        f.write(bytes.fromhex(bytecode))

    # Dump typechecked IR
    typechecked_ir = dump_typechecked_ir(flash_attention_forward_v2, args)
    typechecked_ir_path = os.path.join(output_dir, f"{kernel_name}.typechecked_ir")
    with open(typechecked_ir_path, "w") as f:
        f.write(typechecked_ir)

    files = {
        "cutileir": cutileir_path,
        "bytecode": bytecode_path,
        "typechecked_ir": typechecked_ir_path,
    }

    print("Generated files:")
    for ir_type, path in files.items():
        print(f"  - {ir_type}: {path}")

    func_ir = get_function_repr(flash_attention_forward_v2, args)
    print(f"\n`get_function_repr` returns func_ir of type:\n{type(func_ir)}")


if __name__ == "__main__":
    main()
