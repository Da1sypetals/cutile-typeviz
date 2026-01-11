"""
Pass to eliminate tokens from the IR, converting token-ordered operations to plain operations.
"""

from cuda.tile._ir import ir
from cuda.tile._ir.ops import (
    TileLoadTokenOrdered,
    TileStoreTokenOrdered,
    MakeToken,
    JoinTokens,
    TileLoad,
    TileStore,
    TileReduce,
    TileArgReduce,
)


def simplify_for_numpy(func: ir.Function):
    _eliminate_in_block(func.root_block)
    _add_keepdim(func.root_block)


def _add_keepdim(block: ir.Block):
    """
    Add keepdims to the operation if it's a reduction operation.
    If the ndim of input are the same as the ndim of output, add keepdims=True;
    otherwise, add keepdims=False.
    """
    from cuda.tile._ir.type import TileTy
    
    for op in block:
        if isinstance(op, TileReduce) or isinstance(op, TileArgReduce):
            x = op.x
            result_var = op.result_var
            
            x_type = x.get_type()
            res_type = result_var.get_type()
            
            if isinstance(x_type, TileTy) and isinstance(res_type, TileTy):
                keepdims = x_type.ndim == res_type.ndim
                op.attributes["keepdims"] = keepdims
        
        # Process nested blocks
        for nested_block in op.nested_blocks:
            _add_keepdim(nested_block)


def _eliminate_in_block(block: ir.Block):
    """Eliminate token operations in a block and its nested blocks."""
    new_ops = []

    for op in block:
        if isinstance(op, MakeToken):
            # Skip make_token operations entirely
            pass

        elif isinstance(op, TileLoadTokenOrdered):
            # Convert tile_load_token_ordered to tile_load
            # The token_ordered version returns (result, token), but we only need result
            result_var = op.result_vars[0]
            new_op = TileLoad(
                array=op.array,
                index=op.index,
                order=op.order,
                padding_mode=op.padding_mode,
                latency=op.latency,
                allow_tma=op.allow_tma,
                result_var=result_var,
                loc=op.loc,
            )
            new_ops.append(new_op)

        elif isinstance(op, TileStoreTokenOrdered):
            # Convert tile_store_token_ordered to tile_store
            # The token_ordered version takes a token and returns a token, but we ignore both
            new_op = TileStore(
                array=op.array,
                index=op.index,
                tile=op.tile,
                order=op.order,
                latency=op.latency,
                allow_tma=op.allow_tma,
                loc=op.loc,
            )
            new_ops.append(new_op)

        elif isinstance(op, JoinTokens):
            # Skip join_tokens operations entirely
            pass

        else:
            # Process nested blocks
            for nested_block in op.nested_blocks:
                _eliminate_in_block(nested_block)

            # Keep other operations as-is
            new_ops.append(op)

    block[:] = new_ops
