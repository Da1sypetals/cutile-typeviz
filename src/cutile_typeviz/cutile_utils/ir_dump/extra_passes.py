from cuda.tile._ir import ir
from cuda.tile._ir.ops import BindMethod, GetBoundSelf, Assign


def eliminate_bound(func: ir.Function):
    """
    Eliminate bind_method and get_bound_self operations.

    This pass finds patterns like:
        v1 = bind_method(obj, func)
        v2 = get_bound_self(v1)
        v3 = tile_reshape(v2, ...)

    And replaces them with:
        v2 = obj
        v3 = tile_reshape(v2, ...)

    The key insight is that we need to track variable objects, not just variable names,
    because variable names may have been renumbered by earlier optimization passes.
    """
    # Find all bind_method and get_bound_self operations
    # Use dictionaries that map the result Var objects themselves
    bind_ops = {}  # result_var -> BindMethod operation
    get_bound_self_ops = {}  # result_var -> GetBoundSelf operation

    for op in func.root_block.traverse():
        if isinstance(op, BindMethod):
            bind_ops[op.result_vars[0]] = op
        elif isinstance(op, GetBoundSelf):
            get_bound_self_ops[op.result_vars[0]] = op

    # If no operations found, return early
    if not bind_ops and not get_bound_self_ops:
        return

    # Build a mapping from get_bound_self result Var to the object Var
    gbs_to_object = {}

    for gbs_result_var, gbs_op in get_bound_self_ops.items():
        bound_method_var = gbs_op.bound_method
        if bound_method_var in bind_ops:
            bind_op = bind_ops[bound_method_var]
            # Map get_bound_self result to bind_method object
            gbs_to_object[gbs_result_var] = bind_op.object

    # Build mappings for Assign operations
    # When we eliminate get_bound_self, we need to replace it with: result_var = object_var
    # When we eliminate bind_method, we don't need to insert anything (the result is no longer used)
    assign_to_insert = {}  # get_bound_self.result_var -> object_var

    for gbs_result_var, object_var in gbs_to_object.items():
        assign_to_insert[gbs_result_var] = object_var

    # Build a mapper to replace variables
    mapper = ir.Mapper(func.root_block.ctx, preserve_vars=True)

    # Set up the variable mappings
    for old_var, new_var in gbs_to_object.items():
        mapper.set_var(old_var, new_var)

    # Process the function
    def eliminate_in_block(block):
        new_ops = []
        for op in block:
            if isinstance(op, GetBoundSelf):
                # Replace get_bound_self with an assign operation
                if op.result_vars[0] in gbs_to_object:
                    object_var = gbs_to_object[op.result_vars[0]]
                    # Create an assign: result_var = object_var
                    result_var = op.result_vars[0]
                    assign_op = Assign(object_var, result_var, op.loc)
                    new_ops.append(assign_op)
                else:
                    # Keep this get_bound_self as-is (shouldn't happen)
                    for nested_block in op.nested_blocks:
                        eliminate_in_block(nested_block)
                    new_op = op.clone(mapper)
                    new_ops.append(new_op)

            elif isinstance(op, BindMethod):
                # Don't add bind_method operation
                # Its result is no longer needed after we insert the assign
                pass

            else:
                # Process nested blocks
                for nested_block in op.nested_blocks:
                    eliminate_in_block(nested_block)

                # Clone the operation with the mapper
                new_op = op.clone(mapper)
                new_ops.append(new_op)

        block[:] = new_ops

    eliminate_in_block(func.root_block)
