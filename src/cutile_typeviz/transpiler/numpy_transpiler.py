import json
import re
import numpy as np

# FATAL: EVERY unimplemented or unsupported feature should raise an error.
# DO NOT silently ignore or skip of fallback for unsupported features.


class NumpyTranspiler:
    def __init__(self, json_data: dict):
        self.json_data = json_data
        self.lines = []
        self.indent_level = 0
        self.var_map = {}  # Map IR var names to Python var names
        self.imports = set(["import numpy as np", "import itertools"])
        self.loop_stack = []  # Stack of {carried_names: [], result_names: []}
        self.block_vars = {}  # Map axis to block variable name
        self.grid_dims = (0, 0, 0)  # Will be set from grid parameter

    def emit(self, line):
        indent = "    " * self.indent_level
        self.lines.append(f"{indent}{line}")

    def get_var_name(self, ir_name):
        # Convert $123 to _123, etc.
        # Function args like 'out' stay 'out'.
        # Loop vars like 'it.1' -> 'it', dropping the version identifier after '.'

        clean_name = ir_name
        if clean_name.startswith("$"):
            assert "." not in clean_name
            clean_name = clean_name.replace("$", "_")
        else:
            clean_name = clean_name.split(".")[0]

        return clean_name

    def transpile(self):
        func_name = self.json_data["qualname"]
        params = self.json_data["parameters"]

        # Add imports
        for imp in sorted(list(self.imports)):
            self.emit(imp)
        self.emit("")

        # Generate internal tile function (without block loops)
        self.tile_func_name = f"{func_name}_tile"
        param_names = [self.get_var_name(p["name"]) for p in params]
        # Internal tile function takes block indices as parameters instead of grid
        tile_param_str = ", ".join(param_names + ["block_0: int", "block_1: int", "block_2: int"])
        self.emit(f"def {self.tile_func_name}({tile_param_str}):")
        self.indent_level += 1

        # Process body (will handle tile_bid by using block parameters)
        self.process_block(self.json_data["operations"])

        # Add return if needed (IR might have explicit return, or void)
        # self.emit("return") # Optional

        self.indent_level -= 1
        self.emit("")

        # Generate wrapper function with grid loop
        wrapper_param_str = ", ".join(param_names + ["grid: tuple[int, int, int]"])
        self.emit(f"def {func_name}({wrapper_param_str}):")
        self.indent_level += 1

        # Store grid dimensions
        self.emit(
            "if not isinstance(grid, tuple) or len(grid) != 3 or not all(isinstance(x, int) for x in grid):"
        )
        self.indent_level += 1
        self.emit("raise TypeError(f'grid must be a tuple of 3 integers, got {grid}')")
        self.indent_level -= 1

        self.emit("grid_x, grid_y, grid_z = grid")
        self.emit("")

        # bid_axes = [0, 1, 2]
        # for axis in sorted(bid_axes):
        #     loop_var = f"block_{axis}"
        #     self.block_vars[axis] = loop_var
        #     if axis == 0:
        #         self.emit(f"for {loop_var} in range(grid_x):")
        #     elif axis == 1:
        #         self.emit(f"for {loop_var} in range(grid_y):")
        #     elif axis == 2:
        #         self.emit(f"for {loop_var} in range(grid_z):")
        #     self.indent_level += 1

        self.emit(
            " ".join(
                [
                    "for",
                    "block_0, block_1, block_2",
                    "in",
                    "itertools.product(range(grid_x), range(grid_y), range(grid_z)):",
                ]
            )
        )

        # Call internal tile function
        tile_args = list(param_names)
        tile_args.append("block_0=block_0")
        tile_args.append("block_1=block_1")
        tile_args.append("block_2=block_2")

        self.indent_level += 1
        self.emit(f"{self.tile_func_name}({', '.join(tile_args)})")

        # Close block loop
        self.indent_level -= 1

        # Close wrapper function
        self.indent_level -= 1

        return "\n".join(self.lines)

    def process_block(self, operations):
        if not operations:
            self.emit("pass")
            return
        for op in operations:
            self.handle_op(op)

    def handle_op(self, op):
        op_type = op["op"]
        method_name = f"handle_{op_type}"
        if hasattr(self, method_name):
            getattr(self, method_name)(op)
        else:
            raise TypeError(f"Unhandled {op_type}: {op}")

    def get_result_var(self, op):
        if op["result_vars"]:
            return self.get_var_name(op["result_vars"][0]["name"])
        return None

    def get_operand(self, op, name):
        if name not in op["operands"]:
            return None
        val = op["operands"][name]
        if isinstance(val, dict) and "name" in val:
            return self.get_var_name(val["name"])
        if isinstance(val, list):
            # List of vars (tuple items)
            return [self.get_var_name(v["name"]) for v in val]
        return val

    def get_operand_value(self, op, name):
        # Operands are always Vars in this IR (SSA).
        # Constants are created via typed_const.
        return self.get_operand(op, name)

    # --- Op Handlers ---

    def handle_tile_bid(self, op):
        res = self.get_result_var(op)
        axis = op["attributes"]["axis"]
        assert axis in (0, 1, 2), f"Invalid axis: {axis}, should be in (0, 1, 2)"
        # Map to block parameter
        block_var_name = f"block_{axis}"
        self.emit(f"{res} = {block_var_name}")

    def handle_typed_const(self, op):
        res = self.get_result_var(op)
        val = op["attributes"]["value"]

        # Check if result is a Tile type
        result_type = op["result_vars"][0]["type"]
        is_tile = result_type["type"] == "TileTy"

        if is_tile:
            # Parse shape from type string: "Tile[float32,(32,8,1)]"
            type_str = result_type["str"]
            shape_match = re.search(r"\(([\d,]+)\)", type_str)
            if shape_match:
                shape_str = shape_match.group(1)
                # Parse dtype
                dtype_match = re.search(r"Tile\[([^,]+),", type_str)
                dtype_str = dtype_match.group(1) if dtype_match else "float32"

                # Map Cutile dtype to numpy dtype
                np_dtype = "np.float32"
                if "float" in dtype_str:
                    np_dtype = f"np.{dtype_str}"
                elif "int" in dtype_str:
                    np_dtype = f"np.{dtype_str}"

                # Generate array filled with constant value
                self.emit(f"{res} = np.full(({shape_str}), {val}, dtype={np_dtype})")
            else:
                self.emit(f"{res} = {val}")
        elif isinstance(val, list):  # Tuple constant
            val_str = f"({', '.join(map(str, val))})"
            self.emit(f"{res} = {val_str}")
        else:
            val_str = str(val)
            self.emit(f"{res} = {val_str}")

    def handle_build_tuple(self, op):
        res = self.get_result_var(op)
        items = self.get_operand(op, "items")
        # items is a list of var names
        self.emit(f"{res} = ({', '.join(items)})")

    def handle_tile_load(self, op):
        res = self.get_result_var(op)
        arr = self.get_operand(op, "array")
        idx = self.get_operand(op, "index")

        # Parse tile shape from result type
        # Result type str: "Tile[float32,(1,32,4,4)]"
        res_type_str = op["result_vars"][0]["type"]["str"]
        shape_match = re.search(r"\(([\d,]+)\)", res_type_str)
        if shape_match:
            shape_str = shape_match.group(1)
            shape = [int(x) for x in shape_str.split(",")]
        else:
            shape = []  # Should not happen

        slice_parts = []
        for i, s in enumerate(shape):
            # slice_parts.append(f"{idx}[{i}]:{idx}[{i}]+{s}")
            slice_parts.append(f"{idx}[{i}] * {s} : {idx}[{i}] * {s} + {s}")

        slice_str = ", ".join(slice_parts)
        self.emit(f"{res} = {arr}[{slice_str}]")

    def handle_tile_store(self, op):
        arr = self.get_operand(op, "array")
        idx = self.get_operand(op, "index")
        tile = self.get_operand(op, "tile")

        tile_type_str = op["operands"]["tile"]["type"]["str"]
        shape_match = re.search(r"\(([\d,]+)\)", tile_type_str)
        if shape_match:
            shape_str = shape_match.group(1)
            shape = [int(x) for x in shape_str.split(",")]
        else:
            shape = []

        slice_parts = []
        for i, s in enumerate(shape):
            slice_parts.append(f"{idx}[{i}] * {s} : {idx}[{i}] * {s} + {s}")

        slice_str = ", ".join(slice_parts)
        self.emit(f"{arr}[{slice_str}] = {tile}")

    def handle_assign(self, op):
        res = self.get_result_var(op)
        val = self.get_operand(op, "value")
        self.emit(f"{res} = {val}")

    def handle_tile_reshape(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")

        # Get target shape from result type
        res_type_str = op["result_vars"][0]["type"]["str"]
        shape_match = re.search(r"\(([\d,]+)\)", res_type_str)
        if shape_match:
            shape_str = shape_match.group(1)
            # Tuple string in python needs comma for single element? (32,4,1) is fine.
            # Check if x is a scalar (ArithmeticDType) vs Tile
            x_type = op["operands"]["x"]["type"]["type"]
            is_scalar = x_type == "ArithmeticDType"

            if is_scalar:
                # Scalar needs to be converted to array first
                self.emit(f"{res} = np.full(({shape_str}), {x}, dtype=np.float32)")
            else:
                self.emit(f"{res} = {x}.reshape({shape_str})")

    def handle_tile_permute(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        axes = op["attributes"]["axes"]
        self.emit(f"{res} = {x}.transpose({axes})")

    def handle_tile_broadcast(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")

        # Target shape
        res_type_str = op["result_vars"][0]["type"]["str"]
        shape_match = re.search(r"\(([\d,]+)\)", res_type_str)
        if shape_match:
            shape_str = shape_match.group(1)
            self.emit(f"{res} = np.broadcast_to({x}, ({shape_str}))")

    def handle_tile_cat(self, op):
        res = self.get_result_var(op)
        tiles = self.get_operand(op, "tiles")  # This is a tuple var name
        axis = op["attributes"]["axis"]
        self.emit(f"{res} = np.concatenate({tiles}, axis={axis})")

    def handle_unaryop(self, op):
        res = self.get_result_var(op)
        operand = self.get_operand(op, "operand")
        fn = op["attributes"]["fn"]

        match fn:
            case "abs":
                np_fn = "np.abs"
            case "neg":
                np_fn = "-"
            case "exp":
                np_fn = "np.exp"
            case "exp2":
                np_fn = "np.exp2"
            case "sin":
                np_fn = "np.sin"
            case "cos":
                np_fn = "np.cos"
            case "sinh":
                np_fn = "np.sinh"
            case "cosh":
                np_fn = "np.cosh"
            case "tan":
                np_fn = "np.tan"
            case "tanh":
                np_fn = "np.tanh"
            case "log":
                np_fn = "np.log"
            case "log2":
                np_fn = "np.log2"
            case "sqrt":
                np_fn = "np.sqrt"
            case "rsqrt":
                np_fn = "np.rsqrt"
            case "floor":
                np_fn = "np.floor"
            case "ceil":
                np_fn = "np.ceil"
            case "rsqrt":
                np_fn = "np.rsqrt"
            case "invert":
                np_fn = "~"
            case _:
                raise TypeError(f"Unknown unary op: {fn}")

        self.emit(f"{res} = {np_fn}({operand})")

    def handle_raw_binary_arith(self, op):
        res = self.get_result_var(op)
        lhs = self.get_operand(op, "lhs")
        rhs = self.get_operand(op, "rhs")
        fn = op["attributes"]["fn"]

        match fn:
            case "add":
                self.emit(f"{res} = {lhs} + {rhs}")
            case "sub":
                self.emit(f"{res} = {lhs} - {rhs}")
            case "mul":
                self.emit(f"{res} = {lhs} * {rhs}")
            case "floordiv":
                self.emit(f"{res} = {lhs} // {rhs}")
            case "cdiv":
                # ceil division
                self.emit(f"{res} = np.ceil({lhs} / {rhs}).astype({lhs}.dtype)")
            case "truediv":
                self.emit(f"{res} = {lhs} / {rhs}")
            case "mod":
                self.emit(f"{res} = {lhs} % {rhs}")
            case "pow":
                self.emit(f"{res} = {lhs} ** {rhs}")
            case "min":
                self.emit(f"{res} = np.minimum({lhs}, {rhs})")
            case "max":
                self.emit(f"{res} = np.maximum({lhs}, {rhs})")
            case "c_mod":
                # C-style modulo
                raise NotImplementedError("C-style modulo is not implemented yet")
            case _:
                raise TypeError(f"Unknown binary op: {fn}")

    def handle_tile_reduce(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        fn = op["attributes"]["fn"]
        axis = op["attributes"]["axis"]
        keepdims = op["attributes"]["keepdims"]

        match fn:
            case "add":
                np_fn = "np.sum"
            case "mul":
                np_fn = "np.prod"
            case "max":
                np_fn = "np.max"
            case "min":
                np_fn = "np.min"
            case "argmax":
                np_fn = "np.argmax"
            case "argmin":
                np_fn = "np.argmin"
            case _:
                raise TypeError(f"Unknown reduce op: {fn}")

        self.emit(f"{res} = {np_fn}({x}, axis={axis}, keepdims={keepdims})")

    def handle_tile_mma(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        y = self.get_operand(op, "y")
        acc = self.get_operand(op, "acc")

        # MMA: D = A * B + C
        self.emit(f"{res} = np.matmul({x}, {y}) + {acc}")

    def handle_scalar_to_tile(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        # Just a scalar wrap? Or 0-dim array?
        self.emit(f"{res} = np.array({x})")

    def handle_range(self, op):
        res = self.get_result_var(op)
        start = self.get_operand(op, "start")
        stop = self.get_operand(op, "stop")
        step = self.get_operand(op, "step")
        self.emit(f"{res} = range({start}, {stop}, {step})")

    def handle_fma(self, op):
        res = self.get_result_var(op)
        lhs = self.get_operand(op, "lhs")
        rhs = self.get_operand(op, "rhs")
        acc = self.get_operand(op, "acc")
        self.emit(f"{res} = {lhs} * {rhs} + {acc}")

    def handle_tile_extract(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        idx = self.get_operand(op, "index")  # Tuple of offsets

        shape = op.get("attributes", {}).get("shape", [])

        slice_parts = []
        for i, s in enumerate(shape):
            slice_parts.append(f"{idx}[{i}] * {s} : {idx}[{i}] * {s} + {s}")

        slice_str = ", ".join(slice_parts)
        self.emit(f"{res} = {x}[{slice_str}]")

    def handle_loop(self, op):
        # Result vars of loop are the final values of carried vars
        result_vars = [self.get_var_name(v["name"]) for v in op["result_vars"]]

        # Loop attributes
        attrs = op["attributes"]
        for_loop = attrs.get("for_loop")
        carried_vars = attrs.get("carried_vars")

        loop_var_names = []
        if carried_vars:
            loop_var_names = carried_vars["names"]
            initial_vals = [self.get_var_name(v["name"]) for v in carried_vars["initial"]]
            body_vals = [self.get_var_name(v["name"]) for v in carried_vars["body"]]

            # Initialize the body vars with initial values
            for name, val in zip(body_vals, initial_vals):
                self.emit(f"{name} = {val}")

        # Push loop context
        self.loop_stack.append({"carried_names": loop_var_names, "result_names": result_vars})

        # Loop header
        if for_loop:
            ind_var = self.get_var_name(for_loop["induction_var"]["name"])
            iterable = self.get_var_name(for_loop["iterable"]["name"])
            self.emit(f"for {ind_var} in {iterable}:")
        else:
            self.emit("while True: # Generic loop")

        self.indent_level += 1

        # Loop body
        if op["nested_blocks"]:
            self.process_block(op["nested_blocks"][0])

        self.indent_level -= 1
        self.loop_stack.pop()

        # After loop, assign final values to result vars (only for for-loops that finish naturally)
        if for_loop and carried_vars:
            for res, loop_name in zip(result_vars, body_vals):
                self.emit(f"{res} = {loop_name}")

    def handle_ifelse(self, op):
        cond = self.get_operand(op, "cond")
        result_vars = [self.get_var_name(v["name"]) for v in op["result_vars"]]

        # IfElse has nested_blocks: [true_block, false_block]
        true_block = op["nested_blocks"][0]
        false_block = op["nested_blocks"][1] if len(op["nested_blocks"]) > 1 else None

        # Check if else block is just an end_branch (meaning: do nothing, continue)
        # In this case, we should not generate an else clause EXCEPT if there are result vars
        has_empty_else = False
        if false_block:
            # If else block only has end_branch, treat it as empty
            if len(false_block) == 1 and false_block[0]["op"] == "end_branch":
                has_empty_else = True

        # Check if true block is just an end_branch (meaning: do nothing, continue)
        has_empty_then = False
        if true_block:
            # If then block only has end_branch, treat it as empty
            if len(true_block) == 1 and true_block[0]["op"] == "end_branch":
                has_empty_then = True

        # If there are result_vars and the else block is empty with end_branch,
        # we still need to generate else to handle the assignment
        needs_else_for_results = False
        if result_vars and has_empty_else:
            needs_else_for_results = True

        self.emit(f"if {cond}:")
        self.indent_level += 1

        if has_empty_then:
            self.emit("pass")
        else:
            self.process_block(true_block)

            # Handle assignments from true block
            # Look for end_branch
            if true_block:
                last_op = true_block[-1]
                if last_op["op"] == "end_branch":
                    outputs = self.get_operand(last_op, "outputs")
                    if not isinstance(outputs, list):
                        outputs = [outputs] if outputs else []
                    for res, out in zip(result_vars, outputs):
                        if out:  # Only assign if there's an output
                            self.emit(f"{res} = {out}")
                        else:
                            raise ValueError(f"Missing output for result var {res}")

        self.indent_level -= 1

        # Generate else if: false_block exists AND (it's not empty OR we need it for result vars)
        if false_block and (not has_empty_else or needs_else_for_results):
            self.emit("else:")
            self.indent_level += 1
            if has_empty_else:
                # Empty else block (just end_branch), but we need to handle result vars
                last_op = false_block[-1]
                if last_op["op"] == "end_branch":
                    outputs = self.get_operand(last_op, "outputs")
                    if not isinstance(outputs, list):
                        outputs = [outputs] if outputs else []
                    for res, out in zip(result_vars, outputs):
                        if out:  # Only assign if there's an output
                            self.emit(f"{res} = {out}")
            else:
                self.process_block(false_block)

                # Handle assignments from false block
                last_op = false_block[-1]
                if last_op["op"] == "end_branch":
                    outputs = self.get_operand(last_op, "outputs")
                    if not isinstance(outputs, list):
                        outputs = [outputs] if outputs else []
                    for res, out in zip(result_vars, outputs):
                        if out:  # Only assign if there's an output
                            self.emit(f"{res} = {out}")

            self.indent_level -= 1

    def handle_end_branch(self, op):
        # Handled by parent ifelse
        pass

    def handle_tile_astype(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")

        dtype_attr = op["attributes"].get("dtype")
        if isinstance(dtype_attr, dict):
            dtype_str = dtype_attr.get("str", "float32")  # fallback
        else:
            dtype_str = str(dtype_attr)

        # Clean up dtype str (e.g. "float32" -> "np.float32")
        if "float" in dtype_str:
            np_dtype = f"np.{dtype_str}"
        elif "int" in dtype_str:
            np_dtype = f"np.{dtype_str}"
        else:
            np_dtype = "np.float32"  # Fallback

        self.emit(f"{res} = {x}.astype({np_dtype})")

    def handle_tile_item(self, op):
        res = self.get_result_var(op)
        x = self.get_operand(op, "x")
        self.emit(f"{res} = {x}")  # Pass through, or .item() if scalar needed

    def handle_raw_cmp(self, op):
        res = self.get_result_var(op)
        lhs = self.get_operand(op, "lhs")
        rhs = self.get_operand(op, "rhs")
        fn = op["attributes"]["fn"]

        match fn:
            case "eq":
                op_str = "=="
            case "ne":
                op_str = "!="
            case "lt":
                op_str = "<"
            case "le":
                op_str = "<="
            case "gt":
                op_str = ">"
            case "ge":
                op_str = ">="
            case _:
                raise TypeError(f"Unknown raw cmp op: {fn}")

        self.emit(f"{res} = {lhs} {op_str} {rhs}")

    def handle_continue(self, op):
        # Get next iteration values
        next_vars = self.get_operand(op, "next_vars")
        if not isinstance(next_vars, list):
            next_vars = [next_vars] if next_vars else []

        # Get loop context
        if self.loop_stack:
            carried_names = self.loop_stack[-1]["carried_names"]
            if len(next_vars) == len(carried_names):
                for name, val in zip(carried_names, next_vars):
                    self.emit(f"{name} = {val}")
            else:
                raise ValueError(f"continue vars count mismatch {len(next_vars)} vs {len(carried_names)}")

        self.emit("continue")

    def handle_break(self, op):
        # Get output values
        output_vars = self.get_operand(op, "output_vars")
        if not isinstance(output_vars, list):
            output_vars = [output_vars] if output_vars else []

        # Get loop context
        if self.loop_stack:
            result_names = self.loop_stack[-1]["result_names"]
            if len(output_vars) == len(result_names):
                for name, val in zip(result_names, output_vars):
                    self.emit(f"{name} = {val}")
            else:
                raise ValueError(f"break vars count mismatch {len(output_vars)} vs {len(result_names)}")

        self.emit("break")

    def handle_return(self, op):
        val = self.get_operand(op, "value")
        if val:
            self.emit(f"return {val}")
        else:
            self.emit("return")
