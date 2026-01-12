from typing import Any
from cuda.tile._ir import ir
from cuda.tile._ir.type import Type


def serialize_function(func: ir.Function) -> dict[str, Any]:
    """Serialize a Function to a dictionary."""
    return {
        "qualname": func.qualname,
        "parameters": [_serialize_var(p) for p in func.parameters],
        "return_value": _serialize_var(func.return_value) if func.return_value else None,
        "loc": _serialize_loc(func.loc),
        "operations": _serialize_block(func.root_block),
    }


def _serialize_block(block: ir.Block) -> list[dict[str, Any]]:
    """Serialize a Block to a list of operation dictionaries."""
    return [_serialize_operation(op) for op in block.operations]


def _serialize_operation(op: ir.Operation) -> dict[str, Any]:
    """Serialize an Operation to a dictionary."""
    # Special handling for Loop operations
    if op.op == "loop":
        attrs = {}
        # Serialize for_loop (ForLoopInfo) if present
        if hasattr(op, "for_loop") and op.for_loop is not None:
            attrs["for_loop"] = {
                "induction_var": _serialize_var(op.for_loop.induction_var),
                "iterable": _serialize_var(op.for_loop.iterable),
            }
        # Serialize carried_vars (CarriedVariables) if present
        if hasattr(op, "carried_vars") and op.carried_vars is not None:
            attrs["carried_vars"] = {
                "names": op.carried_vars.names,
                "initial": [_serialize_var(v) for v in op.carried_vars.initial],
                "body": [_serialize_var(v) for v in op.carried_vars.body],
                "results": [_serialize_var(v) for v in op.carried_vars.results],
            }
        return {
            "op": op.op,
            "result_vars": [_serialize_var(v) for v in op.result_vars],
            "operands": {k: _serialize_operand(v) for k, v in op.operands.items()},
            "attributes": attrs,
            "loc": _serialize_loc(op.loc),
            "nested_blocks": [_serialize_block(b) for b in op.nested_blocks],
        }

    return {
        "op": op.op,
        "result_vars": [_serialize_var(v) for v in op.result_vars],
        "operands": {k: _serialize_operand(v) for k, v in op.operands.items()},
        "attributes": _serialize_attributes(op.attributes),
        "loc": _serialize_loc(op.loc),
        "nested_blocks": [_serialize_block(b) for b in op.nested_blocks],
    }


def _serialize_var(var: ir.Var) -> dict[str, Any]:
    """Serialize a Var to a dictionary."""
    ty = var.try_get_type()
    return {
        "name": var.name,
        "type": _serialize_type(ty) if ty is not None else None,
        "is_constant": var.is_constant(),
        "constant_value": _serialize_constant(var.get_constant()) if var.is_constant() else None,
    }


def _serialize_operand(value) -> dict[str, Any] | None:
    """Serialize an operand value (Var or tuple of Vars)."""
    if value is None:
        return None
    if isinstance(value, ir.Var):
        return _serialize_var(value)
    elif isinstance(value, tuple):
        return [_serialize_var(v) for v in value]
    else:
        return str(value)


def _serialize_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """Serialize operation attributes."""
    result = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            result[k] = v
        elif isinstance(v, (int, float, bool)):
            result[k] = v
        elif isinstance(v, tuple):
            result[k] = list(v)
        elif isinstance(v, type):
            result[k] = v.__name__
        else:
            result[k] = str(v)
    return result


def _serialize_type(ty: Type) -> dict[str, Any]:
    """Serialize a Type to a dictionary."""
    return {
        "type": ty.__class__.__name__,
        "str": str(ty),
    }


def _serialize_loc(loc) -> dict[str, Any] | None:
    """Serialize a Loc (source location) to a dictionary."""
    if loc is None:
        return None
    # Try to get filename, line, column if available
    try:
        return {
            "filename": loc.filename if hasattr(loc, "filename") else None,
            "line": loc.lineno if hasattr(loc, "lineno") else None,
            "column": loc.col_offset if hasattr(loc, "col_offset") else None,
        }
    except Exception:
        return {"unknown": True}


def _serialize_constant(value: Any) -> Any:
    """Serialize a constant value."""
    if isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, tuple):
        return [_serialize_constant(v) for v in value]
    elif isinstance(value, list):
        return [_serialize_constant(v) for v in value]
    else:
        return str(value)
