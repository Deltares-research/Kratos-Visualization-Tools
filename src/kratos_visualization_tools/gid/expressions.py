"""Restricted arithmetic expressions evaluated on aligned nodal fields."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .calculations import compute_nodal_gradient, tensor_component, vector_component
from .models import ResultHeader, TENSOR_COMPONENTS, VECTOR_COMPONENTS


FUNCTIONS = ("abs", "mag", "min", "max", "grad", "grad_x", "grad_y")
TENSOR_ATTRIBUTES = TENSOR_COMPONENTS + ("zz", "yz", "zy", "xz", "zx")
Array = NDArray[np.float64]


class FieldExpression:
    def __init__(self, text: str, headers: Sequence[ResultHeader]):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("Enter a formula of at most 2000 characters.")
        try:
            self.tree = ast.parse(text, mode="eval")
        except (SyntaxError, RecursionError) as error:
            raise ValueError(f"Invalid formula: {error}") from error
        if len(list(ast.walk(self.tree))) > 200:
            raise ValueError("The formula is too complex.")
        self.text = text
        self.names = {header.name.casefold(): header.name for header in headers}
        self.dependencies: set[str] = set()
        self._validate(self.tree.body)
        if not self.dependencies:
            raise ValueError("The formula must reference at least one result field.")
        self.locations = {
            name: ("OnNodes" if any(h.name == name and h.location == "OnNodes"
                                  for h in headers) else "OnGaussPoints")
            for name in self.dependencies
        }
        sets = [{h.step for h in headers if h.name == name and
                 h.location == self.locations[name]} for name in self.dependencies]
        self.steps = sorted(set.intersection(*sets))

    def _field(self, name: str) -> str:
        if name.casefold() not in self.names:
            raise ValueError(f"Unknown result field '{name}'. Choose a name from the input list.")
        result = self.names[name.casefold()]
        self.dependencies.add(result)
        return result

    def _validate(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only finite numeric constants are supported.")
            try:
                finite = np.isfinite(float(node.value))
            except OverflowError as error:
                raise ValueError("Numeric constant is too large.") from error
            if not finite:
                raise ValueError("Only finite numeric constants are supported.")
        elif isinstance(node, ast.Name):
            self._field(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            self._field(node.value.id)
            if node.attr not in TENSOR_ATTRIBUTES + VECTOR_COMPONENTS:
                raise ValueError(f"Unsupported component '{node.attr}'.")
        elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = node.func.id
            if function not in FUNCTIONS or node.keywords:
                raise ValueError("Supported functions: " + ", ".join(FUNCTIONS))
            count = 2 if function in ("min", "max") else 1
            if len(node.args) != count:
                raise ValueError(f"{function} requires {count} argument(s).")
            for argument in node.args:
                self._validate(argument)
        else:
            raise ValueError("Use fields, components, numbers, + - * /, and supported functions.")

    def evaluate(
        self, node_ids: Sequence[int],
        loader: Callable[[str, str], tuple[str, dict[int, Array]]],
        tri,
    ) -> dict[int, Array]:
        fields = {}
        for name in self.dependencies:
            result_type, values = loader(name, self.locations[name])
            size = next((np.asarray(value).size for value in values.values()), 0)
            if not size:
                raise ValueError(f"No values for input field '{name}'.")
            array = np.full((len(node_ids), size), np.nan)
            for index, node_id in enumerate(node_ids):
                if node_id in values:
                    value = np.asarray(values[node_id], dtype=float).ravel()
                    if value.size != size:
                        raise ValueError(f"Inconsistent component counts in '{name}'.")
                    array[index] = value
            fields[name] = result_type.lower(), array
        valid = np.ones(len(node_ids), dtype=bool)
        for _, array in fields.values():
            valid &= np.isfinite(array).all(axis=1)
        if not valid.any():
            raise ValueError("The input fields have no common finite nodal values.")

        def field_value(name, component=None):
            result_type, array = fields[self.names[name.casefold()]]
            if component is not None:
                components = (TENSOR_ATTRIBUTES if result_type == "matrix"
                              else VECTOR_COMPONENTS if result_type == "vector" else ())
                if component not in components:
                    raise ValueError(f"Component '{component}' is not available for '{name}'.")
                reducer = tensor_component if result_type == "matrix" else vector_component
                return np.asarray([reducer(row, component) for row in array])
            if result_type == "matrix":
                raise ValueError(f"Select a tensor component, e.g. {name}.xx or {name}.trace.")
            return array[:, 0] if result_type == "scalar" else array

        def scalar(value):
            if np.ndim(value) > 1:
                raise ValueError("Select a vector component or use abs(vector) / mag(vector).")
            return np.broadcast_to(value, (len(node_ids),)).copy()

        def evaluate(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                return field_value(node.id)
            if isinstance(node, ast.Attribute):
                return field_value(node.value.id, node.attr)
            if isinstance(node, ast.UnaryOp):
                value = scalar(evaluate(node.operand))
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp):
                left, right = scalar(evaluate(node.left)), scalar(evaluate(node.right))
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if np.any(right[valid] == 0):
                    raise ValueError("Division by zero in the formula.")
                return np.divide(left, right, out=np.full_like(left, np.nan),
                                 where=valid)
            function = node.func.id
            value = evaluate(node.args[0])
            if function in ("abs", "mag"):
                return np.linalg.norm(value, axis=1) if np.ndim(value) == 2 else np.abs(value)
            if function in ("min", "max"):
                operation = np.minimum if function == "min" else np.maximum
                return operation(scalar(value), scalar(evaluate(node.args[1])))
            if tri is None:
                raise ValueError("Gradient functions require a valid mesh triangulation.")
            value = scalar(value)
            value[~valid] = np.nan
            gx, gy = compute_nodal_gradient(tri, value)
            return gx if function == "grad_x" else gy if function == "grad_y" else np.hypot(gx, gy)

        try:
            with np.errstate(over="raise", invalid="ignore"):
                result = scalar(evaluate(self.tree.body))
        except FloatingPointError as error:
            raise ValueError("Numeric overflow in the formula.") from error
        if np.any(np.isinf(result)) or not np.isfinite(result[valid]).any():
            raise ValueError("The formula produced no finite values or an infinite result.")
        return {node_id: np.array([result[index]])
                for index, node_id in enumerate(node_ids)
                if valid[index] and np.isfinite(result[index])}


def calculated_headers(headers: Sequence[ResultHeader], definitions: dict[str, str]):
    original = [header for header in headers if header.header_line >= 0]
    result = list(original)
    for name, formula in definitions.items():
        if (not name.isidentifier() or name in FUNCTIONS or
                name.casefold() in {header.name.casefold() for header in original}):
            raise ValueError(f"Invalid or conflicting calculated-field name '{name}'.")
        try:
            tree = ast.parse(formula, mode="eval")
        except (SyntaxError, RecursionError) as error:
            raise ValueError(f"Invalid formula for '{name}': {error}") from error
        functions = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        inputs = {node.id.casefold() for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and id(node) not in functions}
        available = {header.name.casefold() for header in original}
        if not inputs <= available:
            placeholder = [ResultHeader(field, 0, "Scalar", "OnNodes", None, 0)
                           for field in inputs]
            FieldExpression(formula, placeholder)
            continue
        expression = FieldExpression(formula, original)
        result.extend(ResultHeader(name, step, "Scalar", "OnNodes", None, -1)
                      for step in expression.steps)
    return result
