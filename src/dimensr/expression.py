"""Immutable typed expression trees. No eval, protected division or hidden scaling."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .units import ZERO, add, sub, scale, encode, unit

ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "neg": 1,
         "square": 1, "sqrt": 1, "sin": 1, "cos": 1, "exp": 1, "log": 1}


@dataclass(frozen=True)
class Node:
    token: str
    children: tuple = ()
    units: tuple = ZERO

    def __post_init__(self):
        if self.token in ARITY:
            if len(self.children) != ARITY[self.token]:
                raise ValueError("Incorrect operator arity.")
            dims = [c.units for c in self.children]
            if self.token in ("add", "sub"):
                if dims[0] != dims[1]:
                    raise ValueError("Addition/subtraction requires equal units.")
                expected = dims[0]
            elif self.token == "mul":
                expected = add(*dims)
            elif self.token == "div":
                expected = sub(*dims)
            elif self.token == "square":
                expected = scale(dims[0], 2)
            elif self.token == "sqrt":
                expected = scale(dims[0], Fraction(1, 2))
            elif self.token == "neg":
                expected = dims[0]
            else:
                if dims[0] != ZERO:
                    raise ValueError("Transcendental functions require dimensionless input.")
                expected = ZERO
            if expected != self.units:
                raise ValueError("Operator output units do not match its children.")
        elif not (self.token[0:1] in ("x", "c") and self.token[1:].isdigit()):
            raise ValueError(f"Unknown token: {self.token}")
        elif self.children:
            raise ValueError("Terminals cannot have children.")

    @property
    def size(self):
        return 1 + sum(c.size for c in self.children)

    @property
    def depth(self):
        return 1 + max((c.depth for c in self.children), default=0)

    def walk(self, path=()):
        yield path, self
        for i, child in enumerate(self.children):
            yield from child.walk(path + (i,))

    @property
    def n_constants(self):
        return sum(n.token.startswith("c") and not n.children for _, n in self.walk())

    @property
    def n_operator_types(self):
        return len({n.token for _, n in self.walk() if n.children})

    def replace(self, path, replacement):
        if not path:
            if self.units != replacement.units:
                raise ValueError("Subtree replacement must preserve units.")
            return replacement
        children = list(self.children)
        children[path[0]] = children[path[0]].replace(path[1:], replacement)
        return Node(self.token, tuple(children), self.units)

    def evaluate(self, X, constants):
        it = iter(constants)

        def visit(node):
            if not node.children:
                return X[:, int(node.token[1:])] if node.token.startswith("x") else next(it)
            a = visit(node.children[0])
            if len(node.children) == 2:
                b = visit(node.children[1])
                return {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                        "div": np.divide}[node.token](a, b)
            return {"neg": np.negative, "square": np.square, "sqrt": np.sqrt,
                    "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log}[node.token](a)

        if len(constants) != self.n_constants:
            raise ValueError("Incorrect number of constants.")
        with np.errstate(all="ignore"):
            result = np.broadcast_to(visit(self), (len(X),)).copy()
        return result

    def sympy(self, constants, feature_names=None):
        import sympy as sp
        it = iter(constants)

        def visit(n):
            if not n.children:
                if n.token.startswith("c"):
                    return sp.Float(float(next(it)), 16)
                i = int(n.token[1:])
                return sp.Symbol(feature_names[i] if feature_names is not None else n.token)
            args = [visit(c) for c in n.children]
            return {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
                    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
                    "neg": lambda a: -a, "square": lambda a: a ** 2,
                    "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
                    "exp": sp.exp, "log": sp.log}[n.token](*args)
        return visit(self)

    def to_dict(self):
        return {"token": self.token, "units": encode(self.units),
                "children": [c.to_dict() for c in self.children]}

    def canonical(self, constants=None):
        """Order commutative operands and reorder their independent coefficients.

        No cancellation, reassociation or domain-changing identities are used.
        """
        it = iter(np.ones(self.n_constants) if constants is None else constants)

        def visit(node):
            if not node.children:
                return node, [next(it)] if node.token.startswith("c") else []
            parts = [visit(child) for child in node.children]
            if node.token in ("add", "mul"):
                parts.sort(key=lambda pair: repr(pair[0]))
            return (Node(node.token, tuple(p[0] for p in parts), node.units),
                    [v for _, values in parts for v in values])

        root, values = visit(self)
        return root, np.asarray(values, dtype=float)

    @classmethod
    def from_dict(cls, data):
        return cls(data["token"], tuple(cls.from_dict(c) for c in data["children"]), unit(data["units"]))
