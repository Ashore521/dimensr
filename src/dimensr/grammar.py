"""Finite typed grammar with exact completion costs and explicit search bounds."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .expression import ARITY, Node
from .units import ZERO, add, sub, scale


@dataclass(frozen=True)
class Production:
    token: str
    children: tuple = ()


@dataclass
class Trace:
    tokens: list
    parents: list
    contexts: list
    masks: list


class Grammar:
    def __init__(self, feature_units, target_units, constant_units, operators,
                 max_nodes=15, max_depth=6, extra_units=()):
        self.target = target_units
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.terminals = {f"x{i}": u for i, u in enumerate(feature_units)}
        self.terminals.update({f"c{i}": u for i, u in enumerate(constant_units)})
        self.tokens = tuple(self.terminals) + tuple(operators)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.bos = len(self.tokens)
        base = set(self.terminals.values()) | {ZERO, target_units} | set(extra_units)
        pool = set(base)
        for a in base:
            pool.update(scale(a, k) for k in (-2, -1, 2, Fraction(1, 2)))
            for b in base:
                pool.update((add(a, b), sub(a, b)))
        if len(pool) > 256:
            raise ValueError("More than 256 unit types; reduce features or unit types for this bounded grammar.")
        self.units = tuple(sorted(pool))
        self.productions = {u: [] for u in self.units}
        for t, u in self.terminals.items():
            self.productions[u].append(Production(t))
        for t in operators:
            for a in self.units:
                if ARITY[t] == 2:
                    for b in self.units:
                        if t in ("add", "sub"):
                            if a != b:
                                continue
                            out = a
                        else:
                            out = add(a, b) if t == "mul" else sub(a, b)
                        if out in pool:
                            self.productions[out].append(Production(t, (a, b)))
                else:
                    if t in ("sin", "cos", "log", "exp"):
                        if a != ZERO:
                            continue
                        out = ZERO
                    else:
                        out = scale(a, {"square": 2, "sqrt": Fraction(1, 2), "neg": 1}[t])
                    if out in pool:
                        self.productions[out].append(Production(t, (a,)))
        # cost[d][u] = minimum nodes for a tree of units u and depth <= d.
        impossible = max_nodes + 1
        self.cost = [{u: impossible for u in self.units}]
        for depth in range(1, max_depth + 1):
            prev = self.cost[-1]
            self.cost.append({u: min([impossible] + [1 + sum(prev[c] for c in p.children)
                                                    for p in ps])
                              for u, ps in self.productions.items()})
        if self.cost[-1][target_units] > max_nodes:
            raise ValueError("Target units cannot be generated within the grammar bounds. "
                             "Supply suitable constant_units/extra_units or increase max_nodes/max_depth.")
        self._options = {}

    def options(self, units, budget, depth):
        key = (units, budget, depth)
        if key not in self._options:
            by_token = {}
            if depth > 0:
                for p in self.productions[units]:
                    if 1 + sum(self.cost[depth - 1][c] for c in p.children) <= budget:
                        by_token.setdefault(self.index[p.token], []).append(p)
            self._options[key] = by_token
        return self._options[key]

    def generate(self, rng, policy=None, units=None, budget=None, depth=None):
        units = self.target if units is None else units
        budget = self.max_nodes if budget is None else budget
        depth = self.max_depth if depth is None else depth
        trace = Trace([], [], [], [])
        hidden = None

        def visit(required, remaining, height, parent):
            nonlocal hidden
            choices = self.options(required, remaining, height)
            if not choices:
                raise ValueError("No feasible production for subtree.")
            mask = np.zeros(len(self.tokens), dtype=bool)
            mask[list(choices)] = True
            context = [float(x) for x in required] + [remaining / self.max_nodes, height / self.max_depth]
            previous = trace.tokens[-1] if trace.tokens else self.bos
            if policy is None:
                token = int(rng.choice(list(choices)))
            else:
                token, hidden = policy.sample_step(previous, parent, context, mask, hidden)
            p = choices[token][int(rng.integers(len(choices[token])))]
            trace.tokens.append(token)
            trace.parents.append(parent)
            trace.contexts.append(context)
            trace.masks.append(mask)
            children = []
            remaining -= 1
            for i, child_units in enumerate(p.children):
                reserved = sum(self.cost[height - 1][u] for u in p.children[i + 1:])
                child = visit(child_units, remaining - reserved, height - 1, token)
                remaining -= child.size
                children.append(child)
            return Node(p.token, tuple(children), required)

        return visit(units, budget, depth, self.bos), trace

    def trace(self, tree):
        result = Trace([], [], [], [])

        def visit(node, budget, depth, parent):
            choices = self.options(node.units, budget, depth)
            token = self.index[node.token]
            p = Production(node.token, tuple(c.units for c in node.children))
            if p not in choices.get(token, ()):
                raise ValueError("Tree cannot be replayed under this grammar.")
            mask = np.zeros(len(self.tokens), dtype=bool)
            mask[list(choices)] = True
            result.tokens.append(token)
            result.parents.append(parent)
            result.contexts.append([float(u) for u in node.units] +
                                   [budget / self.max_nodes, depth / self.max_depth])
            result.masks.append(mask)
            remaining = budget - 1
            for i, child in enumerate(node.children):
                reserved = sum(self.cost[depth - 1][c.units] for c in node.children[i + 1:])
                visit(child, remaining - reserved, depth - 1, token)
                remaining -= child.size
        visit(tree, self.max_nodes, self.max_depth, self.bos)
        return result
