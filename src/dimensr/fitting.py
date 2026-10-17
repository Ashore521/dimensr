"""Coefficient initialization and exact affine subproblems; no protected operators."""
import numpy as np
from scipy.optimize import lsq_linear


def affine_constants(tree):
    """Whether the expression is jointly affine in all independent constants."""
    if not tree.children:
        return True
    if tree.token in ("add", "sub", "neg"):
        return all(affine_constants(child) for child in tree.children)
    if tree.token == "mul":
        a, b = tree.children
        return (a.n_constants == 0 and affine_constants(b) or
                b.n_constants == 0 and affine_constants(a))
    if tree.token == "div":
        return tree.children[1].n_constants == 0 and affine_constants(tree.children[0])
    return tree.n_constants == 0


def linear_solution(tree, X, y, bound, evaluate):
    """Solve only structurally affine trees, retaining the original program/domain."""
    if not tree.n_constants or not affine_constants(tree):
        return None
    zero = np.zeros(tree.n_constants)
    base = evaluate(zero)
    columns = [evaluate(row) - base for row in np.eye(tree.n_constants)]
    matrix = np.column_stack(columns)
    if not np.isfinite(base).all() or not np.isfinite(matrix).all():
        return None
    scales = np.linalg.norm(matrix, axis=0)
    scales[scales == 0] = 1.0
    scaled = matrix / scales
    solution = np.linalg.lstsq(scaled, y - base, rcond=None)[0] / scales
    if np.any(np.abs(solution) > bound):
        solution = lsq_linear(scaled, y - base, bounds=(-bound * scales, bound * scales)).x / scales
    return solution


def repair_domains(tree, X, constants, bound):
    """Repair direct constant offsets of log/sqrt/div, without changing their semantics.

    This is a feasible-start heuristic, not an interval-domain proof. More complex
    domains are covered by bounded multiscale restarts and may still be rejected.
    """
    c = np.asarray(constants, dtype=float).copy()
    offsets = {}
    cursor = 0
    for path, node in tree.walk():
        offsets[path] = cursor
        if not node.children and node.token.startswith("c"):
            cursor += 1

    def values(node, path):
        start = offsets[path]
        return node.evaluate(X, c[start:start + node.n_constants])

    for _ in range(2):
        for path, node in reversed(list(tree.walk())):
            if node.token not in ("log", "sqrt", "div"):
                continue
            arg_index = 1 if node.token == "div" else 0
            arg = node.children[arg_index]
            arg_path = path + (arg_index,)
            v = values(arg, arg_path)
            if not np.isfinite(v).all():
                continue
            if (node.token == "div" and np.all(np.abs(v) > 1e-8) or
                    node.token != "div" and np.all(v > 0)):
                continue
            if not arg.children and arg.token.startswith("c"):
                c[offsets[arg_path]] = min(1.0, bound * 0.5)
            elif arg.token in ("add", "sub"):
                for i, leaf in enumerate(arg.children):
                    if leaf.children or not leaf.token.startswith("c"):
                        continue
                    index = offsets[arg_path + (i,)]
                    sign = -1 if arg.token == "sub" and i == 1 else 1
                    margin = max(float(np.std(v)), 1e-3 * max(1.0, float(np.max(np.abs(v)))))
                    c[index] = np.clip(c[index] + sign * (margin - float(v.min())), -bound, bound)
                    break
    return c
