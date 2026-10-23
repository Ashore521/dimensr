"""Conservative floating-point interval screening of raw expression trees.

Subdivision can resolve interval dependency, but has a strict box budget.
An unresolved box is never reported as enclosed. This is numerical screening,
not a formally verified interval arithmetic or theorem-proving implementation.
"""
from fractions import Fraction
import numpy as np


def audit_box(tree, constants, bounds, *, max_boxes=32, output_bounds=None):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("bounds must be finite ordered [lower, upper] pairs.")
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes < 1:
        raise ValueError("max_boxes must be a positive integer.")
    constants = np.asarray(constants, dtype=float)
    if constants.shape != (tree.n_constants,) or not np.isfinite(constants).all():
        raise ValueError("Invalid constants for interval audit.")
    if any(int(n.token[1:]) >= len(bounds) for _, n in tree.walk() if n.token.startswith("x")):
        raise ValueError("Input bound count does not cover the tree's variables.")

    def enclosure(box):
        values = iter(constants)
        problem = None
        variables = set()

        def visit(node):
            nonlocal problem, variables
            token = node.token
            if not node.children:
                if token.startswith("x"):
                    return tuple(box[int(token[1:])])
                c = next(values)
                return c, c
            args = [visit(c) for c in node.children]
            if problem is not None:
                return -np.inf, np.inf
            a, b = args[0]
            if token in ("div", "log", "sqrt"):
                lo, hi = args[1] if token == "div" else args[0]
                if token == "div" and lo <= 0 <= hi or token == "log" and lo <= 0 or token == "sqrt" and lo < 0:
                    problem = token + "_domain_unresolved"
                    child = node.children[1] if token == "div" else node.children[0]
                    variables = {int(n.token[1:]) for _, n in child.walk() if n.token.startswith("x")}
                    return -np.inf, np.inf
            if len(args) == 2:
                c, d = args[1]
                if token == "add":
                    lo, hi = a+c, b+d
                elif token == "sub":
                    lo, hi = a-d, b-c
                else:
                    products = [a*c, a*d, b*c, b*d] if token == "mul" else [a/c, a/d, b/c, b/d]
                    lo, hi = min(products), max(products)
            elif token == "neg":
                return -b, -a
            elif token == "square":
                lo, hi = (0. if a <= 0 <= b else min(a*a, b*b)), max(a*a, b*b)
            elif token == "sqrt":
                lo, hi = np.sqrt(a), np.sqrt(b)
            elif token == "log":
                lo, hi = np.log(a), np.log(b)
            elif token == "exp":
                lo, hi = np.exp(a), np.exp(b)
            else:
                # A wide but safe enclosure avoids argument-reduction assumptions.
                return -1., 1.
            if not np.isfinite([lo, hi]).all():
                problem = "nonfinite_enclosure"
                variables = {int(n.token[1:]) for _, n in node.walk() if n.token.startswith("x")}
                return -np.inf, np.inf
            lower, upper = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
            if len(args) == 2 and (lo == 0 or hi == 0):
                # Distinguish exact cancellation/zero from floating underflow.
                fa, fb, fc, fd = map(Fraction, (a, b, c, d))
                exact = ([fa+fc, fb+fd] if token == "add" else [fa-fd, fb-fc] if token == "sub" else
                         [fa*fc, fa*fd, fb*fc, fb*fd] if token == "mul" else [fa/fc, fa/fd, fb/fc, fb/fd])
                if lo == 0 and min(exact) >= 0:
                    lower = 0.
                if hi == 0 and max(exact) <= 0:
                    upper = 0.
            if token in ("square", "sqrt", "exp") and lo == 0:
                lower = 0.
            if token == "log":
                if a == 1:
                    lower = 0.
                if b == 1:
                    upper = 0.
            if not np.isfinite([lower, upper]).all():
                problem = "nonfinite_enclosure"
                variables = {int(n.token[1:]) for _, n in node.walk() if n.token.startswith("x")}
            return lower, upper

        with np.errstate(all="ignore"):
            lower, upper = visit(tree)
        if problem is None and output_bounds is not None:
            lo, hi = output_bounds
            if lo is not None and lower < lo or hi is not None and upper > hi:
                problem = "output_bound_unresolved"
                variables = {int(n.token[1:]) for _, n in tree.walk() if n.token.startswith("x")}
        return problem, variables

    pending = [bounds.copy()]
    checked = 0
    reason = None
    original_width = np.maximum(bounds[:, 1]-bounds[:, 0], np.finfo(float).tiny)
    while pending and checked < max_boxes:
        box = pending.pop()
        checked += 1
        reason, variables = enclosure(box)
        if reason is None:
            continue
        center = box[:, 0]/2 + box[:, 1]/2
        probes = [center]
        for i in sorted(variables):
            for endpoint in box[i]:
                point = center.copy()
                point[i] = endpoint
                probes.append(point)
        predictions = tree.evaluate(np.asarray(probes), constants)
        bad = ~np.isfinite(predictions)
        if output_bounds is not None:
            lo, hi = output_bounds
            if lo is not None:
                bad |= predictions < lo
            if hi is not None:
                bad |= predictions > hi
        if bad.any():
            return dict(status="invalid", boxes_checked=checked, reason=reason,
                        witness=np.asarray(probes)[np.flatnonzero(bad)[0]].tolist())
        splittable = [i for i in variables if box[i, 0] < center[i] < box[i, 1]]
        if not splittable:
            return dict(status="unresolved", boxes_checked=checked, reason=reason, witness=None)
        axis = max(splittable, key=lambda i: (box[i, 1]-box[i, 0])/original_width[i])
        left, right = box.copy(), box.copy()
        left[axis, 1] = right[axis, 0] = center[axis]
        pending.extend((right, left))
    return dict(status="unresolved" if pending else "enclosed", boxes_checked=checked,
                reason="box_budget" if pending else None, witness=None)
