"""Bounded typed feature composition and sparse additive/rational starting trees.

All ranking and coefficient initialization use training data only. These proposals
still pass the ordinary evaluator's budget, coefficient fit and domain checks.
"""
from fractions import Fraction
import numpy as np
from .expression import Node
from .units import ZERO, add, sub, scale


def operator(token, *children):
    units = [c.units for c in children]
    out = (add(*units) if token == "mul" else sub(*units) if token == "div" else
           scale(units[0], 2) if token == "square" else scale(units[0], Fraction(1, 2))
           if token == "sqrt" else ZERO if token in ("sin", "cos", "exp", "log") else units[0])
    return Node(token, tuple(children), out)


def structured_seeds(grammar, X, y, limit):
    constants = {u: Node(t, units=u) for t, u in grammar.terminals.items() if t.startswith("c")}
    features, seen = [], set()
    counts = dict(feature_evaluations=0, linear_subproblems=0, proposed_structures=0)
    if limit <= 0:
        return [], counts

    def admit(tree, values=None):
        if tree.size > min(9, grammar.max_nodes) or tree.depth > grammar.max_depth:
            return None
        tree, c = tree.canonical(values)
        key = tree, tuple(c)
        if key in seen or len(features) >= 512:
            return None
        seen.add(key)
        if tree.units not in grammar.units:
            return None
        # Intermediate terms need not have the target units, but must be replayable.
        try:
            grammar.trace(tree)
        except (ValueError, KeyError):
            return None
        counts["feature_evaluations"] += 1
        v = tree.evaluate(X, c)
        with np.errstate(all="ignore"):
            norm = np.linalg.norm(v)
        if not np.isfinite(v).all() or not np.isfinite(norm) or norm < 1e-12:
            return None
        feature = (tree, c, v)
        features.append(feature)
        return feature

    variables = []
    for token, units in grammar.terminals.items():
        if token.startswith("x"):
            feature = admit(Node(token, units=units))
            if feature is not None:
                variables.append(feature)
    primitive = list(variables)
    for tree, c, _ in variables:
        for token in ("square", "sin", "cos", "log", "sqrt", "exp"):
            if token not in grammar.tokens or token in ("sin", "cos", "log", "exp") and tree.units != ZERO:
                continue
            feature = admit(operator(token, tree), c)
            if feature is not None:
                primitive.append(feature)
    for i, (a, ca, _) in enumerate(variables):
        for j, (b, cb, _) in enumerate(variables):
            for token in ("mul", "div", "add"):
                if token not in grammar.tokens or token in ("mul", "add") and j < i:
                    continue
                if token == "add" and (a.units != b.units or i == j) or token == "div" and i == j:
                    continue
                feature = admit(operator(token, a, b), np.r_[ca, cb])
                if feature is not None:
                    primitive.append(feature)
    # Function compositions and products help preserve useful multi-node building blocks.
    for tree, c, _ in primitive:
        if tree.size > 4:
            continue
        for token in ("square", "sin", "cos", "log", "sqrt"):
            if token in grammar.tokens and (tree.units == ZERO or token in ("square", "sqrt")):
                admit(operator(token, tree), c)
        if tree.units == ZERO and ZERO in constants and "add" in grammar.tokens:
            for token in ("log", "sqrt"):
                if token not in grammar.tokens:
                    continue
                admit(operator(token, operator("add", constants[ZERO], tree)), np.r_[1., c])
                if "mul" in grammar.tokens:
                    inner = operator("add", constants[ZERO], operator("mul", constants[ZERO], tree))
                    for sign in (-1., 1.):
                        admit(operator(token, inner), np.r_[1., sign, c])
    # Squared dimensionless ratios are useful inside many guarded unary functions.
    for tree, c, _ in list(features):
        if tree.token == "square" and tree.units == ZERO and tree.size <= 4 and ZERO in constants:
            if "add" in grammar.tokens and "mul" in grammar.tokens:
                inner = operator("add", constants[ZERO], operator("mul", constants[ZERO], tree))
                for token in ("sqrt", "log"):
                    if token in grammar.tokens:
                        for sign in (-1., 1.):
                            admit(operator(token, inner), np.r_[1., sign, c])
    composable = [f for f in features if f[0].size <= 3]
    for i, (a, ca, _) in enumerate(composable):
        for b, cb, _ in composable[i:]:
            if "mul" in grammar.tokens and a.size + b.size <= 6:
                admit(operator("mul", a, b), np.r_[ca, cb])

    proposals = {}

    def balanced(terms):
        if len(terms) == 1:
            return terms[0]
        mid = len(terms) // 2
        a, ca = balanced(terms[:mid])
        b, cb = balanced(terms[mid:])
        return operator("add", a, b), np.r_[ca, cb]

    def fit_terms(target, target_units, denominator=None):
        if not np.isfinite(target).all():
            return
        available = [f for f in features if sub(target_units, f[0].units) in constants]
        if not available or "mul" not in grammar.tokens:
            return
        intercept = constants.get(target_units) if "add" in grammar.tokens else None
        matrix = np.column_stack([f[2] for f in available])
        base = np.ones((len(y), 1)) if intercept is not None else np.empty((len(y), 0))
        # Only the best 48 training correlations enter the sparse beam.
        residual = target - target.mean() if intercept is not None else target
        centered = matrix - matrix.mean(axis=0) if intercept is not None else matrix
        correlation = np.abs(centered.T @ residual) / np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
        indices = np.argsort(-correlation, kind="stable")[:48]
        beam = [()]
        for _ in range(3 if "add" in grammar.tokens else 1):
            fits = {}
            for selected in beam:
                for j in indices:
                    choice = tuple(sorted(set(selected) | {int(j)}))
                    if len(choice) != len(selected) + 1 or choice in fits:
                        continue
                    design = np.column_stack((base, matrix[:, choice]))
                    norms = np.linalg.norm(design, axis=0)
                    norms[norms == 0] = 1.
                    counts["linear_subproblems"] += 1
                    try:
                        coef = np.linalg.lstsq(design / norms, target, rcond=None)[0] / norms
                    except np.linalg.LinAlgError:
                        continue
                    estimate = design @ coef
                    prediction = estimate if denominator is None else estimate / denominator[2]
                    error = float(np.mean((prediction - y) ** 2))
                    terms = [(intercept, np.array([coef[0]]))] if intercept is not None else []
                    for index, weight in zip(choice, coef[len(base.T):]):
                        term, c, _ = available[index]
                        multiplier = constants[sub(target_units, term.units)]
                        terms.append((operator("mul", multiplier, term), np.r_[weight, c]))
                    tree, c = balanced(terms)
                    if denominator is not None:
                        tree, c = operator("div", tree, denominator[0]), np.r_[c, denominator[1]]
                    if tree.size > grammar.max_nodes or tree.depth > grammar.max_depth or not np.isfinite(error):
                        continue
                    tree, c = tree.canonical(c)
                    try:
                        grammar.trace(tree)
                    except (ValueError, KeyError):
                        continue
                    fits[choice] = error
                    score = error / max(float(np.var(y)), 1e-12) + 1e-8 * tree.size
                    if tree not in proposals or score < proposals[tree][0]:
                        proposals[tree] = score, c
            beam = sorted(fits, key=lambda k: (fits[k], k))[:3]
            if not beam:
                break

    fit_terms(y, grammar.target)
    if "div" in grammar.tokens:
        denominators = [f for f in features if f[0].size <= 3 or
                        f[0].token in ("sqrt", "log") and f[0].size <= 9]
        for denominator in denominators[:64]:
            if np.all(np.abs(denominator[2]) > 1e-12):
                fit_terms(y * denominator[2], add(grammar.target, denominator[0].units), denominator)
    counts["proposed_structures"] = len(proposals)
    ranked = sorted(proposals, key=lambda tree: (proposals[tree][0], tree.size, repr(tree)))[:limit]
    return [(tree, proposals[tree][1]) for tree in ranked], counts
