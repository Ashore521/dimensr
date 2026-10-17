"""Small, typed additive starting structures; all coefficients use training data."""
from .expression import Node
from .units import ZERO, sub


def typed_seeds(grammar, rng):
    def coefficient(units):
        return next((Node(t, units=u) for t, u in grammar.terminals.items()
                     if t.startswith("c") and u == units), None)

    def weighted(node):
        c = coefficient(sub(grammar.target, node.units))
        if c is not None and "mul" in grammar.tokens:
            return Node("mul", (c, node), grammar.target)
        return node if node.units == grammar.target else None

    def summation(terms):
        if len(terms) == 1:
            return terms[0]
        mid = len(terms) // 2
        return Node("add", (summation(terms[:mid]), summation(terms[mid:])), grammar.target)

    features = [Node(t, units=u) for t, u in grammar.terminals.items() if t.startswith("x")]
    raw = [term for x in features if (term := weighted(x)) is not None]
    bases = list(raw)
    for x in features:
        inverse = coefficient(sub(ZERO, x.units))
        normalized = (Node("mul", (inverse, x), ZERO)
                      if inverse is not None and "mul" in grammar.tokens else None)
        for token in ("square", "sqrt", "log", "exp"):
            if token not in grammar.tokens:
                continue
            inner = x if x.units == ZERO and token != "exp" else normalized
            if inner is not None:
                term = weighted(Node(token, (inner,), ZERO))
                if term is not None:
                    bases.append(term)
    if "div" in grammar.tokens:
        for a in features:
            for b in features:
                if a != b and a.units == b.units:
                    term = weighted(Node("div", (a, b), ZERO))
                    if term is not None:
                        bases.append(term)
    intercept = coefficient(grammar.target)
    proposals = ([intercept] if intercept is not None else []) + bases
    if "add" in grammar.tokens:
        if intercept is not None:
            proposals += [summation([intercept, term]) for term in bases]
        for _ in range(3):
            shuffled = [raw[i] for i in rng.permutation(len(raw))]
            for n in range(2, len(shuffled) + 1):
                proposals.append(summation(([intercept] if intercept is not None else []) + shuffled[:n]))
    seen = set()
    for tree in proposals:
        if tree.size > grammar.max_nodes or tree.depth > grammar.max_depth:
            continue
        tree, _ = tree.canonical()
        if tree in seen:
            continue
        try:
            grammar.trace(tree)
        except ValueError:
            continue
        seen.add(tree)
        yield tree
