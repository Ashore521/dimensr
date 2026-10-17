"""Unit-preserving GP operators; no target data used here."""
import numpy as np
from .expression import Node


def mutate(tree, grammar, rng):
    path, new = mutation_site(tree, grammar, rng)
    return tree.replace(path, new)


def mutation_site(tree, grammar, rng):
    nodes = list(tree.walk())
    path, old = nodes[int(rng.integers(len(nodes)))]
    budget = grammar.max_nodes - tree.size + old.size
    depth = grammar.max_depth - len(path)
    new, _ = grammar.generate(rng, units=old.units, budget=budget, depth=depth)
    return path, new


def crossover(tree, donor, grammar, rng):
    path, donor_path, replacement = crossover_site(tree, donor, grammar, rng)
    return tree.replace(path, replacement)


def crossover_site(tree, donor, grammar, rng):
    nodes = list(tree.walk())
    path, old = nodes[int(rng.integers(len(nodes)))]
    candidates = [(p, node) for p, node in donor.walk()
                  if node.units == old.units
                  and node != old
                  and tree.size - old.size + node.size <= grammar.max_nodes
                  and len(path) + node.depth <= grammar.max_depth]
    if not candidates:
        return (), (), tree
    donor_path, replacement = candidates[int(rng.integers(len(candidates)))]
    return path, donor_path, replacement


def coefficient_slice(tree, path):
    start = 0
    for p, node in tree.walk():
        if p == path:
            return slice(start, start + node.n_constants)
        if not node.children and node.token.startswith("c"):
            start += 1
    raise ValueError("Unknown subtree path")


def fitted_offspring(parent, donor, grammar, rng, *, cross=False, term=None):
    """Vary a tree while carrying coefficients of unchanged/donated subtrees."""
    if term is not None and "add" in grammar.tokens:
        tree = Node("add", (parent.tree, term.tree), parent.tree.units)
        if tree.size <= grammar.max_nodes and tree.depth <= grammar.max_depth:
            return tree, np.concatenate((parent.constants, term.constants)), "add_term"
    if cross:
        path, donor_path, replacement = crossover_site(parent.tree, donor.tree, grammar, rng)
        if replacement == parent.tree and not path:
            return parent.tree, parent.constants.copy(), "crossover"
        inherited = donor.constants[coefficient_slice(donor.tree, donor_path)]
        operator = "crossover"
    else:
        path, replacement = mutation_site(parent.tree, grammar, rng)
        inherited = np.ones(replacement.n_constants)
        operator = "mutation"
    interval = coefficient_slice(parent.tree, path)
    constants = np.concatenate((parent.constants[:interval.start], inherited,
                                parent.constants[interval.stop:]))
    return parent.tree.replace(path, replacement), constants, operator
