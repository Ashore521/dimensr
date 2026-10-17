"""Unit-preserving GP operators; no target data used here."""


def mutate(tree, grammar, rng):
    nodes = list(tree.walk())
    path, old = nodes[int(rng.integers(len(nodes)))]
    budget = grammar.max_nodes - tree.size + old.size
    depth = grammar.max_depth - len(path)
    new, _ = grammar.generate(rng, units=old.units, budget=budget, depth=depth)
    return tree.replace(path, new)


def crossover(tree, donor, grammar, rng):
    nodes = list(tree.walk())
    path, old = nodes[int(rng.integers(len(nodes)))]
    candidates = [node for _, node in donor.walk()
                  if node.units == old.units
                  and tree.size - old.size + node.size <= grammar.max_nodes
                  and len(path) + node.depth <= grammar.max_depth]
    if not candidates:
        return tree
    return tree.replace(path, candidates[int(rng.integers(len(candidates)))])
