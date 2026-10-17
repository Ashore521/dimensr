from fractions import Fraction
import numpy as np
import pytest
from dimensr.units import ZERO, unit
from dimensr.expression import Node
from dimensr.grammar import Grammar
from dimensr.genetics import mutate, crossover

L = unit([1, 0, 0, 0, 0, 0, 0])
T = unit([0, 0, 1, 0, 0, 0, 0])
V = unit([1, 0, -1, 0, 0, 0, 0])


def test_units_and_invalid_operations():
    x, t = Node("x0", units=L), Node("x1", units=T)
    assert Node("div", (x, t), V).units == V
    with pytest.raises(ValueError, match="equal units"):
        Node("add", (x, t), L)
    with pytest.raises(ValueError, match="dimensionless"):
        Node("sin", (x,), ZERO)
    assert unit(["1/2", 0, 0, 0, 0, 0, 0])[0] == Fraction(1, 2)
    with pytest.raises(ValueError):
        unit([1, 2])
    with pytest.raises(ValueError):
        unit([np.nan] * 7)


def test_generation_and_gp_preserve_units_and_budgets():
    g = Grammar([L, T], V, [ZERO], ("mul", "div", "add", "square", "sqrt"), 13, 5)
    rng = np.random.default_rng(8)
    trees = [g.generate(rng)[0] for _ in range(60)]
    for original in trees:
        for tree in (original, mutate(original, g, rng), crossover(original, trees[0], g, rng)):
            assert tree.units == V
            assert tree.size <= 13
            assert tree.depth <= 5
            trace = g.trace(tree)
            assert len(trace.tokens) == tree.size
            assert all(mask[t] for t, mask in zip(trace.tokens, trace.masks))


def test_infeasible_grammar_fails_early():
    with pytest.raises(ValueError, match="cannot be generated"):
        Grammar([L], T, [ZERO], ("add", "mul"), 7, 3)


def test_free_constants_keep_declared_dimensions():
    g = Grammar([T], L, [V], ("mul",), 3, 2)
    tree, _ = g.generate(np.random.default_rng(4))
    assert tree.units == L
    np.testing.assert_allclose(tree.evaluate(np.array([[1.0], [3.0]]), [2.0]), [2, 6])


def test_numeric_domains_are_not_silently_protected():
    x = Node("x0")
    tree = Node("div", (x, Node("c0")))
    assert not np.isfinite(tree.evaluate(np.array([[1.0]]), [0.0])).all()
    assert np.isnan(Node("log", (x,)).evaluate(np.array([[-1.0]]), [])[0])


def test_tree_json_preserves_fractional_units():
    root = Node("sqrt", (Node("x0", units=L),), unit(["1/2", 0, 0, 0, 0, 0, 0]))
    assert Node.from_dict(root.to_dict()) == root

