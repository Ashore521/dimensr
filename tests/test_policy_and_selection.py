import numpy as np
import torch
from dimensr.units import ZERO
from dimensr.expression import Node
from dimensr.grammar import Grammar
from dimensr.policy import Policy
from dimensr.pareto import fronts, crowding, selection_order, size_density
from dimensr.complexity import local_rademacher


def test_pareto_fronts_duplicates_and_crowding():
    values = [[1, 4], [2, 3], [3, 2], [4, 1], [3, 4], [1, 4]]
    assert set(fronts(values)[0]) == {0, 1, 2, 3, 5}
    assert fronts(values)[1] == [4]
    assert np.isinf(crowding(values[:4])[[0, 3]]).all()
    assert np.isfinite(crowding([[1, 1], [1, 1], [1, 1]])).all()
    np.testing.assert_allclose(size_density([1, 1, 2, 5], 1), [1, 1, 1, 1 / 3])


def test_size_diversity_changes_truncation_and_keeps_boundaries():
    values = [[i, 8 - i] for i in range(9)]
    sizes = [2, 2, 2, 2, 2, 2, 15, 20, 25]
    base = selection_order(values, sizes, False)
    diverse = selection_order(values, sizes, True)
    assert set(diverse[:2]) == {0, 8}
    assert diverse != base
    assert diverse.index(6) < diverse.index(2)


def test_policy_updates_and_replays_sampling_masks():
    g = Grammar([ZERO], ZERO, [ZERO], ("add", "mul"), 7, 4)
    p = Policy(len(g.tokens), seed=3)
    rng = np.random.default_rng(4)
    traces = []
    for _ in range(12):
        tree, trace = g.generate(rng, p)
        replay = g.trace(tree)
        assert trace.tokens == replay.tokens
        np.testing.assert_array_equal(trace.masks, replay.masks)
        traces.append(trace)
    before = torch.cat([x.detach().flatten().clone() for x in p.parameters()])
    metrics = p.update(traces, np.linspace(0, 1, 12), traces[-2:])
    after = torch.cat([x.detach().flatten() for x in p.parameters()])
    assert metrics["gradient_norm"] > 0
    assert not torch.equal(before, after)
    assert np.isfinite(list(metrics.values())).all()


def test_policy_init_does_not_reset_global_rng():
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    Policy(4, seed=10)
    assert torch.equal(state, torch.get_rng_state())


def test_rademacher_is_zero_for_fixed_family_and_sensitive_to_radius():
    x = Node("x0")
    tree = Node("mul", (Node("c0"), x))
    X = np.linspace(-1, 1, 60).reshape(-1, 1)
    rng = np.random.default_rng(2)
    signs = rng.choice([-1, 1], (50, 60))
    directions = np.array([[1], [-1]])
    fixed = local_rademacher(x, [], X, 1.0, signs, directions)
    small = local_rademacher(tree, [2], X, 1.0, signs, directions, 0.01)
    large = local_rademacher(tree, [2], X, 1.0, signs, directions, 0.1)
    assert fixed == 0
    assert 0 < small < large <= 1
