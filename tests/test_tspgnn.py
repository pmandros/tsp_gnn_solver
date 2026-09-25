import numpy as np
import pytest
import torch

from tspgnn import instances
from tspgnn.graph import EDGE_DIM, NODE_DIM, build_graph, collate
from tspgnn.model import TSPGNN
from tspgnn.solve import solve_gnn, solve_greedy_distance, solve_nearest_neighbor
from tspgnn.tours import is_valid_tour, knn_lists, lkh_tour, tour_length, two_opt


@pytest.mark.parametrize("kind", list(instances.GENERATORS))
def test_instances_are_symmetric_matrices(kind):
    d = instances.generate(kind, 30, np.random.default_rng(0))
    assert d.shape == (30, 30)
    assert np.allclose(d, d.T)
    assert np.all(np.diag(d) == 0)


def test_features_are_scale_invariant():
    d = instances.generate("euclidean", 50, np.random.default_rng(1))
    g1, g2 = build_graph(d), build_graph(d * 1234.5)
    assert np.array_equal(g1.edge_index, g2.edge_index)
    assert np.allclose(g1.edge_attr, g2.edge_attr, atol=1e-5)
    assert np.allclose(g1.x, g2.x, atol=1e-5)


def test_labels_are_symmetric():
    d = instances.generate("random", 40, np.random.default_rng(2))
    g = build_graph(d, tour=lkh_tour(d))
    m = g.edge_index.shape[1] // 2
    assert np.array_equal(g.y[:m], g.y[m:])
    assert g.y.sum() == 2 * 40  # every tour edge covered, in both directions


def test_model_scores_are_symmetric_and_size_agnostic():
    torch.manual_seed(0)
    model = TSPGNN(NODE_DIM, EDGE_DIM, hidden=16, layers=2).eval()
    rng = np.random.default_rng(3)
    graphs = [build_graph(instances.generate("euclidean", n, rng)) for n in (10, 37, 80)]
    batch = collate(graphs)
    with torch.no_grad():
        logits = model(batch)
        alone = model(collate(graphs[1:2]))
    assert torch.allclose(logits, logits[batch["rev"]])
    # A graph's scores do not depend on what else is in the batch.
    off = graphs[0].edge_index.shape[1]
    assert torch.allclose(logits[off:off + alone.numel()], alone, atol=1e-5)


def test_two_opt_never_worsens():
    rng = np.random.default_rng(4)
    for kind in ("euclidean", "random"):
        d = instances.generate(kind, 60, rng)
        start = rng.permutation(60)
        out = two_opt(d, start, knn_lists(d, 20))
        assert is_valid_tour(out, 60)
        assert tour_length(d, out) <= tour_length(d, start) + 1e-9


def test_solvers_return_valid_tours():
    torch.manual_seed(0)
    model = TSPGNN(NODE_DIM, EDGE_DIM, hidden=16, layers=2).eval()
    rng = np.random.default_rng(5)
    for n in (5, 21, 150):
        d = instances.generate("clustered", n, rng)
        for tour in (solve_gnn(model, d), solve_gnn(model, d, use_two_opt=False),
                     solve_greedy_distance(d), solve_nearest_neighbor(d)):
            assert is_valid_tour(tour, n)


@pytest.mark.parametrize("name", ["solve", "solve_without_two_opt", "solve_distance_greedy"])
def test_api_entry_points_return_valid_tours(name):
    from tspgnn import api

    d = instances.generate("euclidean", 40, np.random.default_rng(3))
    assert is_valid_tour(getattr(api, name)(d), 40)
