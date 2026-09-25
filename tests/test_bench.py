import itertools
import os

import numpy as np
import pytest

from tspbench import generators as gen
from tspbench.evaluate import evaluate, summarize
from tspbench.instances import Instance, check_tour, pairwise
from tspbench.io import read_text_tsp, read_tsplib, read_tsplib_tour, write_text_tsp, write_tsplib
from tspbench.solvers import Unsupported, make_solver, registry
from tspbench.solvers.concorde import atsp_to_stsp, stsp_tour_to_atsp
from tspbench.solvers.heuristics import held_karp
from tspbench.suites import load_suite, save_references


def brute_force(d):
    n = len(d)
    best = min(itertools.permutations(range(1, n)), key=lambda p: sum(d[a, b] for a, b in zip((0,) + p, p + (0,))))
    return np.array((0,) + best)


def length(d, t):
    return float(sum(d[a, b] for a, b in zip(t, np.roll(t, -1))))


# --------------------------------------------------------------------------- instances


def test_tsplib_metrics_match_definitions():
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.2])
    assert pairwise("euclidean", a, b) == pytest.approx(np.hypot(3, 4.2))
    assert pairwise("EUC_2D", a, b) == 5  # nint(5.18)
    assert pairwise("CEIL_2D", a, b) == 6
    assert pairwise("MAN_2D", a, b) == 7  # nint(7.2)
    assert pairwise("MAX_2D", a, b) == 4
    # ATT: sqrt(25/10)=1.58 -> nint 2, 2 >= 1.58 so 2
    assert pairwise("ATT", np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 2
    assert pairwise("GEO", a, a) == 0


def test_tour_length_and_validation():
    inst = Instance("sq", coords=[[0, 0], [1, 0], [1, 1], [0, 1]])
    assert inst.tour_length([0, 1, 2, 3]) == pytest.approx(4.0)
    assert inst.tour_length([0, 1, 2, 3, 0]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        check_tour([0, 1, 1, 3], 4)
    with pytest.raises(ValueError):
        check_tour([0, 1, 2], 4)


def test_asymmetric_tour_length_uses_direction():
    m = np.array([[0, 1, 9], [9, 0, 1], [1, 9, 0]])
    inst = Instance("a", matrix=m)
    assert not inst.symmetric
    assert inst.tour_length([0, 1, 2]) == 3
    assert inst.tour_length([0, 2, 1]) == 27


# --------------------------------------------------------------------------- generators


def test_kool_generation_matches_legacy_numpy_stream():
    insts = gen.kool_uniform(20, 3)
    np.random.seed(1234)
    ref = np.random.uniform(size=(3, 20, 2))
    assert np.array_equal(np.stack([i.coords for i in insts]), ref)


def test_atsp_tmat_satisfies_triangle_inequality():
    (inst,) = gen.atsp_tmat(12, 1, seed=0)
    m = inst.matrix
    assert not inst.symmetric
    assert np.all(m[:, None, :] <= m[:, :, None] + m[None, :, :] + 1e-12)  # d(i,j) <= d(i,k) + d(k,j)


def test_nonmetric_is_symmetric():
    (inst,) = gen.nonmetric(10, 1, seed=0)
    assert inst.symmetric and np.all(np.diag(inst.matrix) == 0)


# --------------------------------------------------------------------------- io


def test_tsplib_explicit_formats_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    m = np.triu(rng.integers(1, 100, (6, 6)), 1)
    m = m + m.T
    iu = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    lower_diag = [(i, j) for i in range(6) for j in range(i + 1)]
    for fmt, idx in (("UPPER_ROW", iu), ("LOWER_DIAG_ROW", lower_diag)):
        p = tmp_path / f"t_{fmt}.tsp"
        vals = " ".join(str(m[i, j]) for i, j in idx)
        p.write_text(f"NAME: t\nTYPE: TSP\nDIMENSION: 6\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
                     f"EDGE_WEIGHT_FORMAT: {fmt}\nEDGE_WEIGHT_SECTION\n{vals}\nEOF\n")
        assert np.array_equal(read_tsplib(str(p)).matrix, m)


def test_tsplib_coords_and_tour(tmp_path):
    p = tmp_path / "sq.tsp"
    write_tsplib(str(p), "sq", 4, coords=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]), metric="EUC_2D")
    inst = read_tsplib(str(p), optimum=40)
    assert inst.metric == "EUC_2D" and inst.tour_length([0, 1, 2, 3]) == 40
    t = tmp_path / "sq.opt.tour"
    t.write_text("NAME : sq.opt.tour\nTYPE : TOUR\nDIMENSION : 4\nTOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n")
    assert list(read_tsplib_tour(str(t))) == [0, 1, 2, 3]


def test_atsp_file_with_big_diagonal(tmp_path):
    p = tmp_path / "a.atsp"
    p.write_text("NAME: a\nTYPE: ATSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
                 "EDGE_WEIGHT_SECTION\n9999999 1 9\n9 9999999 1\n1 9 9999999\nEOF\n")
    inst = read_tsplib(str(p))
    assert not inst.symmetric and inst.tour_length([0, 1, 2]) == 3


def test_text_format_roundtrip(tmp_path):
    insts = gen.kool_uniform(10, 2)
    tours = [np.arange(10), np.arange(10)[::-1]]
    p = tmp_path / "x.txt"
    write_text_tsp(str(p), insts, tours)
    back = read_text_tsp(str(p))
    assert len(back) == 2
    assert np.allclose(back[0].coords, insts[0].coords)
    assert back[1].meta["reference_length"] == pytest.approx(insts[1].tour_length(tours[1]))


# --------------------------------------------------------------------------- solvers


@pytest.mark.parametrize("seed", range(3))
def test_held_karp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    d = rng.random((7, 7))
    np.fill_diagonal(d, 0)
    assert length(d, held_karp(d)) == pytest.approx(length(d, brute_force(d)))


def test_atsp_transformation_preserves_optimum():
    rng = np.random.default_rng(1)
    d = rng.integers(1, 50, (6, 6))
    np.fill_diagonal(d, 0)
    s = atsp_to_stsp(d)
    assert np.array_equal(s, s.T)
    tour = stsp_tour_to_atsp(held_karp(s.astype(float)), 6)
    assert length(d, tour) == pytest.approx(length(d, held_karp(d.astype(float))))


CHEAP = ["nearest_neighbor", "nearest_insertion", "farthest_insertion", "random_insertion",
         "cheapest_insertion", "two_opt", "two_opt:init=farthest_insertion", "exact_dp"]


@pytest.mark.parametrize("spec", CHEAP)
@pytest.mark.parametrize("family", ["uniform", "atsp", "nonmetric", "manhattan"])
def test_heuristics_return_valid_tours(spec, family):
    inst = load_suite(f"{family}9:num=1").instances[0]
    try:
        tour = make_solver(spec).solve(inst, seed=0)
    except Unsupported:
        assert spec.startswith("two_opt") and not inst.symmetric
        return
    t = check_tour(tour, inst.n)
    opt = inst.tour_length(held_karp(inst.full_matrix()))
    assert inst.tour_length(t) >= opt - 1e-9
    if spec == "exact_dp":
        assert inst.tour_length(t) == pytest.approx(opt)


def test_two_opt_neighbor_list_is_2opt_optimal_and_improves():
    inst = gen.kool_uniform(200, 1)[0]
    init = make_solver("nearest_neighbor").solve(inst, 0)
    from tspbench.solvers.heuristics import two_opt_dense, two_opt_neighbors

    t_nb = two_opt_neighbors(inst, init, k=199)
    t_dense = two_opt_dense(inst.full_matrix(), init)
    assert inst.tour_length(t_nb) < inst.tour_length(init)
    # with the full candidate list no improving 2-opt move remains
    assert inst.tour_length(two_opt_dense(inst.full_matrix(), t_nb)) == pytest.approx(inst.tour_length(t_nb))
    assert inst.tour_length(t_dense) < inst.tour_length(init)


def test_heatmap_adapter_and_callable():
    inst = gen.kool_uniform(30, 1)[0]
    walk = make_solver("heatmap:fn=tspbench.solvers.model:inverse_distance,decode=greedy_walk").solve(inst)
    from tspbench.solvers.heuristics import nearest_neighbor

    assert np.array_equal(walk, nearest_neighbor(inst, 0))
    edge = make_solver("heatmap:fn=tspbench.solvers.model:inverse_distance").solve(inst)
    check_tour(edge, 30)
    cal = make_solver("callable:fn=tests.test_bench:identity_solver").solve(inst)
    assert np.array_equal(cal, np.arange(30))


def identity_solver(matrix):
    return np.arange(len(matrix))


def _have(name):
    return registry()[name].available()[0]


@pytest.mark.skipif(not _have("lkh"), reason="elkai / LKH not installed")
def test_lkh_is_optimal_on_small_instances():
    for family in ("uniform", "atsp", "nonmetric"):
        inst = load_suite(f"{family}10:num=1").instances[0]
        opt = inst.tour_length(held_karp(inst.full_matrix()))
        assert inst.tour_length(make_solver("lkh").solve(inst, seed=0)) == pytest.approx(opt, rel=1e-4)


@pytest.mark.skipif(not _have("ortools"), reason="ortools not installed")
def test_ortools_valid_on_symmetric_and_asymmetric():
    for family in ("uniform", "atsp"):
        inst = load_suite(f"{family}10:num=1").instances[0]
        opt = inst.tour_length(held_karp(inst.full_matrix()))
        t = make_solver("ortools:time_limit=0.2").solve(inst)
        assert inst.tour_length(t) <= opt * 1.05


@pytest.mark.skipif(not _have("concorde"), reason="Concorde not installed")
def test_concorde_is_optimal():
    for family in ("uniform", "atsp"):
        inst = load_suite(f"{family}10:num=1").instances[0]
        opt = inst.tour_length(held_karp(inst.full_matrix()))
        assert inst.tour_length(make_solver("concorde").solve(inst)) == pytest.approx(opt, rel=1e-4)


# --------------------------------------------------------------------------- harness


def test_evaluate_and_summaries(tmp_path):
    suite = load_suite("uniform8:num=4")
    save_references(str(tmp_path), suite.name, "exact_dp",
                    {i.name: i.tour_length(held_karp(i.full_matrix())) for i in suite.instances})
    recs = evaluate([suite], ["exact_dp", "random_insertion", "two_opt"], seeds=[0, 1, 2],
                    data_dir=str(tmp_path), progress=False)
    assert len(recs) == 4 * (1 + 3 + 3)  # two_opt's default init is stochastic
    rows = {r["solver"]: r for r in summarize(recs)}
    assert rows["exact_dp"]["gap_mean"] == pytest.approx(0.0, abs=1e-12)
    assert rows["exact_dp"]["reference"] == "exact_dp"
    assert rows["random_insertion"]["ci_over"] == "seeds"
    assert rows["random_insertion"]["gap_mean"] >= 0
    assert all(r["gap_best"] >= -1e-12 for r in recs)


def test_evaluate_records_unsupported():
    suite = load_suite("atsp8:num=2")
    recs = evaluate([suite], ["two_opt:init=farthest_insertion"], data_dir="/nonexistent", progress=False)
    assert {r["status"] for r in recs} == {"unsupported"}
    assert summarize(recs)[0]["gap_mean"] is None


def test_suite_specs():
    assert len(load_suite("tsp20", limit=5).instances) == 5
    assert load_suite("tsp500_gen", limit=2).instances[0].n == 500
    assert load_suite("clustered50:num=3,num_clusters=5").instances[0].meta["num_clusters"] == 5
    with pytest.raises(FileNotFoundError):
        load_suite("tsp500", data_dir="/nonexistent")
    with pytest.raises(KeyError):
        load_suite("nope")
