from prma.encoding import cycle_edges, encode_cycle
from prma.metrics import directed_edge_jaccard


def test_encoding_is_stable_and_simple() -> None:
    first = encode_cycle(42, node_count=20, path_length=8, seed=7)
    second = encode_cycle(42, node_count=20, path_length=8, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 8


def test_cycle_edges_include_closing_transition() -> None:
    assert cycle_edges((2, 5, 1)) == ((2, 5), (5, 1), (1, 2))


def test_directed_overlap_distinguishes_direction() -> None:
    assert directed_edge_jaccard((0, 1, 2), (0, 2, 1)) == 0.0
