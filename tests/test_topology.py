import numpy as np

from prma.topology import positions_2d, propagation_delays


def test_legacy_square_shell_position_order() -> None:
    expected = np.array(
        [[1, 1], [2, 1], [2, 2], [1, 2], [3, 1], [3, 2], [3, 3], [2, 3], [1, 3]],
        dtype=float,
    )
    np.testing.assert_array_equal(positions_2d(9), expected)


def test_delays_are_ceiled_scaled_euclidean_distances() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(propagation_delays(positions, 2.0), [[0, 3], [3, 0]])
