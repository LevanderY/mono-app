import pathlib

import pytest

from mono2048 import engine
from mono2048.brains import terminal

MODEL = pathlib.Path(__file__).resolve().parents[1] / "models" / "nt3_warm_2m.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="model models/nt3_warm_2m.pkl not present (see models/README.md)",
)


@pytest.fixture(scope="module")
def brain():
    terminal.load(str(MODEL))
    terminal.get_best_move([2, 0, 0, 0] + [0] * 12, cap=1024, depth=3)
    return terminal


def test_returns_legal_capped_move(brain):
    board = [1024, 512, 256, 128,
             1024, 256, 128, 64,
             32, 16, 8, 4,
             2, 0, 0, 0]
    mv = brain.get_best_move(board, cap=1024, depth=3)
    assert mv in engine.MOVES
    after, _ = engine.MOVES[mv](board)
    assert max(after) < 2048


def test_terminal_bank_reward_is_real_not_sentinel(brain):
    packed = engine._pack([1024, 1024, 0, 0] + [0] * 12)
    reward = brain._bank_reward(packed)
    assert reward == 2048.0
    assert reward < 1e17


def test_returns_none_when_every_move_forms_2048(brain):
    full = [1024, 1024, 512, 256,
            256, 512, 1024, 1024,
            1024, 1024, 512, 256,
            256, 512, 1024, 1024]
    assert brain.get_best_move(full, cap=1024, depth=3) is None
