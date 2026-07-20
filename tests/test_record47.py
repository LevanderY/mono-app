import pathlib
import random

import pytest

from mono2048 import engine
from mono2048.brains import record47
from mono2048.brains import terminal

MODEL = pathlib.Path(__file__).resolve().parents[1] / "models" / "nt3_warm_2m.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="model models/nt3_warm_2m.pkl not present (see models/README.md)",
)

P1 = [1024, 512, 256, 128,
      1024, 256, 128, 64,
      32, 16, 8, 4,
      2, 0, 0, 0]
P1b = [512, 256, 128, 64,
       1024, 128, 64, 32,
       64, 32, 16, 8,
       4, 2, 0, 0]
P2 = [1024, 512, 256, 128,
      512, 256, 128, 64,
      1024, 32, 16, 8,
      1024, 4, 2, 0]
P2open = [1024, 512, 256, 0,
          1024, 128, 64, 0,
          1024, 32, 16, 0,
          8, 4, 0, 0]
GATE = [1024, 1024, 256, 128,
        1024, 1024, 64, 32,
        512, 256, 16, 8,
        128, 64, 4, 2]


@pytest.fixture(scope="module")
def brain():
    record47.load(str(MODEL))
    record47.get_best_move([2, 0, 0, 0] + [0] * 12, score=0.0)
    return record47


def _n1024(b):
    return sum(1 for v in b if v == 1024)


@pytest.mark.parametrize("board", [P1, P1b])
def test_phase1_matches_terminal_brain(brain, board):
    assert _n1024(board) < record47.ACT1024
    random.seed(123)
    got = brain.get_best_move(board, score=0.0)
    random.seed(123)
    want = terminal.get_best_move(board, cap=1024, depth=record47.D1)
    assert got == want


def test_gate_banks_only_when_qualified(brain):
    name, gain = engine.best_bank_move_2048(GATE)
    after, _ = engine.MOVES[name](GATE)
    assert max(after) >= 2048
    assert gain == 4096

    assert brain._bank_qualifies(GATE, 43000.0) is True
    assert brain.get_best_move(GATE, score=43000.0) is None

    assert brain._bank_qualifies(GATE, 42000.0) is False


@pytest.mark.parametrize("board", [P2, P2open])
def test_phase2_moves_are_legal_and_capped(brain, board):
    assert _n1024(board) >= record47.ACT1024
    random.seed(7)
    mv = brain.get_best_move(board, score=44000.0)
    if mv is not None:
        assert mv in engine.MOVES
        after, _ = engine.MOVES[mv](board)
        assert max(after) < 2048
