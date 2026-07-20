import random

from mono2048 import engine


def _rand_board(rng):
    vals = []
    for _ in range(16):
        if rng.random() < 0.35:
            vals.append(0)
        else:
            vals.append(1 << rng.randint(1, 10))
    return vals


def test_pack_unpack_roundtrip():
    rng = random.Random(0)
    for _ in range(5000):
        b = _rand_board(rng)
        assert engine._unpack(engine._pack(b)) == b


def test_transpose_involution_and_parity():
    rng = random.Random(1)
    for _ in range(5000):
        b = _rand_board(rng)
        board = engine._pack(b)
        assert engine._transpose_b(engine._transpose_b(board)) == board
        assert engine._unpack(engine._transpose_b(board)) == engine._transpose(b)


def test_fast_moves_match_reference():
    rng = random.Random(2)
    pairs = (
        (engine.move_left, engine._mv_left),
        (engine.move_right, engine._mv_right),
        (engine.move_up, engine._mv_up),
        (engine.move_down, engine._mv_down),
    )
    for _ in range(5000):
        b = _rand_board(rng)
        board = engine._pack(b)
        for vfn, bfn in pairs:
            vb, vs = vfn(b)
            nb, ns = bfn(board)
            assert engine._unpack(nb) == vb
            assert ns == vs


def test_best_bank_move_2048_requires_forming_2048():
    no_2048 = [2, 4, 0, 0] + [0] * 12
    assert engine.best_bank_move_2048(no_2048) == (None, 0)
    forms = [1024, 1024, 0, 0] + [0] * 12
    name, gain = engine.best_bank_move_2048(forms)
    assert name in engine.MOVES
    assert gain == 2048
