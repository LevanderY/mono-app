from mono2048 import cli, vision

PRED = [4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CLEAN = [4, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
WRONG = [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
UNKNOWNS = [(3, (10, 20, 30)), (4, (11, 22, 33))]
OVERLAY = [(i, (0, 0, 0)) for i in range(13)]


class _Dev:
    def screencap(self):
        return object()


def _run(monkeypatch, frames):
    seq = {"i": -1}

    def fake(img, calib):
        seq["i"] = min(seq["i"] + 1, len(frames) - 1)
        return frames[seq["i"]]

    monkeypatch.setattr(vision, "read_board", fake)
    return cli.read_fast(_Dev(), {}, PRED, timeout=0.3, interval=0.001)


def test_animation_then_clean_is_accepted(monkeypatch):
    img, board, unknowns = _run(monkeypatch, [(WRONG, UNKNOWNS), (CLEAN, [])])
    assert img is not None
    assert board == CLEAN
    assert unknowns == []


def test_clean_frame_immediately(monkeypatch):
    img, board, unknowns = _run(monkeypatch, [(CLEAN, [])])
    assert board == CLEAN


def test_overlay_triggers_fallback(monkeypatch):
    img, _, _ = _run(monkeypatch, [(WRONG, OVERLAY)])
    assert img is None


def test_never_settles_times_out_to_fallback(monkeypatch):
    img, _, _ = _run(monkeypatch, [(WRONG, [])])
    assert img is None


def test_match_with_unknown_colours_is_rejected(monkeypatch):
    img, _, _ = _run(monkeypatch, [(CLEAN, UNKNOWNS)])
    assert img is None
