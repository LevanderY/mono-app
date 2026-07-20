import importlib.util
import pathlib

import numpy as np

_EVAL_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_script", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expmax_returns_terminal_bank_not_sentinel():
    ev = _load_evaluate()
    RL, RLS, RR, RRS = ev.build_rows()
    SYM = ev.build_sym()
    LUT = np.zeros((ev.NSTAGES, ev.K, 16), np.float32)
    board = np.array(
        [10, 10, 10, 10,
         1, 2, 1, 2,
         2, 1, 2, 1,
         1, 2, 1, 2], np.int8)
    value = ev._expmax(board, 3, 6, RL, RLS, RR, RRS, LUT, SYM)
    assert value == 4096.0
