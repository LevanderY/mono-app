import pickle

import numpy as np
from numba import njit

from mono2048 import engine as app

K = 8
TLEN = 6

_LUT = None
_SYM = None
_NST = 5

_TUPLES = ((0, 1, 2, 3, 4, 5), (4, 5, 6, 7, 8, 9), (0, 1, 2, 4, 5, 6),
           (4, 5, 6, 8, 9, 10), (0, 1, 4, 5, 8, 9), (1, 2, 5, 6, 9, 10),
           (0, 1, 2, 4, 8, 12), (5, 6, 7, 9, 10, 11))


def _build_sym():
    def rc(i):
        return divmod(i, 4)

    def rot(i):
        r, c = rc(i)
        return c * 4 + (3 - r)

    def mir(i):
        r, c = rc(i)
        return r * 4 + (3 - c)

    fns = [lambda i: i, rot, lambda i: rot(rot(i)), lambda i: rot(rot(rot(i))),
           mir, lambda i: rot(mir(i)), lambda i: rot(rot(mir(i))),
           lambda i: rot(rot(rot(mir(i))))]
    return [[f(i) for i in range(16)] for f in fns]


def load(path):
    global _LUT, _SYM, _NST
    with open(path, "rb") as f:
        d = pickle.load(f)
    _LUT = np.ascontiguousarray(d["lut"], dtype=np.float32)
    _NST = int(d.get("nstages", _LUT.shape[0]))
    tuples = d.get("tuples", _TUPLES)
    perms = _build_sym()
    sym = np.zeros((len(tuples), 8, TLEN), np.int64)
    for k, t in enumerate(tuples):
        for s in range(8):
            for j in range(TLEN):
                sym[k, s, j] = perms[s][t[j]]
    _SYM = sym


@njit(cache=True)
def _V(arr, LUT, SYM, nst):
    st = 0
    for k in range(16):
        if arr[k] == 10:
            st += 1
    if st >= nst:
        st = nst - 1
    s = 0.0
    for k in range(SYM.shape[0]):
        for sm in range(8):
            idx = 0
            for j in range(6):
                idx |= arr[SYM[k, sm, j]] << (4 * j)
            s += LUT[st, k, idx]
    return s


def _arr(board):
    a = np.empty(16, np.int8)
    for r in range(4):
        row = board[r]
        a[4 * r] = (row >> 12) & 0xF
        a[4 * r + 1] = (row >> 8) & 0xF
        a[4 * r + 2] = (row >> 4) & 0xF
        a[4 * r + 3] = row & 0xF
    return a


def _val(board):
    return float(_V(_arr(board), _LUT, _SYM, _NST))


def _bank_reward(board):
    best = 0
    for _, fn in app._BMOVES:
        nb, r = fn(board)
        if nb != board and r > best:
            best = r
    return float(best)


_TT = {}
_SAMPLE = 6


def _chance(after, depth, cap_exp):
    key = (after, depth, 0)
    v = _TT.get(key)
    if v is not None:
        return v
    empties = app._empty_idx(after)
    n = len(empties)
    if n == 0:
        r = _val(after)
    else:
        if n > _SAMPLE:
            import random
            empties = random.sample(empties, _SAMPLE)
        total = 0.0
        for rk in empties:
            total += 0.9 * _search_max(app._set_cell(after, rk, 1), depth, cap_exp)
            total += 0.1 * _search_max(app._set_cell(after, rk, 2), depth, cap_exp)
        r = total / len(empties)
    _TT[key] = r
    return r


def _search_max(board, depth, cap_exp):
    key = (board, depth, 1)
    v = _TT.get(key)
    if v is not None:
        return v
    best = -1e18
    for _, fn in app._BMOVES:
        nb, r = fn(board)
        if nb == board or app._max_exp(nb) > cap_exp:
            continue
        val = (r + _val(nb)) if depth <= 1 else (r + _chance(nb, depth - 1, cap_exp))
        if val > best:
            best = val
    if best <= -1e17:
        best = _bank_reward(board)
    _TT[key] = best
    return best


def get_best_move(values, cap=1024, depth=4):
    if _LUT is None:
        raise RuntimeError("LUT not loaded — call terminal.load(path)")
    cap_exp = (cap.bit_length() - 1) if cap else 10
    board = app._pack(values)
    _TT.clear()
    best, bv = None, -1e18
    for name, fn in app._BMOVES:
        nb, r = fn(board)
        if nb == board or app._max_exp(nb) > cap_exp:
            continue
        val = (r + _val(nb)) if depth <= 1 else (r + _chance(nb, depth - 1, cap_exp))
        if val > bv:
            bv, best = val, name
    return best
