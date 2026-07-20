import math
import random

from mono2048 import engine as app
from mono2048.brains import terminal as base

TARGET = 47000.0
BONUS = 32000.0
TEMP = 500.0
FW = 4.0
ACT1024 = 3
D1 = 3
D2 = 4
GS = 6
CAP_EXP = 10


def load(path):
    base.load(path)


def _sigmoid(x):
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _bank_gain_packed(board):
    best = 0
    for _, fn in app._BMOVES:
        nb, r = fn(board)
        if nb != board and r > best:
            best = r
    return float(best)


def _gd_leaf(nacc, after):
    est = nacc + base._val(after)
    bf = nacc + _bank_gain_packed(after)
    et = bf if bf > est else est
    return et + BONUS * _sigmoid((et - TARGET) / TEMP)


def _gd_chance(after, depth, nacc):
    empties = app._empty_idx(after)
    n = len(empties)
    if n == 0:
        return _gd_leaf(nacc, after)
    if n > GS:
        empties = random.sample(empties, GS)
    total = 0.0
    for rk in empties:
        total += 0.9 * _gd_max(app._set_cell(after, rk, 1), depth, nacc)
        total += 0.1 * _gd_max(app._set_cell(after, rk, 2), depth, nacc)
    return total / len(empties)


def _gd_max(board, depth, acc):
    best = -1e18
    found = False
    for _, fn in app._BMOVES:
        nb, r = fn(board)
        if nb == board or app._max_exp(nb) > CAP_EXP:
            continue
        found = True
        nacc = acc + r
        if depth <= 1:
            u = _gd_leaf(nacc, nb)
        else:
            u = _gd_chance(nb, depth - 1, nacc)
        if u > best:
            best = u
    if not found:
        total = acc + _bank_gain_packed(board)
        if total > TARGET:
            best = 1e6 + total
        else:
            best = total - FW * (TARGET - total)
    return best


def _gd_root(board, acc, depth):
    best, bname = -1e18, None
    for name, fn in app._BMOVES:
        nb, r = fn(board)
        if nb == board or app._max_exp(nb) > CAP_EXP:
            continue
        nacc = acc + r
        if depth <= 1:
            u = _gd_leaf(nacc, nb)
        else:
            u = _gd_chance(nb, depth - 1, nacc)
        if u > best:
            best, bname = u, name
    return bname


def _bank_qualifies(values, score, target=TARGET):
    bmv, gain = app.best_bank_move_2048(values)
    if bmv is None:
        return False
    return score + gain >= target


def get_best_move(values, cap=1024, depth=None, score=0.0):
    if base._LUT is None:
        raise RuntimeError("LUT not loaded — call record47.load(path)")
    if _bank_qualifies(values, score):
        return None
    n1024 = 0
    for v in values:
        if v == 1024:
            n1024 += 1
    if n1024 < ACT1024:
        return base.get_best_move(values, cap=cap or 1024, depth=D1)
    board = app._pack(values)
    return _gd_root(board, float(score), D2)
