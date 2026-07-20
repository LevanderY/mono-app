import argparse
import multiprocessing as mp
import pickle
import time
from multiprocessing import shared_memory

import numpy as np
from numba import njit

TUPLES = [
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
    (0, 1, 4, 5, 8, 9),
    (1, 2, 5, 6, 9, 10),
    (0, 1, 2, 4, 8, 12),
    (5, 6, 7, 9, 10, 11),
]
TLEN = 6
K = len(TUPLES)
NSTAGES = 5


def _perms():
    def rc(i):
        return divmod(i, 4)

    def rot(i):
        r, c = rc(i)
        return c * 4 + (3 - r)

    def mir(i):
        r, c = rc(i)
        return r * 4 + (3 - c)

    fns = [
        lambda i: i,
        rot,
        lambda i: rot(rot(i)),
        lambda i: rot(rot(rot(i))),
        mir,
        lambda i: rot(mir(i)),
        lambda i: rot(rot(mir(i))),
        lambda i: rot(rot(rot(mir(i)))),
    ]
    return [[f(i) for i in range(16)] for f in fns]


def build_sym():
    perms = _perms()
    sym = np.zeros((K, 8, TLEN), np.int64)
    for k, t in enumerate(TUPLES):
        for s in range(8):
            for j in range(TLEN):
                sym[k, s, j] = perms[s][t[j]]
    return sym


def _merge_left(tiles):
    xs = [t for t in tiles if t != 0]
    res, sc, i = [], 0, 0
    while i < len(xs):
        if i + 1 < len(xs) and xs[i] == xs[i + 1]:
            e = xs[i] + 1
            res.append(e)
            sc += 1 << e
            i += 2
        else:
            res.append(xs[i])
            i += 1
    while len(res) < 4:
        res.append(0)
    return res, sc


def build_rows():
    RL = np.zeros(65536, np.int32)
    RLS = np.zeros(65536, np.int32)
    RR = np.zeros(65536, np.int32)
    RRS = np.zeros(65536, np.int32)
    for row in range(65536):
        tiles = [row & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF]
        res, sc = _merge_left(tiles)
        RL[row] = res[0] | res[1] << 4 | res[2] << 8 | res[3] << 12
        RLS[row] = sc
        rres, rsc = _merge_left(tiles[::-1])
        rres = rres[::-1]
        RR[row] = rres[0] | rres[1] << 4 | rres[2] << 8 | rres[3] << 12
        RRS[row] = rsc
    return RL, RLS, RR, RRS


@njit(cache=True, inline='always')
def _max16(b):
    m = 0
    for k in range(16):
        if b[k] > m:
            m = b[k]
    return m


@njit(cache=True, inline='always')
def _stage(b):
    c = 0
    for k in range(16):
        if b[k] == 10:
            c += 1
    if c >= NSTAGES:
        c = NSTAGES - 1
    return c


@njit(cache=True)
def _do_move(b, out, d, RL, RLS, RR, RRS):
    for k in range(16):
        out[k] = b[k]
    score = 0
    if d == 0:
        for r in range(4):
            o = 4 * r
            row = out[o] | (out[o + 1] << 4) | (out[o + 2] << 8) | (out[o + 3] << 12)
            nr = RL[row]
            score += RLS[row]
            out[o] = nr & 0xF
            out[o + 1] = (nr >> 4) & 0xF
            out[o + 2] = (nr >> 8) & 0xF
            out[o + 3] = (nr >> 12) & 0xF
    elif d == 1:
        for r in range(4):
            o = 4 * r
            row = out[o] | (out[o + 1] << 4) | (out[o + 2] << 8) | (out[o + 3] << 12)
            nr = RR[row]
            score += RRS[row]
            out[o] = nr & 0xF
            out[o + 1] = (nr >> 4) & 0xF
            out[o + 2] = (nr >> 8) & 0xF
            out[o + 3] = (nr >> 12) & 0xF
    elif d == 2:
        for c in range(4):
            col = out[c] | (out[4 + c] << 4) | (out[8 + c] << 8) | (out[12 + c] << 12)
            nc = RL[col]
            score += RLS[col]
            out[c] = nc & 0xF
            out[4 + c] = (nc >> 4) & 0xF
            out[8 + c] = (nc >> 8) & 0xF
            out[12 + c] = (nc >> 12) & 0xF
    else:
        for c in range(4):
            col = out[c] | (out[4 + c] << 4) | (out[8 + c] << 8) | (out[12 + c] << 12)
            nc = RR[col]
            score += RRS[col]
            out[c] = nc & 0xF
            out[4 + c] = (nc >> 4) & 0xF
            out[8 + c] = (nc >> 8) & 0xF
            out[12 + c] = (nc >> 12) & 0xF
    moved = False
    for k in range(16):
        if out[k] != b[k]:
            moved = True
            break
    return score, moved


@njit(cache=True)
def _V(b, LUT, SYM):
    st = _stage(b)
    s = 0.0
    for k in range(K):
        for sm in range(8):
            idx = 0
            for j in range(TLEN):
                idx |= b[SYM[k, sm, j]] << (4 * j)
            s += LUT[st, k, idx]
    return s


@njit(cache=True)
def _best_bank(b, RL, RLS, RR, RRS):
    tmp = np.empty(16, np.int8)
    bg = -1
    found = False
    for d in range(4):
        sc, moved = _do_move(b, tmp, d, RL, RLS, RR, RRS)
        if moved and sc > bg:
            bg = sc
            found = True
    return found, bg


@njit(cache=True)
def _spawn(b):
    cnt = 0
    idxs = np.empty(16, np.int64)
    for k in range(16):
        if b[k] == 0:
            idxs[cnt] = k
            cnt += 1
    if cnt == 0:
        return
    r = idxs[np.random.randint(cnt)]
    b[r] = 2 if np.random.random() < 0.1 else 1


@njit(cache=True)
def _start(b):
    for k in range(16):
        b[k] = 0
    _spawn(b)
    _spawn(b)


@njit
def _expmax(b, depth, S, RL, RLS, RR, RRS, LUT, SYM):
    tmp = np.empty(16, np.int8)
    empties = np.empty(16, np.int64)
    b2 = np.empty(16, np.int8)
    best = -1e18
    for d in range(4):
        sc, moved = _do_move(b, tmp, d, RL, RLS, RR, RRS)
        if not moved or _max16(tmp) > 10:
            continue
        if depth <= 1:
            v = sc + _V(tmp, LUT, SYM)
        else:
            n = 0
            for k in range(16):
                if tmp[k] == 0:
                    empties[n] = k
                    n += 1
            if n == 0:
                v = sc + _V(tmp, LUT, SYM)
            else:
                if S < n:
                    for i in range(S):
                        j = i + np.random.randint(n - i)
                        empties[i], empties[j] = empties[j], empties[i]
                    m = S
                else:
                    m = n
                tot = 0.0
                for i in range(m):
                    rk = empties[i]
                    for k in range(16):
                        b2[k] = tmp[k]
                    b2[rk] = 1
                    tot += 0.9 * _expmax(b2, depth - 1, S, RL, RLS, RR, RRS, LUT, SYM)
                    b2[rk] = 2
                    tot += 0.1 * _expmax(b2, depth - 1, S, RL, RLS, RR, RRS, LUT, SYM)
                v = sc + tot / m
        if v > best:
            best = v
    if best <= -1e17:
        fb, br = _best_bank(b, RL, RLS, RR, RRS)
        return float(br) if fb else 0.0
    return best


@njit
def _root(b, depth, S, RL, RLS, RR, RRS, LUT, SYM):
    tmp = np.empty(16, np.int8)
    empties = np.empty(16, np.int64)
    b2 = np.empty(16, np.int8)
    best = -1e18
    bd = -1
    for d in range(4):
        sc, moved = _do_move(b, tmp, d, RL, RLS, RR, RRS)
        if not moved or _max16(tmp) > 10:
            continue
        if depth <= 1:
            v = sc + _V(tmp, LUT, SYM)
        else:
            n = 0
            for k in range(16):
                if tmp[k] == 0:
                    empties[n] = k
                    n += 1
            if n == 0:
                v = sc + _V(tmp, LUT, SYM)
            else:
                if S < n:
                    for i in range(S):
                        j = i + np.random.randint(n - i)
                        empties[i], empties[j] = empties[j], empties[i]
                    m = S
                else:
                    m = n
                tot = 0.0
                for i in range(m):
                    rk = empties[i]
                    for k in range(16):
                        b2[k] = tmp[k]
                    b2[rk] = 1
                    tot += 0.9 * _expmax(b2, depth - 1, S, RL, RLS, RR, RRS, LUT, SYM)
                    b2[rk] = 2
                    tot += 0.1 * _expmax(b2, depth - 1, S, RL, RLS, RR, RRS, LUT, SYM)
                v = sc + tot / m
        if v > best:
            best = v
            bd = d
    return bd


@njit
def _play_search(seed, depth, S, RL, RLS, RR, RRS, LUT, SYM):
    np.random.seed(seed)
    b = np.empty(16, np.int8)
    tmp = np.empty(16, np.int8)
    _start(b)
    score = 0
    while True:
        d = _root(b, depth, S, RL, RLS, RR, RRS, LUT, SYM)
        if d < 0:
            bf, br = _best_bank(b, RL, RLS, RR, RRS)
            if bf:
                score += br
            break
        sc, _ = _do_move(b, tmp, d, RL, RLS, RR, RRS)
        score += sc
        for k in range(16):
            b[k] = tmp[k]
        _spawn(b)
    h = np.zeros(12, np.int64)
    for k in range(16):
        if b[k] >= 9:
            h[b[k]] += 1
    return score, h


_G = {}


def _init(name, shape, dtype, depth, S):
    shm = shared_memory.SharedMemory(name=name)
    _G["shm"] = shm
    _G["LUT"] = np.ndarray(shape, np.dtype(dtype), buffer=shm.buf)
    _G["SYM"] = build_sym()
    _G["RL"], _G["RLS"], _G["RR"], _G["RRS"] = build_rows()
    _G["depth"] = depth
    _G["S"] = S


def _game(seed):
    sc, h = _play_search(seed, _G["depth"], _G["S"], _G["RL"], _G["RLS"],
                         _G["RR"], _G["RRS"], _G["LUT"], _G["SYM"])
    return int(sc), h.tolist()


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model with multistage expectimax self-play.")
    p.add_argument("--lut", default="models/nt3_warm_2m.pkl")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--sample", type=int, default=6)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--nproc", type=int, default=7)
    a = p.parse_args()

    with open(a.lut, "rb") as f:
        d = pickle.load(f)
    lut = np.ascontiguousarray(d["lut"], dtype=np.float32)
    ns = d.get("nstages", lut.shape[0])
    if ns != NSTAGES:
        raise SystemExit("nstages %d != evaluate NSTAGES %d" % (ns, NSTAGES))
    shm = shared_memory.SharedMemory(create=True, size=lut.nbytes)
    buf = np.ndarray(lut.shape, lut.dtype, buffer=shm.buf)
    buf[:] = lut
    del lut

    seeds = list(range(a.seed, a.seed + a.games))
    t0 = time.perf_counter()
    try:
        with mp.Pool(a.nproc, initializer=_init,
                     initargs=(shm.name, buf.shape, buf.dtype.str, a.depth, a.sample)) as pool:
            res = pool.map(_game, seeds)
    finally:
        shm.close()
        shm.unlink()
    sc = sorted(r[0] for r in res)
    dt = time.perf_counter() - t0
    best = max(res, key=lambda r: r[0])
    names = {9: "512", 10: "1024", 11: "2048"}
    hist = {names.get(e, str(2 ** e)): c for e, c in enumerate(best[1]) if c}
    n = len(sc)
    print("depth=%d sample=%d  %d games (%.0fs, %.1fs/game)"
          % (a.depth, a.sample, a.games, dt, dt / a.games))
    print("  median=%d  mean=%d  max=%d  min=%d  >=47000: %d/%d   best tiles: %s"
          % (sc[n // 2], sum(sc) // n, max(sc), min(sc),
             sum(1 for x in sc if x >= 47000), n, hist))


if __name__ == "__main__":
    main()
