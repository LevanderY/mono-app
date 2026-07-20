import argparse
import os
import pickle
import time
from multiprocessing import Pool, shared_memory

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
SIZE = 16 ** TLEN
NINST = K * 8
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
def _update(b, err, LUT, E, A, alpha, use_tc, SYM):
    st = _stage(b)
    per = err / NINST
    ap = abs(per)
    for k in range(K):
        for sm in range(8):
            idx = 0
            for j in range(TLEN):
                idx |= b[SYM[k, sm, j]] << (4 * j)
            if use_tc:
                a = A[st, k, idx]
                if a > 1e-12:
                    coh = abs(E[st, k, idx]) / a
                else:
                    coh = 1.0
                LUT[st, k, idx] += alpha * coh * per
                E[st, k, idx] += per
                A[st, k, idx] += ap
            else:
                LUT[st, k, idx] += alpha * per


@njit(cache=True)
def _best_capped(b, out, RL, RLS, RR, RRS, LUT, SYM):
    tmp = np.empty(16, np.int8)
    bv = -1e18
    br = 0
    bvv = 0.0
    found = False
    for d in range(4):
        sc, moved = _do_move(b, tmp, d, RL, RLS, RR, RRS)
        if not moved or _max16(tmp) > 10:
            continue
        v = _V(tmp, LUT, SYM)
        if sc + v > bv:
            bv = sc + v
            br = sc
            bvv = v
            for k in range(16):
                out[k] = tmp[k]
            found = True
    return found, br, bvv


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


@njit(cache=True)
def _play_learn(alpha, use_tc, RL, RLS, RR, RRS, LUT, E, A, SYM):
    board = np.empty(16, np.int8)
    cur_after = np.empty(16, np.int8)
    next_after = np.empty(16, np.int8)
    _start(board)
    score = 0
    found, r, v = _best_capped(board, cur_after, RL, RLS, RR, RRS, LUT, SYM)
    while found:
        score += r
        for k in range(16):
            board[k] = cur_after[k]
        _spawn(board)
        nfound, r2, v2 = _best_capped(board, next_after, RL, RLS, RR, RRS, LUT, SYM)
        if nfound:
            target = r2 + v2
        else:
            bf, breward = _best_bank(board, RL, RLS, RR, RRS)
            target = float(breward) if bf else 0.0
        _update(cur_after, target - v, LUT, E, A, alpha, use_tc, SYM)
        for k in range(16):
            cur_after[k] = next_after[k]
        found, r, v = nfound, r2, v2
    bf, breward = _best_bank(board, RL, RLS, RR, RRS)
    if bf:
        score += breward
    return score


@njit(cache=True)
def _train_chunk(n, alpha, use_tc, RL, RLS, RR, RRS, LUT, E, A, SYM):
    tot = 0.0
    for _ in range(n):
        tot += _play_learn(alpha, use_tc, RL, RLS, RR, RRS, LUT, E, A, SYM)
    return tot / n


@njit(cache=True)
def _greedy_eval(seed, RL, RLS, RR, RRS, LUT, SYM):
    np.random.seed(seed)
    board = np.empty(16, np.int8)
    after = np.empty(16, np.int8)
    _start(board)
    score = 0
    while True:
        found, r, v = _best_capped(board, after, RL, RLS, RR, RRS, LUT, SYM)
        if not found:
            bf, breward = _best_bank(board, RL, RLS, RR, RRS)
            if bf:
                score += breward
            break
        score += r
        for k in range(16):
            board[k] = after[k]
        _spawn(board)
    return score


_G = {}


def _init(meta, tc):
    _G["shms"] = []
    for key, (name, shape, dtype) in meta.items():
        shm = shared_memory.SharedMemory(name=name)
        _G["shms"].append(shm)
        _G[key] = np.ndarray(shape, np.dtype(dtype), buffer=shm.buf)
    _G["tc"] = tc
    _G["RL"], _G["RLS"], _G["RR"], _G["RRS"] = build_rows()
    _G["SYM"] = build_sym()


def _train(alpha, n):
    return _train_chunk(n, alpha, _G["tc"], _G["RL"], _G["RLS"], _G["RR"],
                        _G["RRS"], _G["LUT"], _G["E"], _G["A"], _G["SYM"])


def _eval(seed):
    return int(_greedy_eval(seed, _G["RL"], _G["RLS"], _G["RR"], _G["RRS"],
                            _G["LUT"], _G["SYM"]))


def _save(path, lut):
    with open(path, "wb") as f:
        pickle.dump({"tuples": TUPLES, "lut": np.array(lut), "nstages": NSTAGES}, f,
                    protocol=4)


def _mk(shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype, buffer=shm.buf)
    arr[:] = 0
    return shm, arr


def main():
    p = argparse.ArgumentParser(
        description="Train the multistage n-tuple value network (Hogwild TD/TC).")
    p.add_argument("--games", type=int, default=8_000_000)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--chunk", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--alpha-end", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--init-from", default="",
                   help="single-stage .pkl (K, SIZE): warm-start every stage from it")
    p.add_argument("--save", default="models/trained.pkl",
                   help="where to write the trained model")
    p.add_argument("--force", action="store_true",
                   help="overwrite --save (and its .best) if they already exist")
    a = p.parse_args()
    existing = [p for p in (a.save, a.save + ".best") if os.path.exists(p)]
    if existing and not a.force:
        raise SystemExit(
            "refusing to overwrite existing %s (pass --force, or choose another --save)"
            % ", ".join(existing))
    tc = not a.no_tc

    shape = (NSTAGES, K, SIZE)
    shms, arrs = {}, {}
    for key in ("LUT", "E", "A"):
        shm, arr = _mk(shape, np.float32)
        shms[key], arrs[key] = shm, arr
    meta = {k: (shms[k].name, shape, arrs[k].dtype.str) for k in shms}

    if a.init_from:
        with open(a.init_from, "rb") as f:
            ss = pickle.load(f)["lut"]
        for st in range(NSTAGES):
            arrs["LUT"][st] = ss
        print("warm-start: %s -> all %d stages" % (a.init_from, NSTAGES), flush=True)

    per_round = a.workers * a.chunk
    nrounds = max(1, a.games // per_round)
    best_med = -1
    print("multistage Hogwild: %d workers x %d/round, %d rounds (~%d games), "
          "%d stages x %dx%d, LUT=%.1fGB x3, TC=%s, alpha %.2f->%.2f"
          % (a.workers, a.chunk, nrounds, nrounds * per_round, NSTAGES, K, TLEN,
             NSTAGES * K * SIZE * 4 / 1e9, tc, a.alpha, a.alpha_end), flush=True)

    t0 = time.perf_counter()
    with Pool(a.workers, initializer=_init, initargs=(meta, tc)) as pool:
        for rd in range(nrounds):
            frac = rd / max(1, nrounds - 1)
            alpha = a.alpha + (a.alpha_end - a.alpha) * frac
            trs = pool.starmap(_train, [(alpha, a.chunk)] * a.workers)
            if (rd + 1) % a.eval_every == 0 or rd == nrounds - 1:
                ev = sorted(pool.map(_eval, [1000 + i for i in range(a.eval_n)]))
                done = (rd + 1) * per_round
                dt = time.perf_counter() - t0
                med = ev[len(ev) // 2]
                print("games=%8d alpha=%.3f train=%6.0f  greedy median=%6d "
                      "mean=%6d max=%6d min=%5d  (%.0fs, %.0f games/s)"
                      % (done, alpha, float(np.mean(trs)), med, sum(ev) // len(ev),
                         max(ev), min(ev), dt, done / dt), flush=True)
                _save(a.save, arrs["LUT"])
                if med > best_med:
                    best_med = med
                    _save(a.save + ".best", arrs["LUT"])
    _save(a.save, arrs["LUT"])
    print("done in %.0fs -> %s" % (time.perf_counter() - t0, a.save), flush=True)
    for shm in shms.values():
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
