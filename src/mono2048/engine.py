import argparse
import logging
import random
import time

SIZE = 4

LOG = logging.getLogger("bot2048")


def new_board():
    b = [0] * (SIZE * SIZE)
    add_random_tile(b)
    add_random_tile(b)
    return b

def add_random_tile(b):
    empties = [i for i, v in enumerate(b) if v == 0]
    if not empties:
        return False
    i = random.choice(empties)
    b[i] = 4 if random.random() < 0.1 else 2
    return True

def _compress_merge_row(row):
    nums = [v for v in row if v != 0]
    merged, score, i = [], 0, 0
    while i < len(nums):
        if i + 1 < len(nums) and nums[i] == nums[i + 1]:
            val = nums[i] * 2
            merged.append(val)
            score += val
            i += 2
        else:
            merged.append(nums[i])
            i += 1
    merged += [0] * (SIZE - len(merged))
    return merged, score

def _rows(b):
    return [b[r * SIZE:(r + 1) * SIZE] for r in range(SIZE)]

def _from_rows(rows):
    out = []
    for r in rows:
        out.extend(r)
    return out

def _transpose(b):
    rows = _rows(b)
    t = [[rows[c][r] for c in range(SIZE)] for r in range(SIZE)]
    return _from_rows(t)

def move_left(b):
    out, total = [], 0
    for row in _rows(b):
        merged, s = _compress_merge_row(row)
        out.append(merged)
        total += s
    return _from_rows(out), total

def move_right(b):
    out, total = [], 0
    for row in _rows(b):
        merged, s = _compress_merge_row(row[::-1])
        out.append(merged[::-1])
        total += s
    return _from_rows(out), total

def move_up(b):
    moved, s = move_left(_transpose(b))
    return _transpose(moved), s

def move_down(b):
    moved, s = move_right(_transpose(b))
    return _transpose(moved), s

MOVES = {
    'up': move_up,
    'down': move_down,
    'left': move_left,
    'right': move_right,
}

def game_over(b):
    if 0 in b:
        return False
    for fn in MOVES.values():
        nb, _ = fn(b)
        if nb != b:
            return False
    return True


SUM_POWER = 3.5
SUM_WEIGHT = 0.0
MONO_POWER = 4.0
MONO_WEIGHT = 20.0
MERGE_WEIGHT = 150.0
EMPTY_WEIGHT = 300.0
GRAD_WEIGHT = 0.0
GRAD_RATIO = 0.7
CVAL_WEIGHT = 2.0
CORNER_WEIGHT = 0.0
SNAKE_WEIGHT = 30.0
TOPCNT_WEIGHT = 0.0
TOP_EXP = 10

SC_EMPTY = 300.0
SC_MERGE = 400.0
SC_SNAKE = 500.0

_SNAKE = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]

_ROW_RESULT = [0] * 65536
_ROW_SCORE = [0] * 65536
_ROW_REV = [0] * 65536
_ROW_HEUR = [0.0] * 65536
_ROW_CVAL = [0.0] * 65536
_ROW_MERGES = [0] * 65536
_SNAKE_LR = [0] * 65536
_SNAKE_RL = [0] * 65536
_GRADROW = [[0.0] * 65536 for _ in range(SIZE)]


def _cells(row):
    return [(row >> 12) & 0xF, (row >> 8) & 0xF, (row >> 4) & 0xF, row & 0xF]

def _pack_row(c):
    return (c[0] << 12) | (c[1] << 8) | (c[2] << 4) | c[3]

def _compress_exps(cells):
    nums = [x for x in cells if x]
    out, score, i = [], 0, 0
    while i < len(nums):
        if i + 1 < len(nums) and nums[i] == nums[i + 1]:
            e = nums[i] + 1
            out.append(e)
            score += 1 << e
            i += 2
        else:
            out.append(nums[i])
            i += 1
    out += [0] * (SIZE - len(out))
    return out, score

def _heur_line(cells):
    s_sum = 0.0
    empty = 0
    merges = 0
    prev = 0
    counter = 0
    for e in cells:
        s_sum += e ** SUM_POWER
        if e == 0:
            empty += 1
        else:
            if prev == e:
                counter += 1
            elif counter > 0:
                merges += 1 + counter
                counter = 0
            prev = e
    if counter > 0:
        merges += 1 + counter

    left = right = 0.0
    for i in range(SIZE - 1):
        a = cells[i] ** MONO_POWER
        b = cells[i + 1] ** MONO_POWER
        if cells[i] > cells[i + 1]:
            left += a - b
        else:
            right += b - a
    mono = min(left, right)

    return EMPTY_WEIGHT * empty + MERGE_WEIGHT * merges - MONO_WEIGHT * mono - SUM_WEIGHT * s_sum

def _build_move_tables():
    for row in range(65536):
        cells = _cells(row)
        nc, s = _compress_exps(cells)
        _ROW_RESULT[row] = _pack_row(nc)
        _ROW_SCORE[row] = s
        _ROW_REV[row] = _pack_row(cells[::-1])
        _ROW_CVAL[row] = sum((e - 1) * (1 << e) for e in cells if e > 0)
        _ROW_MERGES[row] = sum(1 for k in range(SIZE - 1) if cells[k] and cells[k] == cells[k + 1])
        _SNAKE_LR[row] = sum(max(0, cells[k + 1] - cells[k]) for k in range(SIZE - 1))
        _SNAKE_RL[row] = sum(max(0, cells[k] - cells[k + 1]) for k in range(SIZE - 1))

def build_heuristic_table():
    for row in range(65536):
        _ROW_HEUR[row] = _heur_line(_cells(row))
    if not GRAD_WEIGHT:
        return
    factor = [0.0] * (SIZE * SIZE)
    for pos, cell in enumerate(_SNAKE):
        factor[cell] = GRAD_RATIO ** pos
    for r in range(SIZE):
        f = factor[r * SIZE:r * SIZE + SIZE]
        for row in range(65536):
            cells = _cells(row)
            _GRADROW[r][row] = sum(f[k] * (1 << cells[k] if cells[k] else 0) for k in range(SIZE))

def configure(**kw):
    g = globals()
    for k, v in kw.items():
        if k.upper() in g:
            g[k.upper()] = v
    build_heuristic_table()

_build_move_tables()
build_heuristic_table()


def _pack(values):
    board = []
    for r in range(SIZE):
        c = []
        for k in range(SIZE):
            v = values[r * SIZE + k]
            c.append(v.bit_length() - 1 if v > 0 else 0)
        board.append(_pack_row(c))
    return tuple(board)

def _unpack(board):
    out = []
    for row in board:
        for e in _cells(row):
            out.append(1 << e if e else 0)
    return out

def _transpose_b(board):
    r0, r1, r2, r3 = board
    def col(s):
        return ((((r0 >> s) & 0xF) << 12) | (((r1 >> s) & 0xF) << 8)
                | (((r2 >> s) & 0xF) << 4) | ((r3 >> s) & 0xF))
    return (col(12), col(8), col(4), col(0))

def _mv_left(board):
    b0, b1, b2, b3 = board
    return ((_ROW_RESULT[b0], _ROW_RESULT[b1], _ROW_RESULT[b2], _ROW_RESULT[b3]),
            _ROW_SCORE[b0] + _ROW_SCORE[b1] + _ROW_SCORE[b2] + _ROW_SCORE[b3])

def _mv_right(board):
    out, sc = [], 0
    for r in board:
        rr = _ROW_REV[r]
        out.append(_ROW_REV[_ROW_RESULT[rr]])
        sc += _ROW_SCORE[rr]
    return tuple(out), sc

def _mv_up(board):
    t = _transpose_b(board)
    nb, sc = _mv_left(t)
    return _transpose_b(nb), sc

def _mv_down(board):
    t = _transpose_b(board)
    nb, sc = _mv_right(t)
    return _transpose_b(nb), sc

_BMOVES = (('up', _mv_up), ('down', _mv_down), ('left', _mv_left), ('right', _mv_right))

def _heur(board):
    t = _transpose_b(board)
    h = (_ROW_HEUR[board[0]] + _ROW_HEUR[board[1]] + _ROW_HEUR[board[2]] + _ROW_HEUR[board[3]]
         + _ROW_HEUR[t[0]] + _ROW_HEUR[t[1]] + _ROW_HEUR[t[2]] + _ROW_HEUR[t[3]])
    if GRAD_WEIGHT:
        h += GRAD_WEIGHT * (_GRADROW[0][board[0]] + _GRADROW[1][board[1]]
                            + _GRADROW[2][board[2]] + _GRADROW[3][board[3]])
    if CVAL_WEIGHT:
        h += CVAL_WEIGHT * (_ROW_CVAL[board[0]] + _ROW_CVAL[board[1]]
                            + _ROW_CVAL[board[2]] + _ROW_CVAL[board[3]])
    if CORNER_WEIGHT:
        c = ((board[0] >> 12) & 0xF, board[0] & 0xF, (board[3] >> 12) & 0xF, board[3] & 0xF)
        mx = _max_exp(board)
        if mx in c:
            h += CORNER_WEIGHT * mx
    if SNAKE_WEIGHT:
        t3, t7 = board[0] & 0xF, board[1] & 0xF
        t4, t8 = (board[1] >> 12) & 0xF, (board[2] >> 12) & 0xF
        t11, t15 = board[2] & 0xF, board[3] & 0xF
        trans = max(0, t7 - t3) + max(0, t8 - t4) + max(0, t15 - t11)
        h -= SNAKE_WEIGHT * (_SNAKE_LR[board[0]] + _SNAKE_RL[board[1]]
                             + _SNAKE_LR[board[2]] + _SNAKE_RL[board[3]] + trans)
    if TOPCNT_WEIGHT:
        cnt = 0
        for r in board:
            for s in (12, 8, 4, 0):
                if ((r >> s) & 0xF) >= TOP_EXP:
                    cnt += 1
        h += TOPCNT_WEIGHT * cnt * cnt
    return h

def _max_exp(board):
    m = 0
    for r in board:
        for s in (12, 8, 4, 0):
            e = (r >> s) & 0xF
            if e > m:
                m = e
    return m

def _empty_idx(board):
    idx = []
    for r in range(SIZE):
        row = board[r]
        for k, s in enumerate((12, 8, 4, 0)):
            if not (row >> s) & 0xF:
                idx.append((r, k))
    return idx

def _set_cell(board, rk, e):
    r, k = rk
    s = 12 - 4 * k
    row = board[r] & ~(0xF << s) | (e << s)
    return board[:r] + (row,) + board[r + 1:]


SAMPLE_EMPTY = 8
DEPTH_LADDER = ((10, 3), (6, 4), (3, 5), (1, 6), (0, 8))

_TT = {}
_CAP_EXP = 0

def _adaptive_depth(n_empty):
    for thr, d in DEPTH_LADDER:
        if n_empty > thr:
            return d
    return DEPTH_LADDER[-1][1]

def _emax(board, depth):
    if depth <= 0:
        return _heur(board)
    key = (board, depth, 1)
    v = _TT.get(key)
    if v is not None:
        return v
    best = -1e18
    moved = False
    for _, fn in _BMOVES:
        nb, _sc = fn(board)
        if nb == board:
            continue
        if _CAP_EXP and _max_exp(nb) > _CAP_EXP:
            continue
        moved = True
        val = _echance(nb, depth - 1)
        if val > best:
            best = val
    r = best if moved else _heur(board)
    _TT[key] = r
    return r

def _echance(board, depth):
    if depth <= 0:
        return _heur(board)
    key = (board, depth, 0)
    v = _TT.get(key)
    if v is not None:
        return v
    empties = _empty_idx(board)
    n = len(empties)
    if n == 0:
        r = _heur(board)
    else:
        if n > SAMPLE_EMPTY:
            empties = random.sample(empties, SAMPLE_EMPTY)
        total = 0.0
        for rk in empties:
            total += 0.9 * _emax(_set_cell(board, rk, 1), depth - 1)
            total += 0.1 * _emax(_set_cell(board, rk, 2), depth - 1)
        r = total / len(empties)
    _TT[key] = r
    return r

def get_best_move(b, cap=None):
    global _CAP_EXP
    board = _pack(b)
    _CAP_EXP = (cap.bit_length() - 1) if cap else 0
    _TT.clear()
    depth = _adaptive_depth(len(_empty_idx(board)))
    best_move, best_val = None, -1e18
    for name, fn in _BMOVES:
        nb, _sc = fn(board)
        if nb == board:
            continue
        if _CAP_EXP and _max_exp(nb) > _CAP_EXP:
            continue
        val = _echance(nb, depth - 1)
        if val > best_val:
            best_val, best_move = val, name
    return best_move


def _bank_value(board):
    return _ROW_CVAL[board[0]] + _ROW_CVAL[board[1]] + _ROW_CVAL[board[2]] + _ROW_CVAL[board[3]]

def _heur_score(board):
    base = _bank_value(board)
    empty = 0
    for r in board:
        for s in (12, 8, 4, 0):
            if not (r >> s) & 0xF:
                empty += 1
    t = _transpose_b(board)
    merges = (_ROW_MERGES[board[0]] + _ROW_MERGES[board[1]] + _ROW_MERGES[board[2]] + _ROW_MERGES[board[3]]
              + _ROW_MERGES[t[0]] + _ROW_MERGES[t[1]] + _ROW_MERGES[t[2]] + _ROW_MERGES[t[3]])
    t3, t7 = board[0] & 0xF, board[1] & 0xF
    t4, t8 = (board[1] >> 12) & 0xF, (board[2] >> 12) & 0xF
    t11, t15 = board[2] & 0xF, board[3] & 0xF
    trans = max(0, t7 - t3) + max(0, t8 - t4) + max(0, t15 - t11)
    snakepen = (_SNAKE_LR[board[0]] + _SNAKE_RL[board[1]] + _SNAKE_LR[board[2]]
                + _SNAKE_RL[board[3]] + trans)
    return base + SC_EMPTY * empty + SC_MERGE * merges - SC_SNAKE * snakepen

def _emax_bank(board, depth):
    if depth <= 0:
        return _heur_score(board)
    key = (board, depth, 3)
    v = _TT.get(key)
    if v is not None:
        return v
    best = -1e18
    moved = False
    for _, fn in _BMOVES:
        nb, _sc = fn(board)
        if nb == board:
            continue
        moved = True
        val = _bank_value(nb) if _max_exp(nb) >= 11 else _echance_bank(nb, depth - 1)
        if val > best:
            best = val
    r = best if moved else _bank_value(board)
    _TT[key] = r
    return r

def _echance_bank(board, depth):
    if depth <= 0:
        return _heur_score(board)
    key = (board, depth, 2)
    v = _TT.get(key)
    if v is not None:
        return v
    empties = _empty_idx(board)
    n = len(empties)
    if n == 0:
        r = _heur_score(board)
    else:
        if n > SAMPLE_EMPTY:
            empties = random.sample(empties, SAMPLE_EMPTY)
        total = 0.0
        for rk in empties:
            total += 0.9 * _emax_bank(_set_cell(board, rk, 1), depth - 1)
            total += 0.1 * _emax_bank(_set_cell(board, rk, 2), depth - 1)
        r = total / len(empties)
    _TT[key] = r
    return r

def get_best_move_bank(b):
    global _CAP_EXP
    board = _pack(b)
    _CAP_EXP = 0
    _TT.clear()
    depth = _adaptive_depth(len(_empty_idx(board)))
    best_move, best_val = None, -1e18
    for name, fn in _BMOVES:
        nb, _sc = fn(board)
        if nb == board:
            continue
        val = _bank_value(nb) if _max_exp(nb) >= 11 else _echance_bank(nb, depth - 1)
        if val > best_val:
            best_val, best_move = val, name
    return best_move


def best_bank_move(b):
    best, best_gain = None, -1
    for name, fn in MOVES.items():
        nb, s = fn(b)
        if nb != b and s > best_gain:
            best_gain, best = s, name
    return (best, best_gain) if best else (None, 0)


def best_bank_move_2048(b):
    best, best_gain = None, -1
    for name, fn in MOVES.items():
        nb, s = fn(b)
        if nb != b and max(nb) >= 2048 and s > best_gain:
            best_gain, best = s, name
    return (best, best_gain) if best else (None, 0)


def board_lines(b):
    width = max([len(str(v)) for v in b if v] + [4])
    line = "+" + ("-" * (width + 2) + "+") * SIZE
    out = [line]
    for row in _rows(b):
        cells = " | ".join((str(v) if v else "").center(width) for v in row)
        out.append("| " + cells + " |")
        out.append(line)
    return out

def print_board(b):
    for l in board_lines(b):
        print(l)


def setup_logging(level="INFO", logfile=None):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

def play_demo(seed=None, cap=None):
    if seed is not None:
        random.seed(seed)
    b = new_board()
    score = moves = 0
    started = time.perf_counter()
    LOG.info("=== New game (seed=%s, cap=%s) ===", seed, cap)

    while not game_over(b):
        mv = get_best_move(b, cap=cap)
        if mv is None:
            mv, gained = best_bank_move(b)
            if mv is None:
                break
            b, _ = MOVES[mv](b)
            score += gained
            moves += 1
            LOG.info("Cap reached at move %d — banking final '%s' +%d -> %d.",
                     moves, mv, gained, score)
            break
        b, gained = MOVES[mv](b)
        score += gained
        moves += 1
        add_random_tile(b)
        LOG.info("move %4d  %-5s  +%-5d  score=%-7d max=%-5d empty=%-2d",
                 moves, mv, gained, score, max(b), b.count(0))

    elapsed = time.perf_counter() - started
    LOG.info("=== Game over ===")
    for l in board_lines(b):
        LOG.info(l)
    LOG.info("Moves: %d   Score: %d   Max tile: %d   Time: %.1f s (%.1f moves/s)",
             moves, score, max(b), elapsed, moves / elapsed if elapsed else 0.0)
    return score, max(b)


def main(argv=None):
    p = argparse.ArgumentParser(description="2048 bot (expectimax, nneonneo heuristic)")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cap", type=int, default=None, help="tile cap (e.g. 1024)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default=None)
    a = p.parse_args(argv)
    setup_logging(a.log_level, a.log_file)

    results = []
    for i in range(a.games):
        results.append(play_demo(seed=None if a.seed is None else a.seed + i, cap=a.cap))

    if a.games > 1:
        scores = sorted(r[0] for r in results)
        LOG.info("Summary of %d games: median=%d  max=%d  best tile=%d",
                 a.games, scores[len(scores) // 2], max(scores), max(r[1] for r in results))
    return results

if __name__ == "__main__":
    main()
