import itertools
import json
import logging
import os

import numpy as np
from PIL import ImageDraw

LOG = logging.getLogger("bot2048.vision")

SIZE = 4
DEFAULT_CALIB = "calib.json"

DEFAULT_TOLERANCE = 10

FRAME_OUTER = 0.10
FRAME_INNER = 0.30


class CalibrationError(RuntimeError):
    pass


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _mode_color(pixels):
    if len(pixels) == 0:
        raise ValueError("empty pixel sample")
    p = np.asarray(pixels, dtype=np.uint32)
    packed = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    best = int(vals[counts.argmax()])
    return (best >> 16) & 255, (best >> 8) & 255, best & 255


def load_calib(path=DEFAULT_CALIB):
    if not os.path.exists(path):
        raise CalibrationError(
            "No %s. Calibrate first: python3 scripts/calibrate.py --from-phone" % path
        )
    with open(path, "r", encoding="utf-8") as f:
        c = json.load(f)
    c["colors"] = {int(k): tuple(v) for k, v in c.get("colors", {}).items()}
    c["empty"] = [tuple(v) if v else None for v in c.get("empty", [None] * 16)]
    for key in ("cols", "rows", "cell", "board"):
        if key not in c:
            raise CalibrationError("%s is missing field '%s' — recalibrate" % (path, key))
        c[key] = list(c[key])
    return c


def save_calib(calib, path=DEFAULT_CALIB):
    out = dict(calib)
    out["colors"] = {str(k): list(v) for k, v in sorted(calib.get("colors", {}).items())}
    out["empty"] = [list(v) if v else None for v in calib.get("empty", [None] * 16)]
    for key in ("cols", "rows", "cell", "board"):
        out[key] = list(calib[key])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    known = sum(1 for v in out["empty"] if v)
    LOG.info("Calibration saved to %s (%d tiles, %d/16 empty cells)",
             path, len(out["colors"]), known)


def new_calib(grid, tolerance=DEFAULT_TOLERANCE):
    return {
        "cols": list(grid["cols"]), "rows": list(grid["rows"]),
        "cell": list(grid["cell"]), "board": list(grid["board"]),
        "tolerance": tolerance, "colors": {}, "empty": [None] * 16,
    }


def cell_rects(calib):
    cols, rows = calib["cols"], calib["rows"]
    cw, ch = calib["cell"]
    return [(cols[i % SIZE], rows[i // SIZE], cols[i % SIZE] + cw, rows[i // SIZE] + ch)
            for i in range(SIZE * SIZE)]


def cell_color(arr, rect):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    ox, oy = int(w * FRAME_OUTER), int(h * FRAME_OUTER)
    ix, iy = int(w * FRAME_INNER), int(h * FRAME_INNER)
    patch = arr[y0 + oy: y1 - oy, x0 + ox: x1 - ox]
    if patch.size == 0:
        raise CalibrationError("cell %s too small to sample" % (rect,))
    mask = np.ones(patch.shape[:2], dtype=bool)
    top, left = iy - oy, ix - ox
    if top > 0 and left > 0 and patch.shape[0] - top > top and patch.shape[1] - left > left:
        mask[top:patch.shape[0] - top, left:patch.shape[1] - left] = False
    pix = patch[mask]
    if len(pix) == 0:
        pix = patch.reshape(-1, 3)
    return _mode_color(pix)


def cell_colors(img, calib):
    arr = np.asarray(img)
    return [cell_color(arr, r) for r in cell_rects(calib)]


def _dense_bbox(mask, frac=0.15):
    rp, cp = mask.sum(axis=1), mask.sum(axis=0)
    if rp.max() == 0 or cp.max() == 0:
        return None
    ys = np.nonzero(rp > rp.max() * frac)[0]
    xs = np.nonzero(cp > cp.max() * frac)[0]
    if len(ys) < 2 or len(xs) < 2:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def _bands(profile, thr, minw):
    out, start = [], None
    for i, on in enumerate(profile > thr):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= minw:
                out.append((start, i - 1))
            start = None
    if start is not None and len(profile) - start >= minw:
        out.append((start, len(profile) - 1))
    return out


def _pick4(bands):
    if len(bands) < SIZE:
        return None, None
    best, best_score = None, None
    for combo in itertools.combinations(bands, SIZE):
        widths = [e - s + 1 for s, e in combo]
        starts = [s for s, _ in combo]
        pitches = [starts[i + 1] - starts[i] for i in range(SIZE - 1)]
        if min(pitches) <= 0:
            continue
        score = float(np.std(widths) + np.std(pitches))
        if best_score is None or score < best_score:
            best, best_score = combo, score
    return best, best_score


def _gap_candidates(arr, top_k):
    flat = arr.reshape(-1, 3)
    p = flat.astype(np.uint32)
    packed = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
    vals, counts = np.unique(packed, return_counts=True)

    out = []
    for idx in np.argsort(counts)[::-1][:top_k]:
        v = int(vals[idx])
        out.append(((v >> 16) & 255, (v >> 8) & 255, v & 255))

    luma = flat.mean(axis=1)
    for lo, hi in ((0, 12), (88, 100)):
        sel = flat[(luma >= np.percentile(luma, lo)) & (luma <= np.percentile(luma, hi))]
        if len(sel):
            out.append(_mode_color(sel))

    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def find_grid(img, top_k=10, gap_tol=18, max_score=6.0):
    arr = np.asarray(img).astype(np.int16)

    for gap in _gap_candidates(arr, top_k):
        gap_mask = np.abs(arr - np.array(gap)).sum(axis=2) <= gap_tol
        box = _dense_bbox(gap_mask)
        if box is None:
            continue
        X0, Y0, X1, Y1 = box
        if X1 - X0 < 80 or Y1 - Y0 < 80:
            continue

        cells = ~gap_mask[Y0:Y1 + 1, X0:X1 + 1]
        minw = max(10, (Y1 - Y0) // 12)
        rows4, rs = _pick4(_bands(cells.sum(axis=1), cells.shape[1] * 0.55, minw))
        if rows4 is None or rs > max_score:
            continue
        sub = cells[rows4[0][0]:rows4[-1][1] + 1, :]
        cols4, cs = _pick4(_bands(sub.sum(axis=0), sub.shape[0] * 0.55, minw))
        if cols4 is None or cs > max_score:
            continue

        cols = [s + X0 for s, _ in cols4]
        rows = [s + Y0 for s, _ in rows4]
        cw = int(round(np.mean([e - s + 1 for s, e in cols4])))
        ch = int(round(np.mean([e - s + 1 for s, e in rows4])))
        if not 0.85 < cw / float(ch) < 1.18:
            continue

        board = [cols[0], rows[0], cols[-1] + cw - cols[0], rows[-1] + ch - rows[0]]
        LOG.info("Grid found by gap color RGB%s", gap)
        LOG.info("  cols=%s rows=%s cell=%dx%d step=%d/%d",
                 cols, rows, cw, ch, cols[1] - cols[0], rows[1] - rows[0])
        return {"cols": cols, "rows": rows, "cell": [cw, ch], "board": board, "gap": gap}

    raise CalibrationError(
        "Could not find the grid automatically. Set coordinates manually: "
        "--cols X0,X1,X2,X3 --rows Y0,Y1,Y2,Y3 --cell W,H"
    )


def classify(idx, color, calib):
    tol = calib.get("tolerance", DEFAULT_TOLERANCE)
    ranked = sorted((_dist(color, rgb), val) for val, rgb in calib["colors"].items())
    empty_ref = calib["empty"][idx]
    d_empty = _dist(color, empty_ref) if empty_ref else None

    cands = []
    if d_empty is not None:
        cands.append((d_empty, 0))
    cands.extend(ranked)
    cands.sort()
    if not cands:
        return None

    best_d, best_val = cands[0]
    if best_d > tol:
        return None
    if best_d > 0 and len(cands) > 1 and cands[1][0] < 2 * best_d:
        LOG.warning("Cell %d: color RGB%s is equally similar to %s (%d) and %s (%d) — "
                    "treating as unknown", idx, color, best_val, best_d, cands[1][1], cands[1][0])
        return None
    return best_val


def read_board(img, calib):
    board, unknowns = [], []
    for i, col in enumerate(cell_colors(img, calib)):
        val = classify(i, col, calib)
        if val is None:
            unknowns.append((i, col))
            board.append(0)
        else:
            board.append(val)
    return board, unknowns


def learn_colors(img, calib, true_board):
    learned = []
    for i, col in enumerate(cell_colors(img, calib)):
        val = true_board[i]
        if val == 0:
            if calib["empty"][i] is None:
                calib["empty"][i] = tuple(col)
                learned.append(("empty", i, tuple(col)))
                LOG.info("Learned: empty cell %-2d -> RGB%s", i, col)
        elif val not in calib["colors"]:
            calib["colors"][val] = tuple(col)
            learned.append(("tile", val, tuple(col)))
            LOG.info("Learned: tile %-5d      -> RGB%s", val, col)
    return learned


def check_palette(calib):
    tol = calib.get("tolerance", DEFAULT_TOLERANCE)
    tiles = [("tile %s" % v, rgb) for v, rgb in sorted(calib["colors"].items())]
    empties = [("empty %d" % i, rgb) for i, rgb in enumerate(calib["empty"]) if rgb]

    pairs = list(itertools.combinations(tiles, 2))
    pairs += [(e, t) for e in empties for t in tiles]

    worst = None
    for (na, ca), (nb, cb) in pairs:
        d = _dist(ca, cb)
        if worst is None or d < worst[0]:
            worst = (d, na, nb)
    if worst and worst[0] <= 2 * tol:
        LOG.warning("Closest color pair (%s and %s) differs by only %d at "
                    "tolerance %d. Ambiguity protection will trigger, but it is safer to "
                    "lower --tolerance to %d.",
                    worst[1], worst[2], worst[0], tol, max(1, worst[0] // 3))
    elif worst:
        LOG.info("Closest color pair (%s and %s) differs by %d — there is margin.",
                 worst[1], worst[2], worst[0])
    return worst


def annotate(img, calib, path):
    out = img.copy()
    d = ImageDraw.Draw(out)
    x, y, w, h = calib["board"]
    d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
    for i, (x0, y0, x1, y1) in enumerate(cell_rects(calib)):
        d.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=2)
        cw, ch = x1 - x0, y1 - y0
        ox, oy = int(cw * FRAME_OUTER), int(ch * FRAME_OUTER)
        ix, iy = int(cw * FRAME_INNER), int(ch * FRAME_INNER)
        d.rectangle([x0 + ox, y0 + oy, x1 - ox, y1 - oy], outline=(0, 255, 0), width=2)
        d.rectangle([x0 + ix, y0 + iy, x1 - ix, y1 - iy], outline=(255, 0, 255), width=2)
        d.text((x0 + 8, y0 + 6), str(i), fill=(255, 60, 60))
    out.save(path)
    LOG.info("Check the markup visually: %s "
             "(green frame = where we sample color, pink = cut-out center)", path)
    return path


def board_str(b):
    width = max([len(str(v)) for v in b if v] + [4])
    line = "+" + ("-" * (width + 2) + "+") * SIZE
    out = [line]
    for r in range(SIZE):
        row = b[r * SIZE:(r + 1) * SIZE]
        out.append("| " + " | ".join((str(v) if v else "").center(width) for v in row) + " |")
        out.append(line)
    return "\n".join(out)
