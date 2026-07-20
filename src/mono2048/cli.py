import argparse
import inspect
import logging
import random
import time

from mono2048 import vision
from mono2048.engine import (MOVES, best_bank_move, best_bank_move_2048,
                             game_over, get_best_move, setup_logging)

LOG = logging.getLogger("bot2048.bot")

NEW_TILES = (2, 4)

DEFAULT_MODEL = "models/nt3_warm_2m.pkl"

DEFAULT_MIN_DELAY = 0.6
DEFAULT_MAX_DELAY = 1.8

WIN_CLOSE_FRAC = (0.155, 0.319)
WIN_CLOSE_WAIT = 0.8
MAX_DISMISS = 6

OVERLAY_SYNC_WAIT = 12.0
OVERLAY_POLL = 0.3


class ParseMismatch(RuntimeError):
    pass


def move_delay(min_delay, max_delay):
    lo, hi = sorted((max(0.0, min_delay), max(0.0, max_delay)))
    return random.uniform(lo, hi) if hi > 0 else 0.0


def swipe_for(dev, move, rect, frac=0.32, ms=90):
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    dx, dy = int(w * frac), int(h * frac)
    vec = {"up": (0, -dy), "down": (0, dy), "left": (-dx, 0), "right": (dx, 0)}[move]
    dev.swipe(cx, cy, cx + vec[0], cy + vec[1], ms)


def tap_win_close(dev, img):
    dev.tap(int(img.width * WIN_CLOSE_FRAC[0]), int(img.height * WIN_CLOSE_FRAC[1]))


def read_settled(dev, calib, timeout=3.0, interval=0.05):
    prev = None
    deadline = time.time() + timeout
    while True:
        img = dev.screencap()
        colors = vision.cell_colors(img, calib)
        if prev is not None and colors == prev:
            board, unknowns = vision.read_board(img, calib)
            return img, board, unknowns
        if time.time() > deadline:
            board, unknowns = vision.read_board(img, calib)
            LOG.warning("Board did not settle within %.1f s — reading as-is (%d unknowns)",
                        timeout, len(unknowns))
            return img, board, unknowns
        prev = colors
        time.sleep(interval)


def read_fast(dev, calib, predicted, timeout=1.6, interval=0.025):
    deadline = time.time() + timeout
    while time.time() < deadline:
        img = dev.screencap()
        board, unknowns = vision.read_board(img, calib)
        if len(unknowns) > MAX_UNKNOWN:
            return None, None, None
        if not unknowns and check_prediction(predicted, board)[0]:
            return img, board, []
        time.sleep(interval)
    return None, None, None


def wait_for_board(dev, calib, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(OVERLAY_POLL)
        _, unk = vision.read_board(dev.screencap(), calib)
        if len(unk) <= MAX_UNKNOWN:
            return True
    return False


def recover_from_overlay(dev, calib, img, expected_app):
    if expected_app and dev.foreground_activity() != expected_app:
        raise ParseMismatch("overlay, but the game is no longer on screen — I won't swipe blind")
    LOG.info("Overlay — waiting patiently up to %.0f s for the server to load the round…",
             OVERLAY_SYNC_WAIT)
    if wait_for_board(dev, calib, OVERLAY_SYNC_WAIT):
        LOG.info("Board came back on its own (server responded) — continuing to play.")
        return True
    for attempt in range(1, MAX_DISMISS + 1):
        LOG.info("It didn't clear on its own — possibly a congratulations window. Pressing X (%d/%d)…",
                 attempt, MAX_DISMISS)
        tap_win_close(dev, img)
        if wait_for_board(dev, calib, WIN_CLOSE_WAIT + 2.0):
            LOG.info("Window closed — continuing to play.")
            return True
    LOG.warning("Overlay didn't clear by waiting or by the X button.")
    return False


def check_prediction(predicted, observed):
    diffs = [i for i in range(16) if predicted[i] != observed[i]]
    if len(diffs) != 1:
        return False, "expected 1 new tile, but there are %d differences: %s" % (len(diffs), diffs)
    i = diffs[0]
    if predicted[i] != 0:
        return False, "cell %d should have been %d, but became %d" % (i, predicted[i], observed[i])
    if observed[i] not in NEW_TILES:
        return False, "new tile in cell %d = %d (should have been 2 or 4)" % (i, observed[i])
    return True, ""


MAX_UNKNOWN = 10


def apply_learned(calib, learned):
    for kind, key, rgb in learned:
        if kind == "tile":
            calib["colors"][key] = rgb
        else:
            calib["empty"][key] = rgb


def revert_learned(calib, learned):
    for kind, key, _ in learned:
        if kind == "tile":
            calib["colors"].pop(key, None)
        else:
            calib["empty"][key] = None


def resolve_unknowns(img, board, unknowns, predicted, calib, interactive):
    if len(unknowns) > MAX_UNKNOWN:
        raise ParseMismatch(
            "%d unknown cells at once — looks like the grid shifted or there's an "
            "overlay on screen (pause / game over / ad)" % len(unknowns)
        )

    learned, blind = [], []
    for idx, rgb in unknowns:
        val = predicted[idx] if predicted else None
        if val is None:
            blind.append((idx, rgb))
        elif val > 0 and val not in calib["colors"]:
            learned.append(("tile", val, tuple(rgb)))
            board[idx] = val
            LOG.info("Learned tile %-5d -> RGB%s (from prediction)", val, rgb)
        elif val == 0 and calib["empty"][idx] is None:
            learned.append(("empty", idx, tuple(rgb)))
            board[idx] = 0
            LOG.info("Learned empty cell %-2d -> RGB%s (from prediction)", idx, rgb)
        else:
            blind.append((idx, rgb))

    if len(blind) > 2:
        raise ParseMismatch(
            "%d cells with an unknown color that can't be derived from the prediction — "
            "this doesn't look like new tiles" % len(blind)
        )

    for idx, rgb in blind:
        crop = "unknown_cell%d_%d-%d-%d.png" % (idx, rgb[0], rgb[1], rgb[2])
        img.crop(vision.cell_rects(calib)[idx]).save(crop)
        if not interactive:
            raise ParseMismatch(
                "unknown color RGB%s in cell %d. Saved the crop to %s. "
                "Restart with --learn to specify the value." % (rgb, idx, crop)
            )
        LOG.warning("Unknown color RGB%s in cell %d (crop: %s)", rgb, idx, crop)
        val = int(input("  What value is in this cell? (0 = empty) > ").strip())
        learned.append(("tile", val, tuple(rgb)) if val else ("empty", idx, tuple(rgb)))
        board[idx] = val
        LOG.info("Remembered: %s -> RGB%s", val if val else "empty", rgb)

    return board, learned


def play(dev, calib, max_moves=5000, min_delay=DEFAULT_MIN_DELAY,
         max_delay=DEFAULT_MAX_DELAY, dry_run=False,
         interactive=False, calib_path=vision.DEFAULT_CALIB,
         continue_past_win=True, cap_tile=0, brain=None, fast_read=False):
    if brain is None:
        brain = get_best_move
    try:
        _brain_wants_score = "score" in inspect.signature(brain).parameters
    except (TypeError, ValueError):
        _brain_wants_score = False
    rect = calib["board"]
    score = moves = mismatches = 0
    predicted = None
    pending = 0
    started = time.time()
    _prev_iter_t = 0.0
    _fastok = _fastfb = 0
    if fast_read:
        LOG.info("FAST-READ enabled (0 unknowns + check_prediction; fallback to settled).")
    expected_app = dev.foreground_activity()
    unlimited = not max_moves or max_moves <= 0
    LOG.info("=== Start on phone (board %s) ===", rect)
    LOG.info("Game in focus: %s", expected_app or "?")
    LOG.info("Move limit: %s   pause between moves: %.1f–%.1f s",
             "unlimited" if unlimited else max_moves, min_delay, max_delay)

    while unlimited or moves < max_moves:
        _now = time.time()
        _spm = (_now - _prev_iter_t) if _prev_iter_t else 0.0
        _prev_iter_t = _now
        if fast_read and predicted is not None:
            img, board, unknowns = read_fast(dev, calib, predicted)
            if img is None:
                _fastfb += 1
                img, board, unknowns = read_settled(dev, calib)
            else:
                _fastok += 1
        else:
            img, board, unknowns = read_settled(dev, calib)

        if len(unknowns) > MAX_UNKNOWN:
            if continue_past_win and recover_from_overlay(dev, calib, img, expected_app):
                predicted = None
                continue

        learned = []
        if unknowns:
            board, learned = resolve_unknowns(
                img, board, unknowns, predicted, calib, interactive)
            apply_learned(calib, learned)

        if predicted is not None:
            ok, why = check_prediction(predicted, board)
            if ok:
                score += pending
                pending = 0
                mismatches = 0
                if learned:
                    vision.save_calib(calib, calib_path)
            else:
                revert_learned(calib, learned)
                pending = 0
                mismatches += 1
                LOG.error("MISMATCH #%d: %s", mismatches, why)
                LOG.error("predicted:\n%s", vision.board_str(predicted))
                LOG.error("on screen:\n%s", vision.board_str(board))
                now = dev.foreground_activity()
                if expected_app and now != expected_app:
                    raise ParseMismatch(
                        "the game is no longer on screen: %s is in focus (was %s). "
                        "I won't swipe blind." % (now or "?", expected_app))
                if mismatches >= 3:
                    raise ParseMismatch("three mismatches in a row — stopping, "
                                        "recalibrate (see annotated.png)")
                LOG.warning("Resynchronizing: playing from what's on screen.")
                predicted = None
                time.sleep(move_delay(min_delay, max_delay))
                continue
        elif learned:
            vision.save_calib(calib, calib_path)

        if game_over(board):
            LOG.info("Game finished — no more moves.")
            break

        if _brain_wants_score:
            mv = brain(board, cap=cap_tile or None, score=score)
        else:
            mv = brain(board, cap=cap_tile or None)
        if mv is None:
            if cap_tile and not game_over(board):
                bmv, bgain = best_bank_move_2048(board)
                if bmv is None:
                    bmv, bgain = best_bank_move(board)
                if bmv is None:
                    LOG.info("Cap %d reached, and nothing left to bank — stopping.", cap_tile)
                    break
                predicted, pending = MOVES[bmv](board)
                moves += 1
                score += pending
                LOG.info("Cap %d reached. BANKING with final '%s' +%d — dumping 2048, "
                         "the game will end. Summary: score=%d max=%d",
                         cap_tile, bmv, bgain, score, max(predicted))
                if not dry_run:
                    swipe_for(dev, bmv, rect)
                    time.sleep(WIN_CLOSE_WAIT)
                board = predicted
                break
            LOG.info("get_best_move found no move — stopping.")
            break

        predicted, pending = MOVES[mv](board)
        moves += 1
        LOG.info("move %4d  %-5s  +%-5d  score=%-7d max=%-5d empty=%-2d  spm=%.2f",
                 moves, mv, pending, score, max(board), board.count(0), _spm)
        if fast_read and moves % 25 == 0:
            LOG.info("[fast-read] accepted=%d fallback=%d (%.0f%% fast)",
                     _fastok, _fastfb, 100.0 * _fastok / max(1, _fastok + _fastfb))

        if dry_run:
            LOG.info("[dry-run] not swiping. Board on screen:\n%s",
                     vision.board_str(board))
            LOG.info("[dry-run] after move '%s' it should become:\n%s",
                     mv, vision.board_str(predicted))
            return score, max(board), moves

        swipe_for(dev, mv, rect)
        delay = move_delay(min_delay, max_delay)
        LOG.debug("pause %.2f s", delay)
        time.sleep(delay)

    if pending and not dry_run and predicted is not None:
        _, final_board, final_unk = read_settled(dev, calib)
        if not final_unk and check_prediction(predicted, final_board)[0]:
            score += pending
            board = final_board

    elapsed = time.time() - started
    best = max(board) if board else 0
    LOG.info("=== End ===")
    LOG.info("Board:\n%s", vision.board_str(board))
    LOG.info("Moves: %d   Score: %d   Max tile: %d   Time: %.0f s (%.1f moves/s)",
             moves, score, best, elapsed, moves / elapsed if elapsed else 0.0)
    return score, best, moves


def main(argv=None):
    p = argparse.ArgumentParser(description="2048 bot that plays on Android via adb")
    p.add_argument("--serial", default=None, help="device serial, if there are multiple devices")
    p.add_argument("--calib", default=vision.DEFAULT_CALIB)
    p.add_argument("--dry-run", action="store_true",
                   help="read the screen and report the move, but do NOT swipe")
    p.add_argument("--learn", action="store_true",
                   help="ask for the value when the color is unknown")
    p.add_argument("--max-moves", type=int, default=0,
                   help="move limit; 0 = unlimited, play to the end of the game")
    p.add_argument("--min-delay", type=float, default=DEFAULT_MIN_DELAY,
                   help="minimum pause between moves, s")
    p.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY,
                   help="maximum pause between moves, s (each move is a random "
                        "value in [min, max])")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default=None)
    p.add_argument("--stop-on-win", action="store_true",
                   help="stop at the congratulations window «You reached 2048!» instead of "
                        "automatically pressing X and continuing to play")
    p.add_argument("--cap-tile", type=int, default=0,
                   help="tile cap: don't make a tile larger than this (e.g. 1024 = "
                        "never build 2048, but pack the board for maximum score)")
    p.add_argument("--nt3-terminal", action="store_true",
                   help="trained n-tuple brain + terminal-bank fix (median ~46k). "
                        "Recommended --depth 3 (fast, p95~60ms).")
    p.add_argument("--record47", action="store_true",
                   help="goal-directed controller, target LOCK IN 47k "
                        "(phase1=terminal-fix d3, from 3×1024 -> D-controller d4).")
    p.add_argument("--fast-read", action="store_true",
                   help="fast move confirmation: 1 screencap + check_prediction instead of "
                        "two settled frames; fallback to read_settled. Speeds up the game ~2×.")
    p.add_argument("--lut", default=None,
                   help="LUT file for the trained network (default %s)" % DEFAULT_MODEL)
    p.add_argument("--depth", type=int, default=4,
                   help="expectimax search depth for --nt3-terminal")
    a = p.parse_args(argv)
    setup_logging(a.log_level, a.log_file)

    from mono2048 import device as phone
    try:
        calib = vision.load_calib(a.calib)
        vision.check_palette(calib)
        dev = phone.Phone.connect(serial=a.serial)
        dev.wake()
    except (vision.CalibrationError, phone.AdbError) as e:
        LOG.error("%s", e)
        return 2

    if a.min_delay > a.max_delay:
        LOG.error("--min-delay (%.2f) is greater than --max-delay (%.2f)", a.min_delay, a.max_delay)
        return 2

    brain = None
    if a.record47:
        from mono2048.brains import record47 as bnn
        lut = a.lut or DEFAULT_MODEL
        bnn.load(lut)
        bnn.get_best_move([2, 0, 0, 0] + [0] * 12, cap=1024, score=0.0)
        brain = lambda board, cap=None, score=0: bnn.get_best_move(
            board, cap=cap or 1024, score=score)
        LOG.info("Brain: RECORD47 goal-directed (%s), target=lock in 47k", lut)
    elif a.nt3_terminal:
        depth = a.depth
        from mono2048.brains import terminal as bnn
        lut = a.lut or DEFAULT_MODEL
        bnn.load(lut)
        bnn.get_best_move([2, 0, 0, 0] + [0] * 12, cap=1024, depth=depth)
        brain = lambda board, cap=None: bnn.get_best_move(board, cap=cap or 1024, depth=depth)
        LOG.info("Brain: trained n-tuple + terminal-fix (%s, depth=%d)", lut, depth)
    else:
        LOG.info("Brain: manual 'mix' heuristic (engine.get_best_move)")

    try:
        play(dev, calib, max_moves=a.max_moves, min_delay=a.min_delay,
             max_delay=a.max_delay, dry_run=a.dry_run,
             interactive=a.learn, calib_path=a.calib,
             continue_past_win=not a.stop_on_win, cap_tile=a.cap_tile, brain=brain,
             fast_read=a.fast_read)
    except KeyboardInterrupt:
        LOG.warning("Interrupted from keyboard.")
    except ParseMismatch as e:
        LOG.error("Stopping: %s", e)
        return 1
    finally:
        vision.save_calib(calib, a.calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
