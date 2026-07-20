import argparse
import logging
import os

from PIL import Image

from mono2048 import vision
from mono2048.engine import setup_logging

LOG = logging.getLogger("bot2048.calibrate")


def parse_values(text):
    vals = [int(v) for v in text.replace("\n", " ").replace(",", " ").split()]
    if len(vals) != 16:
        raise SystemExit("--values must contain exactly 16 numbers, not %d" % len(vals))
    return vals


def _ints(text, n, name):
    vals = [int(v) for v in text.split(",")]
    if len(vals) != n:
        raise SystemExit("%s expects %d comma-separated numbers" % (name, n))
    return vals


def main(argv=None):
    p = argparse.ArgumentParser(description="Calibrate 2048 for your screen")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-phone", action="store_true", help="take a screenshot via adb")
    src.add_argument("--image", help="use an existing PNG (raw only, not from a messenger!)")
    p.add_argument("--serial", default=None, help="device serial, if there are several")
    p.add_argument("--values", default=None,
                   help="16 numbers — what is ACTUALLY on the board right now (0 = empty)")
    p.add_argument("--cols", default=None, help="left edges of columns manually: X0,X1,X2,X3")
    p.add_argument("--rows", default=None, help="top edges of rows manually: Y0,Y1,Y2,Y3")
    p.add_argument("--cell", default=None, help="cell size manually: W,H")
    p.add_argument("--tolerance", type=int, default=vision.DEFAULT_TOLERANCE,
                   help="color tolerance (sum of |dR|+|dG|+|dB|)")
    p.add_argument("--reset", action="store_true", help="start calibration from scratch")
    p.add_argument("--calib", default=vision.DEFAULT_CALIB)
    p.add_argument("--annotated", default="annotated.png")
    p.add_argument("--save-screenshot", default=None, help="where to save the raw screenshot")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = p.parse_args(argv)
    setup_logging(a.log_level)

    if a.from_phone:
        from mono2048 import device as phone
        dev = phone.Phone.connect(serial=a.serial)
        dev.wake()
        LOG.info("In focus: %s", dev.foreground_activity() or "?")
        img = dev.screencap()
        LOG.info("Screenshot %dx%d", img.width, img.height)
        if a.save_screenshot:
            img.save(a.save_screenshot)
            LOG.info("Raw screenshot -> %s", a.save_screenshot)
    else:
        img = Image.open(a.image).convert("RGB")
        LOG.info("Loaded %s (%dx%d)", a.image, img.width, img.height)

    reuse = os.path.exists(a.calib) and not a.reset

    if a.cols or a.rows or a.cell:
        if not (a.cols and a.rows and a.cell):
            raise SystemExit("--cols, --rows and --cell are set together")
        cols = _ints(a.cols, 4, "--cols")
        rows = _ints(a.rows, 4, "--rows")
        cw, ch = _ints(a.cell, 2, "--cell")
        grid = {"cols": cols, "rows": rows, "cell": [cw, ch],
                "board": [cols[0], rows[0], cols[-1] + cw - cols[0], rows[-1] + ch - rows[0]]}
        LOG.info("Grid set manually: cols=%s rows=%s cell=%dx%d", cols, rows, cw, ch)
    elif reuse:
        grid = vision.load_calib(a.calib)
        LOG.info("Taking grid from existing %s", a.calib)
    else:
        grid = vision.find_grid(img)

    if reuse:
        calib = vision.load_calib(a.calib)
        for key in ("cols", "rows", "cell", "board"):
            calib[key] = list(grid[key])
        calib["tolerance"] = a.tolerance
    else:
        calib = vision.new_calib(grid, a.tolerance)
    calib["screen"] = [img.width, img.height]

    colors = vision.cell_colors(img, calib)
    LOG.info("Cell colors:")
    for i, c in enumerate(colors):
        known = vision.classify(i, c, calib)
        label = "?" if known is None else ("empty" if known == 0 else known)
        LOG.info("  [%2d] RGB%-16s -> %s", i, str(c), label)

    if a.values:
        true_board = parse_values(a.values)
        learned = vision.learn_colors(img, calib, true_board)
        LOG.info("Learned new reference samples: %d", len(learned))
        board, unknown = vision.read_board(img, calib)
        if board != true_board:
            LOG.error("CHECK FAILED: read something other than what you specified.")
            LOG.error("read:\n%s", vision.board_str(board))
            LOG.error("expected:\n%s", vision.board_str(true_board))
            LOG.error("Most likely the grid is misaligned — check %s", a.annotated)
        elif unknown:
            LOG.error("Unknown cells remain: %s", [i for i, _ in unknown])
        else:
            LOG.info("Check passed, board reads correctly:\n%s", vision.board_str(board))
        vision.check_palette(calib)

    vision.annotate(img, calib, a.annotated)
    vision.save_calib(calib, a.calib)

    missing = [i for i, v in enumerate(calib["empty"]) if v is None]
    if missing:
        LOG.info("Empty cells not seen yet: %s. The bot will learn them itself.", missing)
    if not a.values:
        LOG.warning("Colors not learned. Restart with --values, specifying "
                    "what is on the board now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
