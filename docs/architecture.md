# Architecture

The bot is a perception → decision → action loop over ADB, plus an offline training and
evaluation stack for the value network it plays with.

```
screen ── vision ──▶ board ── brain ──▶ move ── device ──▶ swipe
  ▲                                                            │
  └──────────────── self-check (predict vs observe) ◀──────────┘
```

## Board representations (`engine.py`)

Two representations coexist:

- **Value list** — 16 ints `[0, 2, 4, …, 2048]`, row-major. This is what vision produces
  and what the bot predicts, so mismatches are human-readable.
- **Packed board** — 4 rows of 16 bits, each cell a 4-bit *exponent* (`0` = empty,
  `e` = tile `2^e`). All search runs on this. Row moves and their score are precomputed
  into 65536-entry tables (`_ROW_RESULT`, `_ROW_SCORE`, `_ROW_REV`), so a whole-board
  move is four table lookups; columns are handled by transposing the packed board.

**Scoring.** A committed game's score is essentially `Σ C(tile)` with
`C(2^e) = (e-1)·2^e` (so `1024 → 9216`, `2048 → 20480`). Two equal 1024s banked as a
2048 are worth more than the pair, which is why the strategy hoards 1024s and banks a
2048 last. The formula slightly overestimates the real leaderboard score by `4 ×`
(number of spawned-4 tiles), since a spawned 4 scores nothing but `C` credits it 4; for
comparing moves on one board this bias is near-constant.

The manual `get_best_move` heuristic ("mix") scores a packed board from bank value
(`C`), a single global snake ordering, monotonicity, empties, and adjacent merges, and
drives an adaptive-depth expectimax. It is the fallback brain; the trained network is
much stronger.

## Vision (`vision.py`)

Everything device-specific lives in `calib.json`; nothing is hard-coded. Two facts about
the real app shape the module: cell gaps are wide (you cannot just quarter the board
rectangle — the sample would land in a gap, so the grid is stored as explicit column and
row edges with a fixed cell size), and the board background is a vertical gradient (so an
empty cell has a *per-position* reference colour while tiles share one palette). A cell's
colour is the mode of pixels in a ring inside it, avoiding rounded corners and the digit.
`classify` picks the nearest reference within a tolerance and returns "unknown" when two
references tie. Grid geometry can be found automatically from the gap colour.

## Device (`device.py`)

A thin ADB wrapper: `screencap` (raw framebuffer gzipped on the phone, ~4× faster than
on-device PNG, with a PNG fallback), `swipe`, `tap`, foreground-activity/lock detection,
and `wake` (sets `stay_on_while_plugged_in` so the screen doesn't sleep mid-game).

## Main loop (`cli.py`)

Each turn: read the board, run the brain, predict the post-move board, swipe, and on the
next read verify the screen equals the prediction plus exactly one new 2/4 tile. A
mismatch is treated as an error — the bot resyncs from the screen or stops after three in
a row rather than playing an imagined game. The same check gives free colour learning: a
merged tile's value is known, so its colour can be recorded once the prediction confirms.

- `read_settled` waits until two consecutive frames agree on all cell colours (the safe
  read, immune to mid-animation frames).
- `read_fast` (`--fast-read`) takes one screenshot and accepts it only if it is clean and
  matches the prediction; otherwise it retries briefly and falls back to `read_settled`.
  It can never accept a wrong frame — it only skips the second settled read when the first
  is already provably correct.
- Overlays (server "syncing next round", the "you made 2048" dialog) are handled by
  waiting the server out first, then dismissing the win dialog if needed.
- **Cap-gate banking:** under `--cap-tile 1024` the brain returns `None` when every move
  would exceed the cap. That is not a stop — it is the moment to bank: the loop plays one
  2048-forming swipe (`best_bank_move_2048`, falling back to the max-gain move), which ends
  the round and commits the score.

## Trained brain (`brains/terminal.py`)

The value network is a multistage n-tuple network: 8 base 6-cell tuples, each evaluated
over the 8 board symmetries (weight sharing), summed from a lookup table
`LUT[stage, tuple, index]`. `stage` is the number of 1024 tiles on the board (0…4), which
only grows within a capped game, so late-game and early-game positions get separate value
tables instead of averaging into one. `index` packs the six cells' 4-bit exponents into
24 bits. Evaluation (`_V`) is a numba kernel.

The brain runs a capped expectimax with a transposition table: max nodes try each legal
move that keeps the max tile ≤ cap; chance nodes average over spawned 2/4 (sampling
empties when there are many). The **terminal-bank fix** is the key correctness point — in
a position with no capped move left, the search returns the real best bank reward instead
of the `-1e18` sentinel it used to return, so the search no longer avoids the
board-completing bank (see the postmortem).

## Qualification controller (`brains/record47.py`)

A two-phase controller aimed at *committing* a target score (47,000), not maximizing the
average. Below three 1024s it plays the terminal brain (depth 3). At three or more it
switches to a goal-directed expectimax (depth 4) whose leaf utility adds a sigmoid bonus
for crossing the target. A gate fires every move: if the confirmed score plus the gain of
a 2048-*forming* swipe already clears the target, it banks immediately — the 2048
guarantees the round ends and the score commits.

## Training (`scripts/train.py`)

Afterstate TD(0) with temporal-coherence (TC) adaptive step sizes, played greedily under
the 1024 cap; the terminal target at the cap boundary is the best bank swipe's reward. TC
weights each cell's step by `|Σδ| / Σ|δ|` for faster, higher convergence than a fixed
rate. Training is Hogwild — several worker processes update three shared `float32`
tables (`LUT`, `E`, `A`, each `[5, 8, 16777216]` ≈ 2.5 GiB) without locks. The shipped
model was warm-started from a single-stage net and then trained multistage.
`scripts/evaluate.py` plays the trained network with the same expectimax family used
on-device and reports the score distribution.

Numba does not link `@njit` kernels across module boundaries, so `train.py` and
`evaluate.py` each carry their own copy of the shared board kernels rather than importing
them from the package.
