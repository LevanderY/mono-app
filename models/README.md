# Model

The bot plays with a trained multistage n-tuple value network. The file is a large
binary blob and is **not** committed to git (see the repository `.gitignore`); obtain
it separately or retrain it.

## File

| | |
|---|---|
| Filename | `nt3_warm_2m.pkl` |
| Size | 2,684,354,881 bytes (~2.5 GiB) |
| SHA-256 | `9c749afe91a796a75a1dee56e6716be7491e7947f65ef4a9c5160a1a1d5dc29a` |
| Format | `pickle` (protocol 4) of a dict |

The pickled dict has three keys:

| Key | Type | Meaning |
|---|---|---|
| `lut` | `float32[5, 8, 16777216]` | value tables: `[stage, tuple, index]` |
| `tuples` | list of 8 six-cell tuples | cell layout each table indexes |
| `nstages` | `int` (5) | number of endgame stages |

`stage` is the number of 1024 tiles on the board (clamped to 4), `tuple` is one of the
8 base 6-cell tuples (each evaluated over 8 board symmetries), and `index` is the
24-bit packing of the six 4-bit tile exponents. See `docs/architecture.md`.

## How to obtain it

This repository is **source-only** — the 2.5 GiB binary is not distributed here. Get a
model one of two ways.

- **Train one** (produces a *behaviourally equivalent* net, not a byte-identical copy —
  training uses random seeds and lock-free Hogwild updates; needs ~8 GiB RAM and hours of
  CPU):

  ```
  python scripts/train.py --save models/trained.pkl
  ```

  The shipped model was warm-started from a single-stage net of the same architecture and
  then trained multistage. That single-stage checkpoint is *not* distributed, so
  `--init-from <single_stage.pkl>` only reproduces the exact recipe if you first train (or
  supply) that net. Point the bot at your model with `--lut models/trained.pkl`, or rename
  it to `models/nt3_warm_2m.pkl`.

- **Supply your own copy** of `nt3_warm_2m.pkl` at `models/nt3_warm_2m.pkl`, then verify it
  against the SHA-256 above:

  ```
  shasum -a 256 models/nt3_warm_2m.pkl
  ```

Without a model, the device brains (`--nt3-terminal`, `--record47`) cannot load and the
model-dependent tests skip; the heuristic engine and the vision/device layers still work.
