# Postmortem

## The competition

The target game commits a round's score to the leaderboard only when the round **ends**,
and a round ends when you form a 2048 tile. So the score you keep is not "how long you
survived" but "the board you had at the instant you made 2048." That reframes 2048 from a
survival game into a **packing** game: build the densest possible board under a 1024 cap,
then bank a single 2048-forming swipe to commit everything at once. Banked score is about
`Σ C(tile)` with `C(2^e) = (e-1)·2^e`, and because two 1024s banked as a 2048 are worth
more than the pair, the objective is to hoard 1024s and cash out last.

The initial goal was rank 1 (`leader = 50360`, `target = 50364`). It was later corrected
to the practical one: the **top-50 cutoff was a hard 47,000**, and any result below it —
however good a personal best — did not place. That correction changed the optimal policy
(chase the threshold, not the mean), and drove the `record47` controller.

**Outcome:** best committed game **46,416**, final rank **62**. The deployable ceiling
sat just under the 47,000 cutoff.

## The terminal-bank bug (the one real fix)

The recursive expectimax started each node at `best = -1e18` and skipped every move that
would create a 2048 (over the cap). In a position with **no** capped move left, it
returned `-1e18` instead of the reward of the actual bank swipe. Control board:

```
1024 1024 1024 1024
   2    4    2    4
   4    2    4    2
   2    4    2    4
```

There is no capped move here; the final swipe banks 4096, but the buggy search valued the
position at `-1e18`. Consequences: any chance branch where a spawn led to a good bank was
poisoned with `-inf`; deeper search saw the terminal bank more often and could actively
avoid correct endings; and adaptive depth 6–8 results were therefore invalid. The fix:
when no capped move exists, return the best bank reward (a real, positive number), and
only count a swipe as a bank if it genuinely forms a 2048. This lives in
`brains/terminal.py` (`_bank_reward` / the `_search_max` fallback) and is covered by
`tests/test_terminal_brain.py`.

Effect: the pre-fix "searched" median was ~42.4k; the terminal-fixed brain reached a
median of ~46.3k with reliable 4×1024 endgames. This was the single change that made the
bot competitive — no retraining involved.

## Why the direct bank-value bonus failed

The tempting shortcut was to add `Σ C(tile)` (the "bank value" of the board) as a bonus
term so the policy would chase denser boards. It consistently made play **worse**: a
depth-3 sweep gave median 44,176 at weight 0, 42,720 at 0.5, 42,260 at 1.0, 37,608 at 2.0.
The reason is double counting: `Σ C(tile)` mostly encodes score already spent building the
current tiles, while the learned value `V` already predicts *future* score. Adding a
constant proportional to past progress makes the policy short-sighted — it banks early
with a thin tail. The learned TD target already propagates the true terminal bank back
through the game, so the value network needs no such bonus.

Two related corrections from review: `C(2^e)=(e-1)2^e` is not the exact leaderboard score
(it over-credits spawned-4 tiles by 4 each), so `Σ C` tables read a bit high; and
`NSTAGES=5` means stages 0,1,2,3,4+ — boards with 4, 5 and 6 tiles of 1024 all share the
top table, so there is no dedicated high-stage specialization.

## Targeted fine-tune (nt4/nt5) — abandoned

Attempts to fine-tune a copy of the network toward richer 5×1024 endgames (shaped value,
synthetic seeding of near-terminal boards) degraded the working baseline: the shaped
value used a different scale than the warm-start LUT, and most synthetic 5/6×1024 boards
are immediately terminal and yield no useful transitions. The production model stayed the
plain multistage net, `nt3_warm_2m.pkl`.

## The record controller

A two-phase, score-aware controller (`brains/record47.py`) activates goal-directed
expectimax at three 1024s and gates a bank the moment `confirmed_score + 2048_bank_gain`
clears the target. Threshold-chasing without any bank-value term was best (config
"D_aggr": bonus 32k, temperature 500) but only lifted the max from ~46,944 to 47,280, with
**0/120 games over 48k**. Survival-biased chance nodes did not help; the most aggressive
settings landed at the same ~47k. End-to-end, `record47` committed ≥47,000 in **2 of 120**
simulated games. The controller's single-line depth-4/5 search structurally cannot match a
wide search — the gap is pure search width, which a phone cannot afford per move.

## How high could this engine go? (beam-oracle)

To separate "we didn't find it" from "it is unreachable," a wide beam-oracle played from a
real 3×1024 state under **real random spawns**, ranking by the network's own value (not the
poisonous bank-value term), banking only 2048-forming swipes, and measuring
`acc + bank_2048_gain`:

| beam width | max_real | median | over 51k |
|---:|---:|---:|---:|
| 400 | 48,656 | 46,904 | 0 |
| 1000 | 49,880 | 47,620 | 0 |
| 2500 | 50,204 | 49,752 | 0 |

The real bank asymptotes near 50.2k: doubling the width from 1000 to 2500 added only 324.
The oracle is also *optimistic* — it gives each candidate its own spawn and keeps the best,
implicitly cherry-picking lucky spawn sequences a single real game cannot roll. So 50.2k is
"best of ~W parallel worlds," and a real one-trajectory bot lands around 47k. Beam pruning
is not a proof of impossibility, but diminishing returns plus the optimism plus the fact
that even an over-optimistic bank barely grazes 51.1k make the signal strong: **51k is out
of reach under random spawns for any search that runs on the phone.** An online MPC planner
confirmed this from the other side (max 44,416, median 42,704) — the decisive event, lining
up 4×1024 for a forced bank, is hundreds of moves away, far past any rollout horizon.

## Approaches that did not move the ceiling

- **Tile-downgrading expectimax** (relabel big tiles down a rank into the well-learned
  region before evaluating): all variants scored ~44.4k versus a 46.9k control. Downgrading
  makes the network *want* to merge the downgraded 1024s — which is exactly the 2048 bank the
  cap forbids — so it fights the cap-hoard objective. It is a technique for uncapped 2048.
- **Two-model ensemble** (z-score fusion of the base net and a fine-tuned net per root
  action): max 46,948 (+80, noise) and unchanged median; the only real effect was ~+600
  mean, i.e. fewer catastrophic games. The ceiling is structural — it is what the network
  can *build* — and fusing two policies does not change that.

## Fast-read

Waiting for two settled frames per move is safe but slow. `read_fast` takes one screenshot
and accepts it only when it is clean and already matches the predicted board plus exactly
one new 2/4 tile; otherwise it retries briefly and falls back to the settled read. It can
never accept a wrong frame, so it degrades to the safe path under any doubt. In the final
phone game it ran at about **1.6 moves/second with zero desyncs**, roughly 2× faster, which
mattered against the deadline.

## What would actually break 47k

The deployable ceiling of this engine is ~47k; 50,364 is unreachable by any policy that
runs on the phone within a per-move time budget. The one untested theory that could clear
it is a custom **endgame DP tablebase** (in the spirit of macroxue/2048-ai's `plan.h`)
combined with a **perimeter-defense formation**. A tablebase solves the ~350-move endgame
horizon that expectimax, MPC, downgrading and ensembling all fail to see, and a fixed
perimeter formation keeps the board in states the tablebase covers. It requires adapting
the open-source planner to the capped objective (`score + bank > target`, forbidding moves
that break the perimeter) and is a separate, multi-day effort — not attempted here.
