# mono2048

This is a bot that plays 2048 by itself on a real Android phone. Not in a browser
and not in a drawn-up simulator: it connects to the phone over USB, captures the
screen, recognizes the tiles, picks a move, and performs a swipe through ADB.

We wrote it for a contest inside the mono app. The rules there are a little
sneaky: your score only makes it onto the leaderboard when the game ends by
creating a 2048 tile. So simply surviving as long as possible isn't enough. You
have to keep the ceiling at 1024, pack the board tightly, and at the very end
form a 2048 with a single swipe and "bank" the entire score.

The final result of this story: **46,416 points, 62nd place, and 584 points short
of the top 50**.

## How it all started

At first this was an ordinary script with a few heuristics: keep the big tile in
a corner, don't break the ordering, leave empty cells, and don't make obviously
bad moves. Then came board recognition by color, ADB swipes, a check on every
move, and finally a trained n-tuple model with an expectimax search.

After that the race with the leaderboard began. The record kept climbing, the
deadline drew near, and the model very consistently hit a wall at around 46–47
thousand. We tried a deeper search, additional training, synthetic endgames,
bank-value bonuses, an ensemble of two models, tile-downgrading, MPC, and even a
wide beam-oracle. Some ideas gave nothing, some made the game worse, but every
failed experiment explained a little better exactly where the ceiling lay.

## How the bot sees the phone

Roughly this happens on every move:

1. The bot takes a screenshot of the phone through ADB.
2. `vision.py` looks at the colors of the 16 cells and turns them into an
   ordinary board of numbers `2, 4, 8, ...`.
3. The brain evaluates the possible swipes and returns a direction.
4. The bot precomputes what the board should become after that swipe.
5. The phone receives the swipe.
6. The next frame has to match the prediction plus exactly one new `2` or `4`
   tile. If it doesn't match, the bot doesn't play at random: it resynchronizes
   or stops.

For the final attempt we added `--fast-read`. The normal mode waits for two
identical frames in a row, while the fast one accepts the very first frame if it
completely matches the prediction. This sped the phone up to about **1.6 moves
per second** without a single desync in the final game.

## What the model is and how we trained it

Inside it isn't a neural network in the classic sense, but a large n-tuple value
network. It looks not at the whole board at once, but at eight six-cell templates
across eight symmetries. For each combination of tiles there is a value in a big
lookup table.

The network is multistage: early game, one 1024, two 1024s, three 1024s, and the
endgame with four 1024s are evaluated by different tables. This matters, because
a good move on an empty board and a good move in a tight endgame are completely
different things.

Training was self-play:

- the bot played millions of simulated games against itself;
- after every move it updated the value of the previous state via afterstate TD(0);
- temporal coherence automatically reduced the step size for unstable entries;
- several processes updated shared tables at the same time without locks
  (Hogwild training);
- the multistage model started from an already-trained single-stage model, not
  from zeros.

The final LUT weighs about **2.5 GiB**. It's not very elegant, but the lookup
works fast even while playing on the phone.

## The bug that cost us a ton of time

The most useful discovery of the whole project was not a new model, but a single
wrong value in the search.

When there were no moves left under the 1024 ceiling, the old expectimax returned
`-1e18`. In reality this isn't a "terrible losing state," but the moment when you
need to make the final swipe, form 2048, and bank the score. Because of `-1e18`,
the search was afraid of good, tight finishes and could literally avoid the
correct ending.

After the terminal-bank fix, the searched median rose roughly from **42.4k to
46.3k** without any retraining. This was the biggest real gain of the entire
contest.

## The last evening

Toward the end it turned out that the top 50 required at least **47,000**. The
stable model gave around 46.3k, so we made a separate `record47` controller. Up
to three 1024 tiles it played with the usual strong policy, and after that it
switched into hunting mode right at the threshold. As soon as the current score
plus an available 2048 swipe reached 47k, the bot was supposed to end the game
immediately and lock in the result.

Offline this worked in 2 of 120 games. The chance is small, but it was the only
tested option that actually showed **47,004**.

The second-to-last phone game reached **46,416**. It was a new personal record,
but it fell just 584 points short of the top 50. The clock already read about
23:29, so we launched a final attempt with a perfect `4 + 2` start on the edge
and the just-written fast-read.

The last game flew very fast: 1771 moves in about 19 minutes, all frames went
through the fast-path, not a single read error. But the RNG didn't cooperate this
time: the bot didn't finish building the fourth 1024 and banked only **34,984**.
The game ended at 23:48, and by 23:57 it was already clear there was no time for
another full attempt.

And so it stayed: **46,416, 62nd place, minus 584 points from the result we
needed**. A little painful, but now it's a proper, living project instead of a
29-gigabyte folder of logs, checkpoints, and `final_final_really_final.pkl` files.

A more detailed technical breakdown is in
[`docs/postmortem.md`](docs/postmortem.md), and the module diagram is in
[`docs/architecture.md`](docs/architecture.md).
