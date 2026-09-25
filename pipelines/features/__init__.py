"""
Feature builder (model design v3.2, phase B).

One code path turns the pitch files and a few database tables into one row
per team-game: the batting team's inputs for predicting its runs, as they
stood BEFORE the game's prediction point. The same code serves backtests and
(from 2027) daily runs, so a feature can't quietly mean one thing in training
and another live.

Modules, in the order data flows through them:
  inputs.py       load pitch files and tables into one DuckDB connection,
                  and give every game a "data date" (the day it actually
                  finished, which differs for resumed games)
  events.py       pitches -> one row per plate appearance, with who pitched,
                  who batted, the outcome, and which team was in the field
  priors.py       the prior layer: each player's rates blended over this
                  season so far and up to three earlier seasons, shrunk
                  toward the league average by sample size
  pitching.py     starter and bullpen features
  environment.py  park, temperature, roof, league run level, team strength
  lineup.py       lineup quality and platoon share
  build.py        build_features(): assembles one row per team-game

The golden rule, enforced by pipelines/features/leak_test.py: a game's features
may only use games whose data date is BEFORE the game's cutoff date.
"""
