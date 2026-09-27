"""
Late-game engine (design 6.3): 8-inning score grid -> final score grid.

The models predict runs through 8 innings. The 9th inning and extras are
played "by the rules", exactly, state by state, with no random draws:

  top of the 9th     the away team bats. How likely it is to score depends
                     on the score going in (teams use their closer to protect
                     small leads, lesser relievers in blowouts): 5 states
  bottom of the 9th  skipped if the home team leads. Otherwise the home team
                     bats until it takes the lead (walk-off: the game stops
                     at the winning run, except a home run can add more)
  extras             repeated innings with the automatic runner on second
                     (regular season), same walk-off rule, until the chance
                     of a tie left is below one in a million

Per half inning, runs follow a hurdle model: first the chance of scoring at
all, then how many if any. A team's chance of scoring is scaled by how good
it is: odds(score) x (team's expected runs per inning / league's)^g, with g
fitted. Everything is fitted on the fold's training seasons only.

Fitting data: tables built from the pitch files (late_innings), one row per
half inning from the 9th on, with the score going in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

KMAX = 8          # runs in one half inning, 8 = "8 or more"
GRID = 40         # final scores 0..39 tracked
TOP_STATES = ["hl4", "hl13", "tie", "ht13", "ht4"]   # home leads 4+, 1-3, tied, trails 1-3, 4+
BOT_STATES = [0, 1, 2, 3, 4]                         # home trails by d (4 = 4+) going into its half


def top_state(home_minus_away: int) -> str:
    d = home_minus_away
    return "hl4" if d >= 4 else "hl13" if d >= 1 else "tie" if d == 0 else "ht13" if d >= -3 else "ht4"


_H, _A = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
_TOP_MASK = {st: (np.vectorize(top_state)(_H - _A) == st) for st in TOP_STATES}


def _odds_scale(p, r, g, lr=1.0):
    """Odds of scoring x (team strength)^g x (league level change), design E8b:
    r = the team's expected runs relative to the league's forecast level,
    lr = the forecast league level relative to the training seasons' level."""
    o = p / (1 - p) * np.power(r, g) * lr
    return o / (1 + o)


@dataclass
class LateGame:
    top_p: dict = field(default_factory=dict)        # P(score > 0) by top state
    bot_p: dict = field(default_factory=dict)        # P(score > 0) by bottom state (untruncated)
    ext_top_p: float = 0.6
    ext_bot_p: float = 0.6
    shape: np.ndarray = None                          # P(runs = k | runs > 0), k = 1..KMAX
    ext_shape: np.ndarray = None
    walkoff_extra: np.ndarray = None                  # P(margin = 1, 2, 3, 4+) for a walk-off
    g: float = 1.0

    # ---- one half inning ----
    def _dist(self, p, shape):
        d = np.zeros(KMAX + 1)
        d[0] = 1 - p
        d[1:] = p * shape
        return d

    def fit(self, li, rel):
        """li: late_innings rows (game_id, inning, bot, start_bat, runs, h8, a8)
        with the score going into each half. rel: game_id -> (home_r, away_r),
        each team's expected runs per inning relative to the league."""
        li = li.copy()
        li["runs_c"] = li.runs.clip(upper=KMAX)
        # score going into each half inning
        li["opp_start"] = np.nan
        # home - away going into top of 9th = h8 - a8
        top9 = li[(li.inning == 9) & ~li.bot]
        pos = li[li.runs > 0]
        self.shape = np.bincount(pos.runs_c.astype(int), minlength=KMAX + 1)[1:].astype(float)
        self.shape /= self.shape.sum()
        ext = li[li.inning >= 10]
        ext_pos = ext[(ext.runs > 0) & ~ext.bot]
        self.ext_shape = np.bincount(ext_pos.runs_c.astype(int), minlength=KMAX + 1)[1:].astype(float)
        self.ext_shape /= self.ext_shape.sum()

        # top of 9th: scoring chance by state, and g from the team scaling
        top9 = top9.assign(state=[top_state(int(h - a)) for h, a in zip(top9.h8, top9.a8)],
                           r=[rel.get(g, (1.0, 1.0))[1] for g in top9.game_id],
                           lr=[(tuple(rel.get(g, (1.0, 1.0))) + (1.0,))[2] for g in top9.game_id],
                           y=(top9.runs > 0).astype(float))
        for s in TOP_STATES:
            x = top9[top9.state == s]
            self.top_p[s] = float((x.y.sum() + 1) / (len(x) + 2))

        def nll(g):
            p = _odds_scale(np.array([self.top_p[s] for s in top9.state]), top9.r.to_numpy(), g, top9.lr.to_numpy())
            y = top9.y.to_numpy()
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.g = float(minimize_scalar(nll, bounds=(0, 3), method="bounded").x)

        # bottom of 9th: home trails by d >= 0 after the top. Observed runs are
        # cut off by a walk-off, so the chance of scoring is fitted from
        # "scored at all" (never cut off at zero): P(runs > 0).
        b9 = li[(li.inning == 9) & li.bot].copy()
        t9 = li[(li.inning == 9) & ~li.bot].set_index("game_id").runs
        b9["d"] = (b9.a8 + b9.game_id.map(t9)) - b9.h8
        b9 = b9[b9.d >= 0]
        b9["dc"] = b9.d.clip(upper=4)
        for d in BOT_STATES:
            x = b9[b9.dc == d]
            self.bot_p[d] = float(((x.runs > 0).sum() + 1) / (len(x) + 2))

        # extras (automatic runner): chance of scoring, top and bottom
        et = ext[~ext.bot]
        eb = ext[ext.bot]
        self.ext_top_p = float(((et.runs > 0).sum() + 1) / (len(et) + 2))
        self.ext_bot_p = float(((eb.runs > 0).sum() + 1) / (len(eb) + 2))

        # walk-off margin: how many runs the home team won by in walk-offs
        wo = b9[b9.runs > b9.d]
        m = (wo.runs - wo.d).clip(upper=4).astype(int)
        self.walkoff_extra = np.bincount(m, minlength=5)[1:].astype(float) + 0.5
        self.walkoff_extra /= self.walkoff_extra.sum()
        return self

    # ---- the exact engine ----
    def final_grid(self, g8, home_r=1.0, away_r=1.0, league_r=1.0):
        """g8: (n+1)x(n+1) grid of P(home = i, away = j) after 8 innings.
        Returns a GRIDxGRID grid of the final score. league_r: the forecast
        league level over the training level (passes with exponent 1)."""
        lr = league_r
        n = g8.shape[0]
        out = np.zeros((GRID, GRID))
        cur = np.zeros((GRID, GRID))
        cur[:n, :n] = g8[:GRID, :GRID]
        H, A = _H, _A
        # --- top of 9th: away bats, state from home - away
        nxt = np.zeros_like(cur)
        for s in TOP_STATES:
            mask = _TOP_MASK[s]
            p = _odds_scale(self.top_p[s], away_r, self.g, lr)
            dist = self._dist(p, self.shape)
            part = np.where(mask, cur, 0.0)
            for k, pk in enumerate(dist):
                nxt[:, k:] += pk * part[:, :GRID - k]
        cur = nxt
        # home leads after the top: game over
        lead = H > A
        out += np.where(lead, cur, 0.0)
        cur = np.where(lead, 0.0, cur)
        # --- bottom of 9th with walk-off
        cur, out = self._bottom(cur, out, H, A, lambda d: _odds_scale(self.bot_p[min(d, 4)], home_r, self.g, lr), self.shape)
        # --- extras
        for _ in range(30):
            if cur.sum() < 1e-6:
                break
            nxt = np.zeros_like(cur)
            dist = self._dist(_odds_scale(self.ext_top_p, away_r, self.g, lr), self.ext_shape)
            for k, pk in enumerate(dist):
                nxt[:, k:] += pk * cur[:, :GRID - k]
            cur = nxt
            cur, out = self._bottom(cur, out, H, A, lambda d: _odds_scale(self.ext_bot_p, home_r, self.g, lr), self.ext_shape)
        # anything left (vanishingly small): split evenly as a home or away one-run win
        if cur.sum() > 0:
            out[1:, :] += 0.5 * cur[:-1, :]
            out[:, 1:] += 0.5 * cur[:, :-1]
        return out / out.sum()

    def _bottom(self, cur, out, H, A, p_of_d, shape):
        """Home bats trailing by d >= 0. Scoring more than d ends the game with
        a walk-off margin; exactly d ties (on to extras); less, away wins."""
        new_cur = np.zeros_like(cur)
        d_arr = A - H
        live = cur > 1e-15
        for d in np.unique(d_arr[live & (d_arr >= 0)]):
            d = int(d)
            mask = (d_arr == d) & live
            dist = self._dist(p_of_d(d), shape)
            part = np.where(mask, cur, 0.0)
            # less than d (and not reaching the tie): away wins, final
            for k in range(0, min(d, KMAX + 1)):
                out[k:, :] += dist[k] * part[:GRID - k, :]
            p_tie = dist[d] if d <= KMAX else 0.0
            if d <= KMAX:
                new_cur[d:, :] += p_tie * part[:GRID - d, :]
            p_win = dist[d + 1:].sum() if d + 1 <= KMAX else 0.0
            for m, pm in enumerate(self.walkoff_extra, start=1):
                k = d + m
                if k < GRID:
                    out[k:, :] += p_win * pm * part[:GRID - k, :]
        return new_cur, out

    def to_json(self):
        return {"top_p": self.top_p, "bot_p": {str(k): v for k, v in self.bot_p.items()},
                "ext_top_p": self.ext_top_p, "ext_bot_p": self.ext_bot_p, "g": self.g,
                "shape": self.shape.tolist(), "ext_shape": self.ext_shape.tolist(),
                "walkoff_margin": self.walkoff_extra.tolist()}


def build_late_innings(inputs_folder: str, seasons: list[int], out_path: str) -> None:
    """Half innings from the 9th on, with the score going in, from the pitch
    files: the fitting data for the engine.

        python -c "from pipelines.models.late_game import build_late_innings as b; b('inputs', [2021, 2022, 2023, 2024], 'late_innings.parquet')"

    Runs in a half inning = the batting team's score at the first pitch of
    its next half inning (or its final score) minus its score at the first
    pitch of this one. Regular season, 9+ inning games, resumed games left out.
    """
    from pipelines.features import inputs
    con = inputs.connect(inputs_folder, seasons=seasons)
    df = con.execute("""
        with hs as (
          select game_id, inning, (inning_topbot = 'Bot') as bot,
                 arg_min(bat_score, at_bat_id * 1000 + pitch_number) as start_bat
          from pitches where inning >= 9 and game_type = 'R' group by 1, 2, 3),
        fin as (
          select g.game_id, g.season,
                 max(case when t.team_id = g.home_team then t.runs_8 end) h8,
                 max(case when t.team_id = g.away_team then t.runs_8 end) a8,
                 max(case when t.team_id = g.home_team then t.runs_total end) hf,
                 max(case when t.team_id = g.away_team then t.runs_total end) af
          from games g join team_runs t using (game_id)
          where g.game_type = 'R' and g.innings >= 9 and not g.resumed group by 1, 2)
        select f.*, hs.inning, hs.bot, hs.start_bat
        from fin f join hs using (game_id) order by game_id, inning, bot
    """).fetchdf()
    df["bot"] = df.bot.astype(bool)
    df["final"] = np.where(df.bot, df.hf, df.af)
    df["next"] = df.groupby(["game_id", "bot"]).start_bat.shift(-1)
    df["runs"] = df.next.fillna(df.final) - df.start_bat
    if (df.runs < 0).any():
        raise ValueError("negative runs in a half inning: pitch scores and finals disagree")
    df.to_parquet(out_path)
