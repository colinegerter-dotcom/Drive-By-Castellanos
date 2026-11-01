"""Offline tests for the market code (pipelines/models/market_d.py).

Added in the post-mortem (9 Oct 2026): the market code decided G4 but had no
tests. No network, no odds files: tiny hand-built books whose answers are
worked out by hand.

Run: python tests/test_market_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipelines.models import market_d as md  # noqa: E402

failures = []


def check(label, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {float(got):.9f}" + ("" if ok else f", want {float(want):.9f}"))
    if not ok:
        failures.append(label)


print("devig: -150 / +130 (implied 0.6 and 0.434783, total 1.034783)")
a, b = 0.6, 100 / 230
# multiplicative: scale both sides by the same factor
check("multiplicative", md.devig_home(a, b, "multiplicative"), a / (a + b))
# with two outcomes Shin's method reduces to taking half the margin off each side
check("shin = additive for two outcomes", md.devig_home(a, b, "shin"), a - (a + b - 1) / 2, 1e-12)
# power: a^k + b^k = 1, solved independently here by bisection on k
lo, hi = 1.0, 4.0
for _ in range(200):
    k = (lo + hi) / 2
    lo, hi = (k, hi) if a ** k + b ** k > 1 else (lo, k)
check("power", md.devig_home(a, b, "power"), a ** ((lo + hi) / 2), 1e-12)

print("devig: -110 / -110 gives exactly 0.5 for every method")
for m in md.METHODS:
    check(m, md.devig_home(110 / 210, 110 / 210, m), 0.5, 1e-12)

print("prices")
check("to_dec(-150)", md.to_dec(-150), 1 + 100 / 150)
check("to_dec(+130)", md.to_dec(130), 2.3)

print("bet rule and line move on three hand-built games")
# game 1: model 0.60 vs market 0.52, best home price +100 (2.0): blended
#   0.52 + ... , EV = b * 2.0 - 1 > 0 -> home bet; close 0.55 -> move +0.03
# game 2: model equals market -> never a bet
# game 3: model 0.40 vs market 0.50 but the raw gap 0.10 is over the 8-point cap -> no bet
f = pd.DataFrame({"date": pd.to_datetime(["2026-05-01"] * 3), "q_t": [0.52, 0.50, 0.50], "q_c": [0.55, 0.50, 0.45],
                  "p": [0.60, 0.50, 0.40], "dec_h": [2.0, 1.95, 1.95], "dec_a": [1.85, 1.95, 1.95], "y": [1.0, 0.0, 0.0]})
bb = md.bets(f, w=0.30, tau=0.0125)
check("one bet", len(bb), 1)
check("bet is on the home side", bool(bb.home.iloc[0]), True)
check("line move toward the bet = q_c - q_t", bb.mv.iloc[0], 0.03, 1e-12)
blend = md.expit(md.logit(0.52) + 0.30 * (md.logit(0.60) - md.logit(0.52)))
check("EV at the blended probability", bb.ev.iloc[0], blend * 2.0 - 1, 1e-12)
check("price CLV = close fair prob x price - 1", bb.clv.iloc[0], 0.55 * 2.0 - 1, 1e-12)
check("won at +100 pays 1 unit", bb.pnl.iloc[0], 1.0, 1e-12)

print("an away bet moves the other way")
g = f.iloc[[0]].assign(q_t=0.52, p=0.46, q_c=0.50, dec_h=1.80, dec_a=2.2)
ba = md.bets(g, w=0.30, tau=0.0125)
check("one away bet", len(ba), 1)
check("away move = -(q_c - q_t)", ba.mv.iloc[0], 0.02, 1e-12)

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nall market_d checks passed")
