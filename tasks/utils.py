 # Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import logging
import math
from functools import lru_cache
from typing import Dict, List, Tuple

TARGET = 24.0
EPS = 1e-6

def _round_key(vals: List[float], ndigits: int = 6) -> Tuple[float, ...]:
    # canonicalize for caching: sort + round
    return tuple(sorted(round(v, ndigits) for v in vals))

@lru_cache(maxsize=200_000)
def _best_residual(key: Tuple[float, ...]) -> float:
    """
    Returns minimal achievable |value - 24| by fully reducing this multiset via +,-,*,/.
    Smaller is better. 0 means solvable within rounding tolerance.
    """
    vals = list(key)
    n = len(vals)

    if n == 0:
        return 1e9
    if n == 1:
        v = vals[0]
        if not math.isfinite(v):
            return 1e9
        return abs(v - TARGET)

    best = 1e9

    # pick two indices i<j
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(n) if k != i and k != j]

            # generate results; include both orders for - and /
            candidates = [
                a + b,
                a * b,
                a - b,
                b - a,
            ]
            if abs(b) > EPS:
                candidates.append(a / b)
            if abs(a) > EPS:
                candidates.append(b / a)

            for c in candidates:
                if not math.isfinite(c):
                    continue
                # optional pruning for stability (prevents huge blowups)
                if abs(c) > 1e6:
                    continue

                nxt = rest + [c]
                r = _best_residual(_round_key(nxt))
                if r < best:
                    best = r
                    if best <= 0.0 + 1e-9:
                        return 0.0

    return best

def game24_score(state: Dict) -> float:
    logging.debug("utils > game24_score is called.")
    """
    Smaller is better.
    Primary: exact/near-exact residual distance to 24 from remaining items.
    Secondary: very light depth + magnitude tie-breakers for stability.
    """
    try:
        if state.get("invalid_move"):
            return 1e9
        
        items = state.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return 1e9

        vals = []
        mag_pen = 0.0
        for it in items:
            v = float(it.get("value", 0.0))
            if not math.isfinite(v):
                return 1e9
            vals.append(v)
            # much gentler magnitude penalty
            mag_pen += max(0.0, abs(v) - 100.0)
        
        residual = _best_residual(_round_key(vals))

        # depth as *tie-breaker only* (very small weight)
        # fewer items remaining is slightly preferred among equal residuals
        depth_tiebreak = 0.05 * (len(items) - 1)

        return float(residual) + depth_tiebreak + 0.001 * float(mag_pen)
    except Exception:
        return 1e9

def test_game24(state: Dict) -> bool:
    """
    Success condition: exactly one item remains and it equals 24 within tolerance.
    """
    logging.debug("utils > test_game24 is called.")
    try:
        if state.get("invalid_move"):
            return False
        items = state.get("items", [])
        if not isinstance(items, list) or len(items) != 1:
            return False
        v = float(items[0].get("value", 0.0))

        return abs(v - 24.0) < 1e-6
    except Exception:
        return False