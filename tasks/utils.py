 # Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import logging
import math
from typing import Dict, List


def game24_score(state: Dict) -> float:
    logging.debug("utils > game24_score is called.")
    """
    높을수록 안 좋게 설계됨!!
    Heuristic:
    - invalid_move => huge penalty
    - prefer fewer remaining items (deeper states)
    - prefer values closer to 24 (min distance among remaining items)
    - penalize extreme magnitudes / non-finite values for stability
    """
    try:
        if state.get("invalid_move"):
            return 1e9
        
        items = state.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return 1e9

        # proximity to 24 (best remaining item)
        min_dist = 1e9
        mag_pen = 0.0
        for it in items:
            v = float(it.get("value", 0.0))
            if not math.isfinite(v):
                return 1e9
            min_dist = min(min_dist, abs(v - 24.0))
            # mild penalty for exploding magnitudes
            mag_pen += max(0.0, abs(v) - 50.0)
        
        # one-step lookahead when only two items remain:
        # prefer states that can reach 24 in a single operation.
        if len(items) == 2:
            a = float(items[0].get("value", 0.0))
            b = float(items[1].get("value", 0.0))
            candidates = [a + b, a - b, b - a, a * b]
            if abs(b) >= 1e-12:
                candidates.append(a / b)
            if abs(a) >= 1e-12:
                candidates.append(b / a)
            next_min = 1e9
            for v in candidates:
                if not math.isfinite(v):
                    continue
                next_min = min(next_min, abs(v - 24.0))
            # Override the "closest current item" heuristic with one-step feasibility.
            min_dist = next_min

        # prefer depeer (fewer items remaining)
        depth_pen = 10.0 * (len(items) - 1)

        return float(min_dist) + float(depth_pen) + 0.01 * float(mag_pen)
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