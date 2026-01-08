# debug_game24_e2e.py
import argparse
import json
import logging
import math
from typing import Any, Dict, List, Tuple, Optional

from tasks.gameof24 import Gameof24Prompter, Gameof24Parser
from tasks import utils


logger = logging.getLogger(__name__)


def _json_dumps_compact(obj: Any) -> str:
    # Compact JSON (stable for parser)
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def make_initial_state(numbers: List[float], method: str) -> Dict:
    """
    Build a state that matches your unified schema:
      - original: "4 9 10 13" (string)
      - items: [{id,value,expr}, ...]
      - items_json: json string of items
      - next_id, depth
      - plus GoT framework fields: method/current/phase
    """
    items = []
    for i, v in enumerate(numbers):
        fv = float(v)
        expr = str(int(fv)) if float(fv).is_integer() else str(fv)
        items.append({"id": i, "value": fv, "expr": expr})

    state = {
        "original": " ".join(str(int(x)) if float(x).is_integer() else str(x) for x in numbers),
        "items": items,
        "items_json": _json_dumps_compact(items),
        "next_id": len(items),
        "depth": 0,
        "method": method,
        "current": "",
        "phase": 2,  # for tot/got routing in your prompter
    }
    return state


class Replay24LM:
    """
    Deterministic stub LM for E2E debugging.

    It reads CurrentItems from the prompt (your prompter includes items_json)
    and returns JSONL candidate moves. First candidate is usually "good",
    remaining are "noisy but valid" to exercise Score/KeepBestN.
    """

    def __init__(self) -> None:
        self.cost = 0.0
        self.calls = 0

    def _extract_items_from_prompt(self, prompt: str) -> List[Dict]:
        # Your prompt contains:
        # <CurrentItems>
        # [ ...json... ]
        # </CurrentItems>
        start = prompt.find("<CurrentItems>")
        end = prompt.find("</CurrentItems>")
        if start == -1 or end == -1:
            return []
        chunk = prompt[start:end]
        lb = chunk.find("[")
        rb = chunk.rfind("]")
        if lb == -1 or rb == -1 or rb <= lb:
            return []
        try:
            return json.loads(chunk[lb : rb + 1])
        except Exception:
            return []

    def _propose(self, items: List[Dict], B: int) -> List[Dict]:
        # Build a set of valid (pick, op) moves
        ids = [int(it["id"]) for it in items if "id" in it]
        id_to_val = {int(it["id"]): float(it.get("value", 0.0)) for it in items if "id" in it}

        def valid_div(id1: int, id2: int) -> bool:
            # divisor is second picked id (a / b) per your parser policy
            return abs(float(id_to_val.get(id2, 0.0))) >= 1e-12

        all_moves = []
        ops = ["+", "-", "*", "/"]
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                for op in ops:
                    if op == "/" and not valid_div(ids[i], ids[j]):
                        continue
                    all_moves.append({"pick": [ids[i], ids[j]], "op": op})

        # Heuristic “good” move for the known classic case 4,9,10,13:
        # If 4 items exist and those values match, propose (10-4) and (13-9) patterns.
        # Otherwise just return the first B moves.
        values = sorted([float(it.get("value", 0.0)) for it in items])
        good_first: Optional[Dict] = None
        if len(items) == 4 and values == [4.0, 9.0, 10.0, 13.0]:
            # try to locate ids for 10 and 4, and 13 and 9
            id_10 = next(int(it["id"]) for it in items if float(it["value"]) == 10.0)
            id_4 = next(int(it["id"]) for it in items if float(it["value"]) == 4.0)
            good_first = {"pick": [id_10, id_4], "op": "-"}
        elif len(items) == 3:
            # If we have a 6 and 9 and 13, try (13-9)=4 or 9-? etc; leave generic.
            pass
        elif len(items) == 2:
            # If close to 24, try multiply.
            pass

        picked = []
        if good_first is not None and good_first in all_moves:
            picked.append(good_first)

        # Add diverse moves (avoid duplicates)
        seen = set()
        for m in picked:
            seen.add((m["pick"][0], m["pick"][1], m["op"]))

        for m in all_moves:
            key = (m["pick"][0], m["pick"][1], m["op"])
            if key in seen:
                continue
            picked.append(m)
            seen.add(key)
            if len(picked) >= B:
                break

        return picked[:B]

    def query(self, query: str, num_responses: int = 1) -> List[str]:
        """
        Return list[str] to mimic AbstractLanguageModel.query.
        We return ONE completion containing JSONL lines (so parser yields B next states).
        """
        self.calls += 1
        items = self._extract_items_from_prompt(query)

        # Determine requested B from prompt (fallback: 10)
        B = 10
        marker = "output exactly "
        idx = query.lower().find(marker)
        if idx != -1:
            tail = query.lower()[idx + len(marker) : idx + len(marker) + 8]
            num = ""
            for ch in tail:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                try:
                    B = int(num)
                except Exception:
                    pass

        moves = self._propose(items, B)
        jsonl = "\n".join(_json_dumps_compact(m) for m in moves) + "\n"
        return [jsonl]


def beam_search_e2e(
    lm: Any,
    prompter: Gameof24Prompter,
    parser: Gameof24Parser,
    init_state: Dict,
    method: str,
    B: int,
    K: int,
    max_depth: int,
    ipdb_each_step: bool = False,
) -> Tuple[bool, Dict, List[Dict]]:
    """
    Minimal E2E loop:
      state -> prompt -> LM -> parse -> score -> keep top K -> repeat
    """
    frontier = [init_state]
    best_seen = None
    best_score = float("inf")

    for step in range(max_depth):
        next_frontier: List[Dict] = []

        for s in frontier:
            # Sanity checks (schema global)
            assert "items" in s and isinstance(s["items"], list), "state['items'] missing or invalid"
            assert "items_json" in s and isinstance(s["items_json"], str), "state['items_json'] missing or invalid"
            assert "next_id" in s, "state['next_id'] missing"
            assert "depth" in s, "state['depth'] missing"

            prompt = prompter.generate_prompt(
                num_branches=B,
                original=s.get("original", ""),
                current=s.get("current", ""),
                method=method,
                items_json=s.get("items_json", ""),
                phase=s.get("phase", 2),
            )

            if ipdb_each_step:
                import ipdb  # type: ignore
                ipdb.set_trace()

            texts = lm.query(prompt, num_responses=1)
            children = parser.parse_generate_answer(s, texts)

            for c in children:
                sc = utils.game24_score(c)
                c["_score"] = sc
                next_frontier.append(c)

                if sc < best_score:
                    best_score = sc
                    best_seen = c

                if utils.test_game24(c):
                    return True, c, next_frontier

        # KeepBestN(K): smaller score is better (distance+penalty)
        next_frontier.sort(key=lambda x: float(x.get("_score", 1e18)))
        frontier = next_frontier[: max(1, K)]

        logger.info(
            "Step %d/%d | expanded=%d | kept=%d | best_score=%.6f | top0_items=%s",
            step + 1,
            max_depth,
            len(next_frontier),
            len(frontier),
            float(frontier[0].get("_score", 1e18)) if frontier else math.inf,
            str(frontier[0].get("items")) if frontier else "[]",
        )

    return False, (best_seen if best_seen is not None else init_state), frontier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="tot", choices=["tot", "got"])
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--ipdb_each_step", action="store_true")
    ap.add_argument("--numbers", type=str, default="4 9 10 13")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    numbers = [float(x) for x in args.numbers.replace(",", " ").split()]
    prompter = Gameof24Prompter()
    parser = Gameof24Parser()
    lm = Replay24LM()

    init_state = make_initial_state(numbers, method=args.method)

    ok, final_state, last_children = beam_search_e2e(
        lm=lm,
        prompter=prompter,
        parser=parser,
        init_state=init_state,
        method=args.method,
        B=args.B,
        K=args.K,
        max_depth=args.max_depth,
        ipdb_each_step=args.ipdb_each_step,
    )

    print("\n=== E2E RESULT ===")
    print("method:", args.method)
    print("ok:", ok)
    print("lm_calls:", getattr(lm, "calls", None))
    print("final_depth:", final_state.get("depth"))
    print("final_items:", final_state.get("items"))
    print("final_expr:", final_state.get("current"))
    print("final_score:", utils.game24_score(final_state))
    print("is_goal:", utils.test_game24(final_state))

    # show top few children from last expansion (scored)
    last_children.sort(key=lambda x: float(x.get("_score", 1e18)))
    print("\nTop candidates (last expansion):")
    for i, st in enumerate(last_children[: min(10, len(last_children))]):
        print(
            f"- #{i} score={st.get('_score'):.6f} depth={st.get('depth')} invalid={st.get('invalid_move', False)} items={st.get('items')}"
        )


if __name__ == "__main__":
    main()

