# debug_game24_parser.py
from tasks.gameof24 import Gameof24Parser, Gameof24Prompter
from tasks import utils
import ipdb

def main():
    parser = Gameof24Parser()

    # 초기 상태(스키마 통일 기준)
    state = {
        "original": "4 9 10 13",
        "items": [
            {"id": 0, "value": 4.0, "expr": "4"},
            {"id": 1, "value": 9.0, "expr": "9"},
            {"id": 2, "value": 10.0, "expr": "10"},
            {"id": 3, "value": 13.0, "expr": "13"},
        ],
        "items_json": '[{"id":0,"value":4.0,"expr":"4"},{"id":1,"value":9.0,"expr":"9"},{"id":2,"value":10.0,"expr":"10"},{"id":3,"value":13.0,"expr":"13"}]',
        "next_id": 4,
        "depth": 0,
        "method": "got",
        "current": "",
        "phase": 0,
    }

    # JSONL 후보(move). 이 move는 (10 - 4) = 6 을 만들기 위한 예시
    # texts = ['{"pick":[2,0], "op":"-"}\n{"pick":[3,1], "op":"-"}\n']
    texts = ['{"pick":[999,1], "op":"+"}\n']

    prompter = Gameof24Prompter()

    p_tot = prompter.generate_prompt(
        num_branches=5,
        original=state["original"],
        current=state["current"],
        method="tot",
        state=state,
    )

    p_got = prompter.generate_prompt(
        num_branches=5,
        original=state["original"],
        current=state["current"],
        method="got",
        state=state,
    )

    breakpoint()

    next_states = parser.parse_generate_answer(state, texts)

    print("N next states:", len(next_states))
    for i, st in enumerate(next_states[:5]):
        print("----", i)
        print("depth:", st.get("depth"), "next_id:", st.get("next_id"), "invalid:", st.get("invalid_move", False))
        print("items:", st.get("items"))
        print("score:", utils.game24_score(st), "is_goal:", utils.test_game24(st))

if __name__ == "__main__":
    main()
