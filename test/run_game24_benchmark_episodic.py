import argparse
import json
import os

from datetime import datetime
from tqdm import tqdm

from graph_of_thoughts import controller, language_models
from tasks.gameof24 import (
    Gameof24Prompter,
    Gameof24Parser,
    got,
    tot,
    tot2,
    io,
    cot,
)
from tasks import utils  # (선택) 디버깅/스코어 확인용


def _compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

METHODS = {
    "io": io,
    "cot": cot,
    "tot": tot,
    "tot2": tot2,
    "got": got,
}


def run_benchmark(task: str, method: str, model_name: str, benchmark_path: str) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = f"../logs/output_{task}_{method}_{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # ---------
    # Problem input (Game of 24)
    # Language Model (HF) - Initialize once
    # ---------
    lm = language_models.ChatGPT(
        "../config.json",
        model_name=model_name,
        cache=False,
    )

    # ---------
    # Iterate over Benchmark File
    # ---------
    with open(benchmark_path, "r") as f:
        lines = f.readlines()

    method_fn = METHODS[method]

    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Solving Game24"):
        problem = json.loads(line)
        # Input example: "2 5 8 11"
        input_str = problem["input"]
        numbers = [float(x) for x in input_str.split()]
        original = input_str

        items = []
        for i, v in enumerate(numbers):
            expr = str(int(v)) if float(v).is_integer() else str(v)
            items.append({"id": i, "value": float(v), "expr": expr})

        init_state = {
            "original": original,
            "current": "",
            "method": method,
            "phase": 2,
            "items": items,
            "items_json": _compact_json(items),
            "next_id": len(items),
            "depth": 0,
        }

        # Graph of Operations (New graph per problem)
        gop = method_fn()
        # ---------
        # Controller
        # ---------
        ctrl = controller.Controller(
            lm,
            gop,
            Gameof24Prompter(),
            Gameof24Parser(),
            init_state,
        )

        # Run
        ctrl.run()

        # Output graph JSON (with index to prevent overwrite)
        filename = os.path.join(
            log_dir, f"output_{task}_{method}_{model_name}_{timestamp}_{idx}.json"
        )
        ctrl.output_graph(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Game24 episodic benchmark.")
    parser.add_argument(
        "--method",
        choices=sorted(METHODS.keys()),
        default="tot",
        help="Graph method to run.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o-mini",
        help="Model name from config.",
    )
    parser.add_argument(
        "--benchmark-path",
        default="/home/hyunwoo/benchmarks/gameof24BoT.jsonl",
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--task",
        default="game24",
        help="Task label for logging.",
    )
    args = parser.parse_args()

    run_benchmark(
        task=args.task,
        method=args.method,
        model_name=args.model_name,
        benchmark_path=args.benchmark_path,
    )
