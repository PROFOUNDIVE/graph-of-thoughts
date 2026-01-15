import argparse
import concurrent.futures
import json
import logging
import os

from datetime import datetime
from typing import Optional
from tqdm import tqdm

from graph_of_thoughts import controller, language_models
from tasks.gameof24 import (
    Gameof24Prompter,
    get_gameof24_parser,
    got,
    tot,
    tot2,
    io,
    cot,
    cot_sc,
)
from tasks import utils  # (선택) 디버깅/스코어 확인용


def _compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

METHODS = {
    "io": io,
    "cot": cot,
    "cot_sc": cot_sc,
    "tot": tot,
    "tot2": tot2,
    "got": got,
}

LOGGER = logging.getLogger(__name__)


def _log_thoughts(ctrl: controller.Controller) -> None:
    for operation in ctrl.graph.operations:
        if not operation.get_thoughts():
            continue
        LOGGER.debug("Operation %s thoughts:", operation.operation_type)
        for idx, thought in enumerate(operation.get_thoughts()):
            LOGGER.debug(
                "Thought %s state: %s",
                idx,
                json.dumps(thought.state, ensure_ascii=False, sort_keys=True),
            )


def _solve_problem(
    idx: int,
    line: str,
    method: str,
    method_fn,
    model_name: str,
    log_dir: str,
    task: str,
    timestamp: str,
    lm: Optional[language_models.ChatGPT] = None,
) -> None:
    problem = json.loads(line)
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

    local_lm = lm or language_models.ChatGPT(
        "../config.json",
        model_name=model_name,
        cache=False,
    )
    local_lm.reset_usage()

    gop = method_fn()
    ctrl = controller.Controller(
        local_lm,
        gop,
        Gameof24Prompter(),
        get_gameof24_parser(method),
        init_state,
    )
    ctrl.run()
    if LOGGER.isEnabledFor(logging.DEBUG):
        _log_thoughts(ctrl)

    filename = os.path.join(
        log_dir, f"output_{task}_{method}_{model_name}_{timestamp}_{idx}.json"
    )
    ctrl.output_graph(filename)


def run_benchmark(
    task: str,
    method: str,
    model_name: str,
    benchmark_path: str,
    multithread: bool,
    max_workers: int,
) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = f"../logs/output_{task}_{method}_{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    with open(benchmark_path, "r") as f:
        lines = f.readlines()

    method_fn = METHODS[method]

    if multithread:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _solve_problem,
                    idx,
                    line,
                    method,
                    method_fn,
                    model_name,
                    log_dir,
                    task,
                    timestamp,
                )
                for idx, line in enumerate(lines)
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Solving Game24",
            ):
                pass
        return

    lm = language_models.ChatGPT(
        "../config.json",
        model_name=model_name,
        cache=False,
    )
    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Solving Game24"):
        _solve_problem(
            idx,
            line,
            method,
            method_fn,
            model_name,
            log_dir,
            task,
            timestamp,
            lm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Game24 episodic benchmark.")
    parser.add_argument(
        "--method",
        choices=sorted(METHODS.keys()),
        default="tot",
        help="Graph method to run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for debug tracing.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path for debug output.",
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
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Enable multithreaded execution.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker threads when multithreaded.",
    )
    args = parser.parse_args()

    log_kwargs = {
        "level": getattr(logging, args.log_level),
        "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    }
    if args.log_file:
        log_kwargs["filename"] = args.log_file
        log_kwargs["filemode"] = "w"
    logging.basicConfig(**log_kwargs)

    run_benchmark(
        task=args.task,
        method=args.method,
        model_name=args.model_name,
        benchmark_path=args.benchmark_path,
        multithread=args.multithread,
        max_workers=args.max_workers,
    )
