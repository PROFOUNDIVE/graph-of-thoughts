# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import os
import logging
import datetime
import json
import csv
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser

# This is a hack to also allow execution of this file from the examples directory
try:
    from . import utils
except ImportError:
    import utils


class Gameof24Prompter(prompter.Prompter):
    """
    Gameof24Prompter provides the generation of prompts specific to the gameof24
    example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    gameof24_prompt = """<Instruction>
You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Create one valid expression that equals 24.

Output must follow the example format as closely as possible: a single line of the form
<expression> = 24
Do not include any additional explanation or text.
</Instruction>

<Example>
Input: 4 9 10 13
Output: (10 - 4) * (13 - 9) = 24
</Example>

Input: {input}
Output:"""

    gameof24_prompt_cot = """<Instruction>
You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Find one valid expression that equals 24.

You may show intermediate steps, but the final line must start with "Output: " and follow the example format:
Output: <expression> = 24
Do not include any additional text after the final output line.
</Instruction>

<Approach>
1. Consider combining two numbers at a time using +, -, *, / to create intermediate results.
2. Reuse intermediate results with the remaining numbers, ensuring each original number is used exactly once.
3. Try common 24 patterns such as:
   - (a - b) * (c - d)
   - (a + b) * (c - d)
   - (a * b) / (c - d)
   - (a / b) + (c / d), then scale if needed
4. If a path fails, backtrack and try a different pairing or operation.
</Approach>

<Examples>
Input: 4 9 10 13
Work:
(10 - 4) = 6
(13 - 9) = 4
6 * 4 = 24
Output: (10 - 4) * (13 - 9) = 24

Input: 1 3 4 6
Work:
(6 / 3) = 2
4 * 2 = 8
8 * 3 = 24  (using 1 and 3? not allowed) -> backtrack
(4 - 1) = 3
6 * 3 = 18
18 + 3 = 21 -> backtrack
(6 - 1) = 5
5 * 4 = 20
20 + 3 = 23 -> backtrack
(6 - 4) = 2
3 * 2 = 6
6 * 4 = 24 (uses 4 twice) -> backtrack
(6 / (1 - 3/4)) = 24
Output: 6 / (1 - 3/4) = 24
</Examples>

Input: {input}
"""

    gameof24_next_move_prompt_jsonl = """<Instruction>
You are proposing next moves for the 24 Game search.

State consists of items with unique ids. A move selects exactly two different item ids and an operator in {+,-,*,/}.
You must output exactly {num_branches} candidate moves.
Each candidate must be on its own line and must be a single JSON object with this schema:

{"pick":[id1,id2], "op":"+"}

Rules:
- id1 and id2 must be two different ids present in the current items.
- op must be one of: "+", "-", "*", "/".
- Avoid duplicate moves: do not output the same (pick, op) twice.
- For division, do NOT choose a divisor that is 0 (based on the current value of that item).
- Prefer diverse candidates (different pairs/operators), not minor variations.
- Do not include any other text, explanations, code fences, or commentary.
</Instruction>

<CurrentItems>
{items_json}
</CurrentItems>

Output:
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        :raise AssertionError: If not exactly two thought states are provided.
        """
        raise NotImplementedError("Game of 24 does not use aggregation in this baseline.")

    def generate_prompt(
        self, num_branches: int, original: str, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input list of numbers.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """

        state = kwargs.get("state") or kwargs
        items_json = state["item_json"]

        if current is None or current == "":
            input = original
        else:
            input = current
        if method.startswith("io"):
            return self.gameof24_prompt.format(input=input)
        elif method.startswith("cot"):
            return self.gameof24_prompt_cot.format(input=input)
        elif method.startswith("got"):
            return self.gameof24_next_move_prompt_jsonl.format(
                num_branches=num_branches,
                items_json=items_json
            )

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        pass


class Gameof24Parser(parser.Parser):
    """
    Gameof24Parser provides the parsing of language model reponses specific to
    the sorting example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        :raise AssertionError: If not exactly two thought states are provided.
        """

        assert len(states) == 2, "Expected two states for aggregation answer."
        new_states = []
        for text in texts:
            answers = text.strip().split("\n")
            if any(["Output" in answer for answer in answers]):
                # cut elements until last output is found
                for answer in reversed(answers):
                    if "Output" in answer:
                        answers = answers[answers.index(answer) :]
                        break

            answers_stripped = [
                answer for answer in answers if "[" in answer and "]" in answer
            ]
            if len(answers_stripped) == 0:
                for answer in answers:
                    answer = "[" + answer + "]"
                    try:
                        answer_converted = utils.string_to_list(answer)
                        if len(answer_converted) > 0:
                            answers_stripped.append(answer)
                    except:
                        pass
            if len(answers_stripped) == 0:
                logging.warning(
                    f"Could not parse aggregation answer: {text}. Returning empty list."
                )
                answer = "[]"
            else:
                answer = [
                    answer[answer.index("[") : answer.index("]") + 1]
                    for answer in answers_stripped
                ][0]
            states = sorted(states, key=lambda x: x["part"])
            merged_unsorted_sublists = (
                states[0]["unsorted_sublist"][:-1]
                + ", "
                + states[1]["unsorted_sublist"][1:]
            )
            new_state = states[0].copy()
            new_state["current"] = answer
            new_state["unsorted_sublist"] = merged_unsorted_sublists
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        Expects JSONL lines of the form:
            {"pick":[id1,id2], "op":"+"}

        The parser applies the move to the current state (items) and returns next states.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []

        def _parse_numbers_from_original(original: str) -> List[float]:
            if original is None:
                return []
            s = str(original).strip()
            # Try JSON list first
            try:
                if s.startswith("[") and s.endswith("]"):
                    arr = json.loads(s)
                    return [float(x) for x in arr]
            except Exception:
                pass
            # Fallback: whitespace/comma separated (e.g., "4 9 10 13" or "4,9,10,13")
            s = s.replace(",", " ")
            s = s.replace("[", " ").replace("]", " ")
            parts = [p for p in s.split() if p.strip() != ""]
            out = []
            for p in parts:
                try:
                    out.append(float(p))
                except Exception:
                    pass
            return out

        def _ensure_items_schema(base_state: Dict) -> Dict:
            """
            Ensure state has:
            - items: List[{"id","value","expr"}]
            - next_id: int
            - depth: int
            - item_json: str (json dump for prompter)
            """
            if "items" in base_state and isinstance(base_state["items"], list) and len(base_state["items"]) > 0:
                # Ensure item_json exists
                if "item_json" not in base_state:
                    try:
                        base_state["item_json"] = json.dumps(base_state["items"])
                    except Exception:
                        base_state["item_json"] = str(base_state["items"])
                if "next_id" not in base_state:
                    try:
                        base_state["next_id"] = max([int(it["id"]) for it in base_state["items"]]) + 1
                    except Exception:
                        base_state["next_id"] = len(base_state["items"])
                if "depth" not in base_state:
                    base_state["depth"] = 0
                return base_state

            nums = _parse_numbers_from_original(base_state.get("original", ""))
            items = [{"id": i, "value": float(nums[i]), "expr": str(int(nums[i])) if float(nums[i]).is_integer() else str(nums[i])}
                    for i in range(len(nums))]
            base_state["items"] = items
            base_state["next_id"] = len(items)
            base_state["depth"] = base_state.get("depth", 0) or 0
            try:
                base_state["item_json"] = json.dumps(items)
            except Exception:
                base_state["item_json"] = str(items)
            return base_state

        def _find_item(items: List[Dict], item_id: int) -> Union[Dict, None]:
            for it in items:
                if int(it.get("id", -1)) == int(item_id):
                    return it
            return None

        def _apply_op(a: float, b: float, op: str) -> Union[float, None]:
            try:
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return a * b
                if op == "/":
                    if abs(b) < 1e-12:
                        return None
                    return a / b
            except Exception:
                return None
            return None

        base_state = _ensure_items_schema(state.copy())

        for text in texts:
            # Each 'text' may contain multiple JSON lines (num_branches candidates)
            lines = text.strip().split("\n")
            parsed_any = False

            for line in lines:
                line = line.strip()
                if "{" not in line or "}" not in line:
                    continue

                # Cut everything outside the outermost JSON object on the line
                try:
                    line_json = line[line.index("{") : line.rindex("}") + 1]
                except Exception:
                    continue

                try:
                    move = json.loads(line_json)
                except Exception as e:
                    logging.warning(f"Could not parse move json: {line}. Encountered exception: {e}")
                    continue

                if not isinstance(move, dict):
                    continue
                if "pick" not in move or "op" not in move:
                    continue
                if not isinstance(move["pick"], list) or len(move["pick"]) != 2:
                    continue
                if move["op"] not in ["+", "-", "*", "/"]:
                    continue

                id1, id2 = move["pick"][0], move["pick"][1]
                if id1 == id2:
                    continue

                items = base_state.get("items", [])
                it1 = _find_item(items, id1)
                it2 = _find_item(items, id2)
                if it1 is None or it2 is None:
                    continue

                a = float(it1.get("value", 0.0))
                b = float(it2.get("value", 0.0))
                op = move["op"]

                res = _apply_op(a, b, op)
                if res is None:
                    # invalid move (e.g. division by zero) -> soft skip
                    continue

                expr1 = str(it1.get("expr", it1.get("value", "")))
                expr2 = str(it2.get("expr", it2.get("value", "")))
                new_expr = f"({expr1}{op}{expr2})"

                # Build next items: remove picked ones, add new one
                next_items = []
                for it in items:
                    if int(it.get("id", -1)) in [int(id1), int(id2)]:
                        continue
                    next_items.append(it)

                new_id = int(base_state.get("next_id", len(items)))
                next_items.append({"id": new_id, "value": float(res), "expr": new_expr})

                new_state = base_state.copy()
                new_state["items"] = next_items
                new_state["next_id"] = new_id + 1
                new_state["depth"] = int(base_state.get("depth", 0)) + 1
                new_state["last_move"] = {"pick": [int(id1), int(id2)], "op": op}
                new_state["current"] = new_expr  # keep a human-readable trace

                try:
                    new_state["item_json"] = json.dumps(next_items)
                except Exception:
                    new_state["item_json"] = str(next_items)

                new_states.append(new_state)
                parsed_any = True

            if not parsed_any:
                logging.warning(
                    f"Could not parse any valid move from generate answer: {text}. Returning a penalized state."
                )
                # Soft-failure state: progresses depth to avoid loops; score function should penalize.
                fallback_state = base_state.copy()
                fallback_state["depth"] = int(base_state.get("depth", 0)) + 1
                fallback_state["invalid_move"] = True
                fallback_state["current"] = "INVALID_MOVE"
                new_states.append(fallback_state)

        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 20))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(1):
        operations_graph.append_operation(operations.Generate(1, 20))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def tot2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_1 = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, 10))
        operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
        keep_best_2 = operations.KeepBestN(1, False)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    plans = operations.Generate(1, 1)
    operations_graph.append_operation(plans)  # generate the sublists
    for i in range(1, 3):
        list_id = f"List {i}"
        sub_list = operations.Selector(
            lambda thoughts, list_id=list_id: [
                thought for thought in thoughts if thought.state["part"] == list_id
            ]
        )
        sub_list.add_predecessor(plans)
        operations_graph.add_operation(sub_list)
        sort_sub_list = operations.Generate(1, 5)
        sort_sub_list.add_predecessor(sub_list)
        operations_graph.add_operation(sort_sub_list)
        score_sub_list = operations.Score(1, False, utils.num_errors)
        score_sub_list.add_predecessor(sort_sub_list)
        operations_graph.add_operation(score_sub_list)
        keep_best_sub_list = operations.KeepBestN(1, False)
        keep_best_sub_list.add_predecessor(score_sub_list)
        operations_graph.add_operation(keep_best_sub_list)

    final_aggregate = operations.Aggregate(10)
    operations_graph.append_operation(final_aggregate)
    operations_graph.append_operation(operations.Score(1, False, utils.num_errors))
    keep_best_aggregate_final = operations.KeepBestN(1, False)
    operations_graph.append_operation(keep_best_aggregate_final)

    operations_graph.append_operation(operations.Generate(1, 10))
    score_aggr_3 = operations.Score(1, False, utils.num_errors)
    score_aggr_3.add_predecessor(keep_best_aggregate_final)
    operations_graph.append_operation(score_aggr_3)
    operations_graph.append_operation(operations.KeepBestN(1, False))

    operations_graph.append_operation(operations.GroundTruth(utils.test_sorting))

    return operations_graph


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "sorting_032.csv")
    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2]])

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            lm = language_models.ChatGPT(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/config.json",
                ),
                model_name=lm_name,
                cache=True,
            )
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                Gameof24Prompter(),
                Gameof24Parser(),
                {
                    "original": data[1],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input (x)   : an unordered list of 32 numbers between 0 and 9 (inclusive)
    Output (y)  : a sorted list of 32 numbers between 0 and 9 (inclusive)
    Correct     : y == sorted(x)
    Input Example:
        [0, 1, 9, 4, 2, 2, 0, 5, 1...]
    Output Example:
        [0, 0, 0, 0, 1, 1, 1, 1, 2...]
    """
    budget = 30
    samples = [item for item in range(0, 100)]
    approaches = [io, cot, tot, tot2, got]

    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")
