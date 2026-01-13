# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import ast
import csv
import datetime
import json
import logging
import os
import re
from collections import Counter
from typing import Callable, Dict, List, Union

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
  You are an expert player of the 24 Game.
  Goal: Combine numbers using +, -, *, / to reach 24.
  Task: Generate exactly {num_branches} diverse next moves from the given state.
  
  State: A list of items, each with an 'id' and 'value'.
  Move: Select two different ids and an operator.
  
  Rules for high-quality generation:
  1. Diversity is key. Use a mix of all operators (+, -, *, /). Do not rely only on + and *.
  2. Commutativity:
     - + and * are commutative. picking [a, b] is the same as [b, a].
     - - and / are NOT commutative. Consider BOTH [a, b] and [b, a] when valid.
     - Example: 10 - 4 = 6, but 4 - 10 = -6. Both might be useful.
  3. Strategic thinking:
     - Prioritize operations that create factors of 24 (3, 4, 6, 8, 12).
     - Prioritize operations that create numbers close to 24.
     - Do not ignore fractions if they can lead to 24 (e.g., 6 / (1/4) = 24).
  4. Valid JSONL: output ONLY valid JSON objects, one per line.
  
  Schema:
  {{"pick": [id1, id2], "op": "operator"}}
  </Instruction>
  
  <Example>
  Current Items:
  [{{"id": 0, "value": 10.0, "expr": "10"}}, {{"id": 1, "value": 4.0, "expr": "4"}}, {{"id": 2, "value": 2.0, "expr": "2"}}]
  
  Output:
  {{"pick": [0, 1], "op": "-"}}
  {{"pick": [1, 0], "op": "-"}}
  {{"pick": [0, 2], "op": "/"}}
  {{"pick": [0, 1], "op": "+"}}
  {{"pick": [1, 2], "op": "*"}}
  {{"pick": [0, 2], "op": "-"}}
  ... (up to {num_branches} lines)
  </Example>
  
  <CurrentItems>
  {items_json}
  </CurrentItems>
  
  Output:
  """

    gameof24_improve_prompt_jsonl = """<Instruction>
  You are correcting a failed move in the 24 Game search.
  The last move led to a dead end, so propose a DIFFERENT move.
  
  Previous items:
  {prev_items_json}
  
  Last move (do NOT repeat this exact pick/op):
  {last_move_json}
  
  Output exactly one candidate move as a single JSON object line:
  {{"pick": [id1, id2], "op": "operator"}}
  
  Requirements:
  - pick two different ids from the previous items.
  - op must be one of: "+", "-", "*", "/".
  - avoid repeating the same pick/op as the last move.
  - for division, do NOT choose a divisor that is 0.
  </Instruction>
  
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
        return ""

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        """
        state = kwargs.get("state") or kwargs
        original = state.get("original", "")
        current = state.get("current", "")
        method = state.get("method", "")
        items_json = state.get("items_json", "[]")

        input_value = original if current in [None, ""] else current

        if method.startswith("io"):
            return self.gameof24_prompt.format(input=input_value)
        if method.startswith("cot"):
            return self.gameof24_prompt_cot.format(input=input_value)
        if method.startswith("tot") or method.startswith("got"):
            return self.gameof24_next_move_prompt_jsonl.format(
                num_branches=num_branches,
                items_json=items_json
            )
        return ""

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        state = kwargs.get("state") or kwargs
        prev_items_json = state.get("prev_items_json", state.get("items_json", "[]"))
        last_move_json = json.dumps(state.get("last_move", None))
        
        return self.gameof24_improve_prompt_jsonl.format(
            prev_items_json=prev_items_json,
            last_move_json=last_move_json,
        )

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        return ""

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
        return ""


class Gameof24ExpressionParser(parser.Parser):
    """
    Gameof24ExpressionParser provides parsing for IO/CoT responses that return
    a full expression solving the problem.
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
        :return: The new thought states after parsing the responses from the language model.
        :rtype: Union[Dict, List[Dict]]
        :raise AssertionError: If not exactly two thought states are provided.
        """
        return states[0] if states else {}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        Expects a full expression output such as:
            Output: (10 - 4) * (13 - 9) = 24

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the responses from the language model.
        :rtype: List[Dict]
        """
        base_state = state.copy()
        original_numbers = self._parse_numbers_from_original(base_state.get("original", ""))
        original_counter = Counter(self._normalize_number(num) for num in original_numbers)
        new_states = []

        for text in texts:
            expression = self._extract_expression(text)
            if expression is None:
                new_states.append(self._invalid_state(base_state, text))
                continue

            if not self._numbers_match(expression, original_counter):
                new_states.append(self._invalid_state(base_state, text))
                continue

            value = self._safe_eval(expression)
            if value is None:
                new_states.append(self._invalid_state(base_state, text))
                continue

            new_state = base_state.copy()
            new_state.pop("invalid_move", None)
            new_state["current"] = expression
            new_state["items"] = [
                {"id": 0, "value": float(value), "expr": expression}
            ]
            try:
                new_state["items_json"] = json.dumps(new_state["items"])
            except Exception:
                new_state["items_json"] = str(new_state["items"])
            new_state["next_id"] = 1
            new_state["depth"] = int(base_state.get("depth", 0)) + 1
            new_states.append(new_state)

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
        improved_states = self.parse_generate_answer(state, texts)
        return improved_states[0] if improved_states else {"invalid_move": True}

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
        if len(texts) == 0:
            return False
        text = texts[0].strip().lower()
        return any(token in text for token in ["true", "valid", "yes"])

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
        return [0.0 for _ in states]

    def _parse_numbers_from_original(self, original: str) -> List[float]:
        """
        Parse the original input string into a list of numbers.

        :param original: The original input string.
        :type original: str
        :return: Parsed list of numbers.
        :rtype: List[float]
        """
        if original is None:
            return []
        s = str(original).strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                arr = json.loads(s)
                return [float(x) for x in arr]
        except Exception:
            pass
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

    def _extract_expression(self, text: str) -> Union[str, None]:
        """
        Extract a candidate expression from the model output.

        :param text: Raw model response text.
        :type text: str
        :return: Extracted expression or None.
        :rtype: Union[str, None]
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            lowered = line.lower()
            if lowered.startswith("output"):
                if ":" in line:
                    expr_line = line.split(":", 1)[1].strip()
                else:
                    expr_line = line[len("output") :].strip()
                if expr_line == "" and idx + 1 < len(lines):
                    expr_line = lines[idx + 1].strip()
                return self._strip_expression(expr_line)
        for line in reversed(lines):
            if "=" in line:
                return self._strip_expression(line)
        return None

    def _strip_expression(self, line: str) -> str:
        """
        Strip the expression from a line containing an equals sign.

        :param line: Line containing the expression.
        :type line: str
        :return: Expression portion of the line.
        :rtype: str
        """
        expr_line = line.strip().rstrip(".")
        if "=" in expr_line:
            left, right = (part.strip() for part in expr_line.split("=", 1))
            if self._looks_like_expression(right) and not self._looks_like_expression(left):
                expr_line = right
            else:
                expr_line = left
        return expr_line

    def _looks_like_expression(self, text: str) -> bool:
        """
        Determine whether a string looks like a math expression.

        :param text: Candidate text.
        :type text: str
        :return: True if it resembles an expression, False otherwise.
        :rtype: bool
        """
        if re.search(r"[+\-*/()]", text):
            return True
        return False

    def _extract_numbers(self, expression: str) -> List[float]:
        """
        Extract numeric literals from an expression string.

        :param expression: Expression string.
        :type expression: str
        :return: Numbers found in the expression.
        :rtype: List[float]
        """
        tokens = re.findall(r"\d+(?:\.\d+)?", expression)
        return [float(token) for token in tokens]

    def _normalize_number(self, value: float) -> str:
        """
        Normalize a numeric value for comparison.

        :param value: Numeric value.
        :type value: float
        :return: Normalized string representation.
        :rtype: str
        """
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    def _numbers_match(self, expression: str, original_counter: Counter) -> bool:
        """
        Check whether the expression uses exactly the original numbers.

        :param expression: Expression string.
        :type expression: str
        :param original_counter: Counter of original numbers.
        :type original_counter: Counter
        :return: True if numbers match, False otherwise.
        :rtype: bool
        """
        expr_numbers = self._extract_numbers(expression)
        expr_counter = Counter(self._normalize_number(num) for num in expr_numbers)
        return expr_counter == original_counter

    def _safe_eval(self, expression: str) -> Union[float, None]:
        """
        Safely evaluate a math expression containing +, -, *, / only.

        :param expression: Expression string.
        :type expression: str
        :return: Evaluated numeric result or None on failure.
        :rtype: Union[float, None]
        """
        try:
            node = ast.parse(expression, mode="eval")
        except Exception:
            return None

        def _eval_node(ast_node: ast.AST) -> float:
            if isinstance(ast_node, ast.Expression):
                return _eval_node(ast_node.body)
            if isinstance(ast_node, ast.Constant) and isinstance(ast_node.value, (int, float)):
                return float(ast_node.value)
            if isinstance(ast_node, ast.UnaryOp) and isinstance(ast_node.op, (ast.UAdd, ast.USub)):
                value = _eval_node(ast_node.operand)
                return value if isinstance(ast_node.op, ast.UAdd) else -value
            if isinstance(ast_node, ast.BinOp) and isinstance(
                ast_node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                left = _eval_node(ast_node.left)
                right = _eval_node(ast_node.right)
                if isinstance(ast_node.op, ast.Add):
                    return left + right
                if isinstance(ast_node.op, ast.Sub):
                    return left - right
                if isinstance(ast_node.op, ast.Mult):
                    return left * right
                if isinstance(ast_node.op, ast.Div):
                    return left / right
            raise ValueError("Unsupported expression")

        try:
            return float(_eval_node(node))
        except Exception:
            return None

    def _invalid_state(self, base_state: Dict, text: str) -> Dict:
        """
        Build a fallback state for invalid parse results.

        :param base_state: Base thought state.
        :type base_state: Dict
        :param text: Raw model response text.
        :type text: str
        :return: Penalized state marked as invalid.
        :rtype: Dict
        """
        logging.warning(
            "Could not parse any valid expression from generate answer: "
            f"{text}. Returning a penalized state."
        )
        fallback_state = base_state.copy()
        fallback_state["depth"] = int(base_state.get("depth", 0)) + 1
        fallback_state["invalid_move"] = True
        fallback_state["current"] = "INVALID_MOVE"
        return fallback_state


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
        return states[0] if states else {}

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
            - items_json: str (json dump for prompter)
            """
            if "items" in base_state and isinstance(base_state["items"], list) and len(base_state["items"]) > 0:
                # Ensure items_json exists
                if "items_json" not in base_state:
                    try:
                        base_state["items_json"] = json.dumps(base_state["items"])
                    except Exception:
                        base_state["items_json"] = str(base_state["items"])
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
                base_state["items_json"] = json.dumps(items)
            except Exception:
                base_state["items_json"] = str(items)
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
                new_state["prev_items"] = items
                new_state["prev_items_json"] = base_state.get("items_json", "[]")
                new_state["prev_next_id"] = base_state.get("next_id", len(items))

                try:
                    new_state["items_json"] = json.dumps(next_items)
                except Exception:
                    new_state["items_json"] = str(next_items)

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
        prev_items = state.get("prev_items") or state.get("items", [])
        prev_items_json = state.get("prev_items_json") or state.get("items_json", "[]")
        prev_next_id = state.get("prev_next_id", state.get("next_id", len(prev_items)))
        base_depth = max(0, int(state.get("depth", 0)) - 1)

        base_state = state.copy()
        base_state["items"] = prev_items
        base_state["items_json"] = prev_items_json
        base_state["next_id"] = prev_next_id
        base_state["depth"] = base_depth

        improved_states = self.parse_generate_answer(base_state, texts)
        last_move = state.get("last_move")
        for improved_state in improved_states:
            if improved_state.get("invalid_move"):
                continue
            if last_move is not None and improved_state.get("last_move") == last_move:
                improved_state["invalid_move"] = True
                continue
            return improved_state
        return improved_states[0] if improved_states else {"invalid_move": True}

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
        if len(texts) == 0:
            return False
        text = texts[0].strip().lower()
        return any(token in text for token in ["true", "valid", "yes"])

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
        return [0.0 for _ in states]


def get_gameof24_parser(method: str) -> parser.Parser:
    """
    Select the parser implementation for a given Game24 method.

    :param method: Method name (e.g., io, cot, tot, got).
    :type method: str
    :return: Parser suited to the method.
    :rtype: parser.Parser
    """
    if method.startswith("io") or method.startswith("cot"):
        return Gameof24ExpressionParser()
    return Gameof24Parser()


def _is_solvable_state(state: Dict) -> bool:

    """
    Determine whether a state can still reach 24 using the oracle scorer.

    :param state: Thought state containing current items.
    :type state: Dict
    :return: True if residual can reach 0, False otherwise.
    :rtype: bool
    """
    items = state.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        return False
    try:
        vals = [float(item.get("value", 0.0)) for item in items]
        residual = utils._best_residual(utils._round_key(vals))
        return residual <= 1e-6
    except Exception:
        return False


def _beam_search_graph(
    num_branches: int,
    beam_width: int,
    max_depth: int = 3,
) -> operations.GraphOfOperations:
    """
    Game24 baseline search graph:
    repeat max_depth times: Generate(B) -> Score(game24_score) -> KeepBestN(K)
    then GroundTruth(test_game24)
    """
    operations_graph = operations.GraphOfOperations()

    prev_keep = None
    for _ in range(max_depth):
        operations_graph.append_operation(operations.Generate(1, num_branches))
        operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
        keep = operations.KeepBestN(beam_width, False)
        if prev_keep is not None:
            keep.add_predecessor(prev_keep)
        operations_graph.append_operation(keep)
        prev_keep = keep

    operations_graph.append_operation(operations.GroundTruth(utils.test_game24))

    return operations_graph


def _refined_beam_search_graph(
    num_branches: int,
    beam_width: int,
    max_depth: int = 3,
    num_tries: int = 2,
) -> operations.GraphOfOperations:
    """
    Game24 refined search graph:
    repeat max_depth times: Generate(B) -> ValidateAndImprove -> Score(game24_score) -> KeepBestN(K)
    then GroundTruth(test_game24)
    """
    operations_graph = operations.GraphOfOperations()

    prev_keep = None
    for _ in range(max_depth):
        operations_graph.append_operation(operations.Generate(1, num_branches))
        operations_graph.append_operation(
            operations.ValidateAndImprove(
                num_samples=1,
                improve=True,
                num_tries=num_tries,
                validate_function=_is_solvable_state,
            )
        )
        operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
        keep = operations.KeepBestN(beam_width, False)
        if prev_keep is not None:
            keep.add_predecessor(prev_keep)
        operations_graph.append_operation(keep)
        prev_keep = keep

    operations_graph.append_operation(operations.GroundTruth(utils.test_game24))

    return operations_graph


def _selective_refined_beam_search_graph(
    num_branches: int,
    refine_width: int,
    beam_width: int,
    max_depth: int = 3,
    num_tries: int = 1,
) -> operations.GraphOfOperations:
    """
    Game24 selective refined search graph:
    repeat max_depth times: Generate(B) -> Score -> KeepBestN(refine_width)
    -> ValidateAndImprove -> Score -> KeepBestN(beam_width) then GroundTruth.
    """
    operations_graph = operations.GraphOfOperations()

    prev_keep = None
    for _ in range(max_depth):
        operations_graph.append_operation(operations.Generate(1, num_branches))
        operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
        keep_for_refine = operations.KeepBestN(refine_width, False)
        if prev_keep is not None:
            keep_for_refine.add_predecessor(prev_keep)
        operations_graph.append_operation(keep_for_refine)

        operations_graph.append_operation(
            operations.ValidateAndImprove(
                num_samples=1,
                improve=True,
                num_tries=num_tries,
                validate_function=_is_solvable_state,
            )
        )
        operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
        keep_final = operations.KeepBestN(beam_width, False)
        operations_graph.append_operation(keep_final)
        prev_keep = keep_final

    operations_graph.append_operation(operations.GroundTruth(utils.test_game24))

    return operations_graph


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
    operations_graph.append_operation(operations.GroundTruth(utils.test_game24))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.game24_score))
    operations_graph.append_operation(operations.GroundTruth(utils.test_game24))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """

    # depth=3 (4->3->2->1), B=30, K=3
    return _beam_search_graph(num_branches=30, beam_width=3, max_depth=3)


def tot2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT2 method.
    ToT2 uses a tree with more levels, but with fewer branches per level.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """

    # depth=3 (4 -> 3 -> 2 -> 1), B=30, K=1
    return _beam_search_graph(num_branches=30, beam_width=1, max_depth=3)


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.
    
    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    # GoT baseline uses selective refining before the final beam.
    # depth=3 (4 -> 3 -> 2 -> 1), B=30, refine=10, K=3
    return _selective_refined_beam_search_graph(
        num_branches=30,
        refine_width=10,
        beam_width=3,
        max_depth=3,
        num_tries=1,
    )


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

    def _parse_numbers_for_items(original: str) -> List[float]:
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
        # Fallback: whitespace/comma separated
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

    def _init_items_state(original: str) -> Dict:
        nums = _parse_numbers_for_items(original)
        items = []
        for i, v in enumerate(nums):
            fv = float(v)
            expr = str(int(fv)) if fv.is_integer() else str(fv)
            items.append({"id": i, "value": fv, "expr": expr})
        return {
            "items": items,
            "items_json": json.dumps(items),
            "next_id": len(items),
            "depth": 0,
        }

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
                get_gameof24_parser(method.__name__),
                {
                    "original": data[1],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                    **_init_items_state(data[1]),
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
