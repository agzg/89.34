"""
Arithmetic task data: single-digit addition/subtraction.
Each sample is a sequence of (problem, user_answer, correct) for one user session.
"""

import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple

# Task definition: operands 0..9, ops +/-
MAX_OPERAND = 9
OPS = ["+", "-"]
OP_TO_ID = {"+": 0, "-": 1}
ID_TO_OP = {0: "+", 1: "-"}


def make_problem() -> Tuple[int, str, int]:
    """Sample one arithmetic problem (a, op, b)."""
    op = random.choice(OPS)
    a = random.randint(0, MAX_OPERAND)
    b = random.randint(0, MAX_OPERAND)
    if op == "-" and b > a:
        a, b = b, a
    return a, op, b


def solve(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    return a - b


def problem_to_tensor(a: int, op: str, b: int) -> torch.Tensor:
    """Encode problem as tensor [a, op_id, b] in 0..1 normalized."""
    return torch.tensor(
        [a / MAX_OPERAND, float(OP_TO_ID[op]), b / MAX_OPERAND],
        dtype=torch.float32,
    )


class ArithmeticSession:
    """One user session: list of (problem_tensor, user_answer, correct)."""

    def __init__(self, seq_len: int, slip_prob: float = 0.1):
        self.seq_len = seq_len
        self.slip_prob = slip_prob  # P(user gives wrong answer despite knowing)
        self.steps: List[Tuple[torch.Tensor, int, int]] = []

    def generate(self):
        """Generate a random session."""
        self.steps = []
        for _ in range(self.seq_len):
            a, op, b = make_problem()
            correct_ans = solve(a, op, b)
            # User "knows" answer but may slip; for variety we sometimes flip
            if random.random() < self.slip_prob:
                wrong = correct_ans + random.choice([-2, -1, 1, 2])
                wrong = max(-MAX_OPERAND, min(MAX_OPERAND * 2, wrong))
                user_ans = wrong
                correct = 0
            else:
                user_ans = correct_ans
                correct = 1
            pt = problem_to_tensor(a, op, b)
            self.steps.append((pt, user_ans, correct))
        return self


class ArithmeticStateDataset(Dataset):
    """
    Dataset of sessions for user-state modeling.
    Each item is a full session: (problems, user_answers, corrects) as tensors.
    """

    def __init__(
        self,
        num_sessions: int = 1000,
        seq_len: int = 20,
        slip_prob: float = 0.15,
        max_answer: int = 20,
    ):
        self.seq_len = seq_len
        self.max_answer = max_answer  # for one-hot or scalar scaling
        self.sessions: List[ArithmeticSession] = []
        for _ in range(num_sessions):
            s = ArithmeticSession(seq_len=seq_len, slip_prob=slip_prob)
            s.generate()
            self.sessions.append(s)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int):
        s = self.sessions[idx]
        problems = torch.stack([t[0] for t in s.steps])  # (seq_len, 3)
        user_answers = torch.tensor([t[1] for t in s.steps], dtype=torch.long)
        corrects = torch.tensor([t[2] for t in s.steps], dtype=torch.float32)
        # Normalize answer to 0..1 for regression or keep as class
        answer_scaled = (user_answers.float() + self.max_answer) / (2 * self.max_answer)
        answer_scaled = answer_scaled.clamp(0.0, 1.0)
        return {
            "problems": problems,
            "user_answers": user_answers,
            "answer_scaled": answer_scaled,
            "corrects": corrects,
        }
