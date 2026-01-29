"""
Training script: objective explicitly models user state.

Objective:
  1. Next-answer prediction: given state h_{t-1} (updated from history) and
     next problem, predict user's answer at step t. State is trained to
     encode information that improves this prediction.
  2. Next-correctness prediction: predict whether the user will be correct
     at step t from (h_{t-1}, problem_t). Encourages state to capture
     "mastery" or recent performance.

Loss = CE(answer_pred, answer_true) + BCE(correct_pred, correct_true).
Optional: state regularization (e.g., state predicts running accuracy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

from data import ArithmeticStateDataset, MAX_OPERAND
from model import ArithmeticUserStateModel

# Answer range: addition 0..18, subtraction 0..9. Map to bins -20..20 -> 41 bins
ANSWER_OFFSET = 20
NUM_ANSWER_BINS = 41


def answer_to_bin(answer: torch.Tensor) -> torch.Tensor:
    """Map raw answer (e.g. 0..18) to bin index in [0, NUM_ANSWER_BINS-1]."""
    return (answer + ANSWER_OFFSET).clamp(0, NUM_ANSWER_BINS - 1).long()


def train_step(
    model: ArithmeticUserStateModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    problems = batch["problems"].to(device)
    answer_scaled = batch["answer_scaled"].to(device)
    corrects = batch["corrects"].to(device)
    user_answers = batch["user_answers"].to(device)

    answer_logits, correct_logits, _ = model(
        problems,
        answer_scaled,
        corrects,
        return_states=False,
    )
    # Targets: current step's answer and correctness (we predict step t from state after t-1)
    answer_bins = answer_to_bin(user_answers)  # (B, T)
    loss_answer = F.cross_entropy(
        answer_logits.reshape(-1, NUM_ANSWER_BINS),
        answer_bins.reshape(-1),
    )
    loss_correct = F.binary_cross_entropy_with_logits(
        correct_logits.reshape(-1),
        corrects.reshape(-1),
    )
    loss = loss_answer + loss_correct

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Accuracy
    pred_bins = answer_logits.argmax(dim=-1)
    acc_answer = (pred_bins == answer_bins).float().mean().item()
    pred_correct = (correct_logits > 0).float()
    acc_correct = (pred_correct == corrects).float().mean().item()

    return {
        "loss": loss.item(),
        "loss_answer": loss_answer.item(),
        "loss_correct": loss_correct.item(),
        "acc_answer": acc_answer,
        "acc_correct": acc_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--slip_prob", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--state_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ArithmeticStateDataset(
        num_sessions=args.sessions,
        seq_len=args.seq_len,
        slip_prob=args.slip_prob,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ArithmeticUserStateModel(
        problem_dim=32,
        state_dim=args.state_dim,
        num_answer_bins=NUM_ANSWER_BINS,
        predict_correct=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_acc_a = 0.0
        total_acc_c = 0.0
        n_batches = 0
        for batch in loader:
            metrics = train_step(model, batch, optimizer, device)
            total_loss += metrics["loss"]
            total_acc_a += metrics["acc_answer"]
            total_acc_c += metrics["acc_correct"]
            n_batches += 1
        avg_loss = total_loss / n_batches
        avg_acc_a = total_acc_a / n_batches
        avg_acc_c = total_acc_c / n_batches
        print(
            f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} "
            f"acc_answer={avg_acc_a:.4f} acc_correct={avg_acc_c:.4f}"
        )

    print("Done. User state is trained to predict next answer and next correctness.")


if __name__ == "__main__":
    main()
