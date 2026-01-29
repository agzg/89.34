"""
User-state model for arithmetic: explicitly maintains a latent user state
that is updated from (problem, response, correctness) and used to predict
next response. Training objective ties state to predictive accuracy.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProblemEncoder(nn.Module):
    """Encode (a, op, b) into a fixed-size vector."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        return self.mlp(x)


class UserStateUpdater(nn.Module):
    """
    Update user state from previous state and current observation:
    observation = (problem_embedding, user_answer_scaled, correct).
    State is explicitly the latent representation we train to be predictive.
    """

    def __init__(
        self,
        problem_dim: int,
        state_dim: int,
        obs_dim: int = 2,  # answer_scaled + correct
    ):
        super().__init__()
        self.state_dim = state_dim
        # Concatenate: problem_embed, answer_scaled, correct -> feed into GRU
        self.input_proj = nn.Linear(problem_dim + obs_dim, state_dim)
        self.gru = nn.GRUCell(state_dim, state_dim)

    def forward(
        self,
        prev_state: torch.Tensor,
        problem_embed: torch.Tensor,
        answer_scaled: torch.Tensor,
        correct: torch.Tensor,
    ) -> torch.Tensor:
        # prev_state: (B, state_dim), problem_embed: (B, problem_dim)
        # answer_scaled, correct: (B,) -> (B, 1) each
        obs = torch.stack([answer_scaled, correct], dim=-1)  # (B, 2)
        x = torch.cat([problem_embed, obs], dim=-1)  # (B, problem_dim+2)
        x = self.input_proj(x)
        new_state = self.gru(x, prev_state)
        return new_state


class StateConditionedPredictor(nn.Module):
    """Predict next user answer (and optionally correctness) from current state + next problem."""

    def __init__(
        self,
        state_dim: int,
        problem_dim: int,
        num_answer_bins: int = 41,  # -20..20 or similar; or regression
        predict_correct: bool = True,
    ):
        super().__init__()
        self.num_answer_bins = num_answer_bins
        self.predict_correct = predict_correct
        combined = state_dim + problem_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.answer_head = nn.Linear(state_dim, num_answer_bins)
        self.correct_head = nn.Linear(state_dim, 1) if predict_correct else None

    def forward(
        self,
        state: torch.Tensor,
        next_problem_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        combined = torch.cat([state, next_problem_embed], dim=-1)
        features = self.mlp(combined)
        logits = self.answer_head(features)
        correct_logit = self.correct_head(features).squeeze(-1) if self.correct_head is not None else None
        return logits, correct_logit


class ArithmeticUserStateModel(nn.Module):
    """
    Full model: explicit user state over a session.
    - problem_encoder: problem -> embedding
    - state_updater: (h_prev, problem_embed, answer, correct) -> h_new
    - predictor: (h, next_problem_embed) -> (answer_logits, correct_logit)

    Training: for each step t we have (problem_t, answer_t, correct_t).
    We update state from (h_{t-1}, problem_t, answer_t, correct_t) -> h_t.
    Then we predict (answer_{t+1}, correct_{t+1}) from (h_t, problem_{t+1})
    and train with cross-entropy / BCE. So the objective explicitly trains
    the state to be predictive of next user behavior.
    """

    def __init__(
        self,
        problem_dim: int = 32,
        state_dim: int = 64,
        num_answer_bins: int = 41,
        predict_correct: bool = True,
    ):
        super().__init__()
        self.problem_encoder = ProblemEncoder(
            input_dim=3,
            hidden_dim=32,
            out_dim=problem_dim,
        )
        self.state_updater = UserStateUpdater(
            problem_dim=problem_dim,
            state_dim=state_dim,
        )
        self.predictor = StateConditionedPredictor(
            state_dim=state_dim,
            problem_dim=problem_dim,
            num_answer_bins=num_answer_bins,
            predict_correct=predict_correct,
        )
        self.state_dim = state_dim
        self.num_answer_bins = num_answer_bins

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self,
        problems: torch.Tensor,
        user_answers_scaled: torch.Tensor,
        corrects: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        problems: (B, T, 3)
        user_answers_scaled: (B, T)
        corrects: (B, T)

        We compute states h_0..h_{T-1} from steps 0..T-1, then predictions
        for steps 1..T (next-answer and next-correct).
        """
        B, T, _ = problems.shape
        device = problems.device
        problem_embeds = self.problem_encoder(problems)  # (B, T, problem_dim)

        states = []
        h = self.init_state(B, device)
        for t in range(T):
            h = self.state_updater(
                h,
                problem_embeds[:, t],
                user_answers_scaled[:, t],
                corrects[:, t],
            )
            states.append(h)
        # states: list of (B, state_dim) -> stack (B, T, state_dim)
        states_stack = torch.stack(states, dim=1)

        # Predict next step: from h_{t-1} and problem_t predict answer_t, correct_t
        # So we use state at t-1 and problem at t. For t=0 we have no prior state;
        # use h_0 from first observation. So: predict step t from state after step t-1 and problem t.
        answer_logits_list = []
        correct_logits_list = []
        h_pred = self.init_state(B, device)
        for t in range(T):
            # After step t-1 we have h_pred. Predict step t from h_pred and problem_t
            logits, c_logit = self.predictor(h_pred, problem_embeds[:, t])
            answer_logits_list.append(logits)
            if c_logit is not None:
                correct_logits_list.append(c_logit)
            # Then update state with (problem_t, answer_t, correct_t)
            h_pred = self.state_updater(
                h_pred,
                problem_embeds[:, t],
                user_answers_scaled[:, t],
                corrects[:, t],
            )

        answer_logits = torch.stack(answer_logits_list, dim=1)  # (B, T, num_bins)
        correct_logits = torch.stack(correct_logits_list, dim=1) if correct_logits_list else None  # (B, T)
        if return_states:
            return answer_logits, correct_logits, states_stack
        return answer_logits, correct_logits, None
