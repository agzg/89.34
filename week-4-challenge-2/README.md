## User state modelling

I implement a very small user state modelling mechanism using a small PyTorch model with a raw problem encoder (first number, operand, second number) to a fixed-sized embedding tensor.

The state is a latent vector updated from the the dataset of proble, user answer, and correctness and its trained so that it improves the prediction of the user's next answer and its correctness.

### Task

- Singl digit arithmentic
- A sequence of steps where each step encodes the problem, the user's answer and its correctness.
- Probability of user giving a wrong answer is also encoded, otherwise the answer is marked as correct.

### Architecture

1. The prob;em is first encoded as (first number, operand, second number)
2. The user's state is updated by a state vector update
3.  Maintain an explicit state vector $h_t$:
$$
h_t = \text{GRU}\bigl(
    h_{t-1},\;
    [\text{embed}(\text{problem}_t),\; \text{answer}_t,\; \text{correct}_t]
\bigr).
$$where GRU represents a gated reccurent unit from RNN cells. The state is a function of the full history of problems and user responses.
4. Given current state $h_{t-1}$ and the **next** problem embedding, predicts:
   - Next user answer (classification over answer bins).
   - Next correctness (binary logit).

### Training objective

The objective is chosen so that **user state is explicitly useful**:

- **Next-answer loss**: Cross-entropy between predicted and actual next user answer. The state is the only summary of history, so it must encode whatever helps predict the next answer (e.g. recent performance, “slipiness”).
- **Next-correctness loss**: BCE between predicted and actual next correctness. This pushes the state to capture information that predicts whether the user will be correct on the next step (e.g. mastery, fatigue).

Total loss:
$$
\mathcal{L} = \mathcal{L}_{\text{answer}} + \mathcal{L}_{\text{correct}}.
$$

No auxiliary state regularizer is used; the state is trained purely to improve these two predictions.

### Usage

```bash
pip install -r requirements.txt
python train.py --sessions 2000 --seq_len 20 --epochs 30
```

Options: `--state_dim`, `--lr`, `--batch_size`, `--slip_prob`, etc.

### Ack
Cursor AI used for help with torch-specific code, and ChatGPT for help writing parts of this README.