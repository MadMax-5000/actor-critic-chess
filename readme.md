# ♟️ NeuroChess Zero (NCZ)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**NeuroChess Zero** is a minimalist yet powerful deep reinforcement learning chess engine implementation inspired by **AlphaZero**. It features a custom ResNet-based Policy-Value Network that iteratively improves through a dual-phase training pipeline: Supervised Learning (SL) on grandmaster games followed by Reinforcement Learning (RL) via Monte Carlo Tree Search (MCTS) self-play.

This repository is optimized for local workstations (Linux/Windows/Mac) with CUDA GPU acceleration.

---

## 📂 Repository Structure

Ensure your directory is organized exactly as follows before running scripts:

```text
NeuroChess-Zero/
├── checkpoints/
│   ├── supervised/          # Phase 1 models saved here
│   └── rl/                  # Phase 2 generations saved here
├── data/
│   ├── raw/                 # Place your .pgn files here
│   └── processed/           # Replay buffers (pickle files)
├── src/
│   ├── train_sl.py          # Phase 1: Supervised Learning script
│   └── train_rl.py          # Phase 2: Reinforcement Learning script
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🧠 Architecture

The engine utilizes a shared backbone with two distinct heads:

1.  **Input Representation:** $8 \times 8 \times 19$ tensor (Piece placement, Castling rights, En Passant, Move clock).
2.  **Backbone:** 6-Block Residual Network (ResNet) with 128 filters per convolution.
3.  **Policy Head (Actor):** Outputs a probability distribution over 4,096 distinct moves.
4.  **Value Head (Critic):** Outputs a scalar evaluation $[-1, 1]$ estimating the win probability.

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MadMax-5000/actor-critic-chess.git
cd actor-critic-chess
```

### 2. Set up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
For Phase 1, you need a database of high-quality chess games.
1.  Download a generic PGN file (e.g., [Lichess Elite Database](https://database.lichess.org/)).
2.  Place the file in `data/raw/`.
3.  Rename the file to `grandmaster_games.pgn` (or update `PGN_FILENAME` in `src/train_sl.py`).

---

## 🛠️ Usage Guide

### Phase 1: Supervised Fine-Tuning (SFT)
Before self-play, we bootstrap the engine by teaching it to mimic human grandmasters. This prevents the "cold start" problem where random initialization leads to unstable RL training.

```bash
python src/train_sl.py
```
*   **Process:** Streams games from the PGN, minimizes CrossEntropy (Policy) and MSE (Value).
*   **Target:** Run until loss stabilizes (approx. 2.0 - 2.5).
*   **Output:** Checkpoints are saved to `checkpoints/supervised/`.

### Phase 2: Reinforcement Learning (RL)
The engine plays against itself, generates data, and learns from its own games.

**⚠️ Critical Step: The Handover**
You must manually select your best model from Phase 1 to be the "Generation 0" for Phase 2.

1.  Navigate to `checkpoints/supervised/`.
2.  Copy your best file (e.g., `resnet_step_50000.pth`).
3.  Paste it into `checkpoints/rl/`.
4.  **Rename it** to `gen_0.pth`.

**Start Self-Play:**
```bash
python src/train_rl.py
```
*   **Process:** 
    1.  Loads `gen_N.pth`.
    2.  Plays games using MCTS (200 simulations).
    3.  Adds Dirichlet noise for exploration.
    4.  Trains on the replay buffer.
    5.  Saves `gen_N+1.pth`.

---

## ⚙️ Configuration

Hyperparameters are defined at the top of the scripts for easy tuning.

**`src/train_sl.py`**
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `BATCH_SIZE` | 1024 | Reduce to 512 or 256 if getting OOM errors. |
| `LR` | 1e-4 | Learning rate with Cosine Annealing. |
| `BUFFER_SIZE` | 20,000 | Number of moves held in RAM for shuffling. |

**`src/train_rl.py`**
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `SIMULATIONS` | 200 | MCTS readouts per move. Higher = Stronger but slower. |
| `EPISODES` | 10 | Games played per iteration before retraining. |
| `C_PUCT` | 1.2 | Exploration constant (Higher = more wild moves). |

---

## 📊 Evaluation Arena

The RL script includes an automatic evaluation mode. Every `EVAL_EVERY_N` generations (Default: 5), the current model plays a match against the baseline (`gen_0`).

Sample Output:
```text
⚔️ EVALUATION ARENA ⚔️
  Game 1: White(Current) vs Black(Baseline) -> White
  Game 2: White(Baseline) vs Black(Current) -> Black
🏆 Match Result: +2 -0 =0
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Acknowledgements

*   **DeepMind** for the original AlphaZero paper.
*   **Python-Chess** for the robust board generation library.
*   **PyTorch** for the tensor computation framework.
