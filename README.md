# 🇹🇭 Thai-Checkers-AlphaZero (ระบบ AI หมากฮอสไทยด้วย AlphaZero)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14_Metal-EE4C2C.svg)](https://pytorch.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS_Accelerated-000000.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-16%2F16_Passing-success.svg)](tests/)

> An S-Tier Reinforcement Learning engine that trains an undefeated **Thai Checkers (หมากฮอสไทย)** AI from scratch (**Tabula Rasa**) using the **AlphaZero** architecture (Deep Residual Networks + Monte Carlo Tree Search), optimized natively for **Apple Silicon GPU (Metal Performance Shaders - MPS)**.

---

## 🌟 Key Highlights & Engineering Achievements

- **100% Authentic Thai Checkers Rules Engine (`env/`)**:
  - **32-Square Playable Dark Grid**: Strictly maintains Thai coordinate rules (Top-Left corner light, bottom-left $(0, 0)$ dark, playable squares where $(r + c) \pmod 2 == 0$).
  - **Forced Capture (กฎบังคับกิน)**: If any capture exists on the board, non-capturing moves are strictly illegal.
  - **Multi-Jump Sequence (การกินต่อเนื่อง)**: Atomic sub-turn transitions where chains of captures must be completed.
  - **Promotion Rule (เข้าฮอสแล้วหยุดเดิน)**: Reaching the opponent's back rank promotes the pawn to King and **ends the turn immediately**.
  - **Flying King (ฮอสบิน)**: Long-range diagonal flights across any number of empty squares, jumping enemy pieces and choosing any open landing square beyond the captured piece.
  - **Draw Rules**: 3-fold position repetition detection and 50-move / 100-ply rule without captures.
- **Dual-Head Deep ResNet (`models/`)**:
  - **Policy Head**: Predicts probability distribution over a 1,024-dimensional action space ($32 \times 32$ transitions).
  - **Value Head**: Evaluates board state with scalar $V(s) \in [-1.0, +1.0]$.
  - **Native Apple Silicon Metal Acceleration**: Automatic detection and zero-overhead execution on `torch.device("mps")`.
- **Generalized MCTS with Sub-turn Attribution (`mcts/`)**:
  - **PUCT (Predictor Upper Confidence bounds for Trees)** selection formula.
  - **Dirichlet Exploration Noise** ($\text{Dir}(\alpha=0.3)$) at root during self-play.
  - **Multi-Jump Sub-turn Value Backpropagation**: Correctly preserves reward attribution when the same player acts across multiple consecutive jumps.
- **Benchmark Arena & Heuristic Baseline (`baseline/` & `training/arena.py`)**:
  - Alpha-Beta Minimax player with piece-square center control and advancement heuristics.
  - Automated Elo tracking and model gating ($\ge 55\%$ win rate requirement to become the new best model).
- **Dual-Mode Modern Web Application (`ui/` & `server.py`)**:
  - **Mode 1 (Play & Analysis)**: Interactive 8x8 board with standard algebraic notation ($A-H, 1-8$), move validation, forced capture alerts, live **AI Brain Value Gauge** (Win Probability Bar) and **MCTS Thought Stream** (candidate moves with visit percentages).
  - **Mode 2 (Live Training Dashboard)**: Real-time **Chart.js Loss Curves** (Total, Policy, Value), live animated **Self-Play Board Visualizer** streaming moves asynchronously over **WebSockets**, and one-click training controls (Turbo vs Live Watch speed).
  - Terminal CLI mode (`play_cli.py`) with colored ASCII board.

---

## 📐 System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Interactive Modern Web UI                         │
│       (FastAPI + Tailwind CSS + Web Audio + Live MCTS Thought Stream)  │
└───────────────────▲────────────────────────────────▲───────────────────┘
                    │                                │
┌───────────────────┴───────────────┐     ┌──────────┴───────────────────┐
│     play_cli.py / server.py       │     │       training/arena.py      │
│      (Real-time Human vs AI)      │     │  (Head-to-Head Model Elo)    │
└───────────────────▲───────────────┘     └──────────▲───────────────────┘
                    │                                │
┌───────────────────┴────────────────────────────────┴───────────────────┐
│                     Monte Carlo Tree Search (PUCT)                     │
│         Q(s, a) + c_puct * P(s, a) * sqrt(sum N(s, b)) / (1 + N(s, a)) │
└───────────────────▲────────────────────────────────▲───────────────────┘
                    │                                │
┌───────────────────┴───────────────┐     ┌──────────┴───────────────────┐
│         models/net.py             │     │      env/thai_checkers.py    │
│       Dual-Head ResNet            │     │  100% Authentic Thai Engine  │
│  • Policy Head (1,024 Actions)    │     │  • 32 Dark Squares           │
│  • Value Head (V in [-1, +1])     │     │  • Forced Capture & Multijump│
│  • Apple Silicon MPS Accelerated  │     │  • Flying King & Promotion   │
└───────────────────────────────────┘     └──────────────────────────────┘
```

---

## 🧠 Mathematical Formulation

### 1. PUCT Action Selection
During tree search, child action $a^*$ is selected using:
$$a^* = \arg\max_a \left( Q(s, a) + c_{\text{puct}} \cdot P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)$$

### 2. Dirichlet Noise Exploration
To promote exploratory diversity during self-play, Dirichlet noise is injected into the root priors:
$$P'(s, a) = (1 - \epsilon) P(s, a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha = 0.3), \quad \epsilon = 0.25$$

### 3. AlphaZero Optimization Loss
The network weights $W$ are updated by minimizing:
$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^B \left[ (z_i - v_i)^2 - \boldsymbol{\pi}_i^\top \log \mathbf{p}_i \right] + c \|W\|_2^2$$
- $(z_i - v_i)^2$: Mean Squared Error between terminal game outcome $z_i \in \{-1, 0, 1\}$ and predicted value $v_i$.
- $\boldsymbol{\pi}_i^\top \log \mathbf{p}_i$: Cross-Entropy between MCTS search distribution $\boldsymbol{\pi}_i$ and policy output $\mathbf{p}_i$.
- $c \|W\|_2^2$: $L_2$ weight regularization via AdamW.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment
Requires Python 3.11+ and macOS (Apple Silicon recommended for MPS acceleration):
```bash
git clone https://github.com/your-username/Thai-Checkers-AlphaZero.git
cd Thai-Checkers-AlphaZero

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
Run the complete test suite covering all Thai Checkers rules and neural network components:
```bash
pytest tests/ -v
```

### 3. Launch the Interactive Web UI
Start the FastAPI server and challenge AlphaZero in your browser:
```bash
python server.py
```
Open **`http://localhost:8000`** in your browser:
- 🎮 Click or drag to move pieces.
- ⚡ Visual indicators for forced captures and valid landing squares.
- 🧠 Watch the live win-rate evaluation meter and MCTS candidate visit percentages.
- 🔊 Authentic wooden piece sound effects synthesized via Web Audio API.

### 4. Play via Terminal CLI
Play against AlphaZero directly in your terminal with colored ASCII board:
```bash
python play_cli.py --sims 150 --side 1
```

### 5. Train from Scratch (Self-Play Pipeline)
Launch the end-to-end AlphaZero training loop:
```bash
python train.py --num_iters 10 --episodes_per_iter 20 --mcts_sims 100 --batch_size 64
```

### 6. Benchmark in the Arena
Evaluate AlphaZero against the Alpha-Beta Minimax baseline:
```bash
python arena_eval.py --opponent minimax --games 10 --mcts_sims 100
```

---

## 📂 Project Structure

```
Thai-Checkers-AlphaZero/
├── env/
│   ├── __init__.py
│   └── thai_checkers.py      # Authentic 100% Thai Checkers Engine & Bitboard
├── models/
│   ├── __init__.py
│   └── net.py                # Dual-Head ResNet (Policy & Value) with MPS support
├── mcts/
│   ├── __init__.py
│   └── mcts.py               # Monte Carlo Tree Search with PUCT & Dirichlet Noise
├── baseline/
│   ├── __init__.py
│   └── minimax.py            # Alpha-Beta Minimax with Thai Checkers Heuristic
├── training/
│   ├── __init__.py
│   ├── self_play.py          # Tabula Rasa Self-Play Game Generator
│   ├── trainer.py            # Replay Buffer, Loss Optimization, Checkpoints
│   └── arena.py              # Tournament Arena & Elo Evaluator
├── ui/
│   └── static/
│       ├── index.html        # Modern Tailwind Web Interface
│       └── app.js            # Frontend Board Engine & Web Audio Effects
├── tests/
│   ├── test_rules.py         # Complete rule verification test suite
│   └── test_model.py         # Network, MCTS, and MPS unit tests
├── server.py                 # FastAPI Web Server
├── play_cli.py               # Interactive Terminal CLI Game
├── train.py                  # Training Pipeline CLI
├── arena_eval.py             # Benchmark Tournament Runner
├── requirements.txt          # Frozen dependencies
├── pytest.ini                # Pytest configuration
└── README.md                 # Bilingual Documentation
```

---

## 📜 License
This project is open-source and released under the **Apache-2.0 License**.
