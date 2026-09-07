"""AlphaZero Training Pipeline with MPS Acceleration.

Coordinates:
- Self-play data collection
- Experience Replay Buffer
- Loss optimization (Policy Cross-Entropy + Value MSE + Weight Decay)
- Model checkpointing & Arena tournament gating
"""

from __future__ import annotations
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from models.net import ThaiCheckersNet, get_device
from env.thai_checkers import Move, SQ_TO_COORD, sq_to_algebraic, Board
from training.self_play import SelfPlayWorker
from training.arena import Arena, make_mcts_agent


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs_per_iter: int = 4
    num_iters: int = 10
    episodes_per_iter: int = 10
    mcts_sims: int = 80
    arena_games: int = 10
    arena_threshold: float = 0.55  # Challenger must achieve >= 55% win rate
    buffer_capacity: int = 30000
    checkpoint_dir: str = "checkpoints"


class ReplayBuffer:
    def __init__(self, capacity: int = 30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample: Tuple[np.ndarray, np.ndarray, float]) -> None:
        self.buffer.append(sample)

    def extend(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = random.sample(self.buffer, batch_size)
        states, pis, vs = zip(*samples)

        state_tensor = torch.from_numpy(np.array(states, dtype=np.float32))
        pi_tensor = torch.from_numpy(np.array(pis, dtype=np.float32))
        v_tensor = torch.from_numpy(np.array(vs, dtype=np.float32)).unsqueeze(1)

        return state_tensor, pi_tensor, v_tensor

    def __len__(self) -> int:
        return len(self.buffer)


class Trainer:
    def __init__(
        self,
        model: Optional[ThaiCheckersNet] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_episode_end: Optional[Callable[[int, int, int, int], None]] = None,
        on_epoch_end: Optional[Callable[[int, int, float, float, float], None]] = None,
        on_iter_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        visual_delay: float = 0.0,
    ):
        self.config = config or TrainerConfig()
        self.device = device or get_device()
        self.model = model or ThaiCheckersNet()
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.replay_buffer = ReplayBuffer(capacity=self.config.buffer_capacity)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.self_play_stats: Dict[str, int] = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "total_games": 0,
        }

        self.stop_requested = False
        self.on_step = on_step
        self.on_episode_end = on_episode_end
        self.on_epoch_end = on_epoch_end
        self.on_iter_end = on_iter_end
        self.visual_delay = visual_delay

    def train_epoch(self) -> Tuple[float, float, float]:
        """Runs one training epoch over random batches from replay buffer."""
        self.model.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        num_batches = max(1, len(self.replay_buffer) // self.config.batch_size)

        for _ in range(num_batches):
            states, target_pis, target_vs = self.replay_buffer.sample(self.config.batch_size)
            states = states.to(self.device)
            target_pis = target_pis.to(self.device)
            target_vs = target_vs.to(self.device)

            self.optimizer.zero_grad()
            out_pi_logits, out_v = self.model(states)

            # Policy Loss: Cross-entropy with target probability distribution
            log_probs = F.log_softmax(out_pi_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))

            # Value Loss: Mean Squared Error
            value_loss = F.mse_loss(out_v, target_vs)

            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_p_loss += float(policy_loss.item())
            total_v_loss += float(value_loss.item())

        return total_loss / num_batches, total_p_loss / num_batches, total_v_loss / num_batches

    def run_training_loop(self) -> Dict[str, List[float]]:
        """Executes the full AlphaZero self-play and training loop."""
        history: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "win_rate": [],
        }

        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        if not os.path.exists(best_model_path):
            self.model.save_checkpoint(best_model_path)

        for iteration in range(1, self.config.num_iters + 1):
            if self.stop_requested:
                print("🛑 Stop requested, terminating training loop.")
                break

            print(f"\n==========================================")
            print(f"🚀 Iteration {iteration}/{self.config.num_iters} [Device: {self.device}]")
            print(f"==========================================")

            # 1. Self-Play: Collect new episodes
            print(f"🎮 Generating {self.config.episodes_per_iter} self-play episodes...")
            worker = SelfPlayWorker(self.model, mcts_simulations=self.config.mcts_sims)
            new_samples = 0

            for ep in range(1, self.config.episodes_per_iter + 1):
                if self.stop_requested:
                    break

                def _step_cb(b: Board, a: int, acting_p: int):
                    if self.on_step:
                        f_sq, t_sq = Move.from_action_id(a)
                        self.on_step({
                            "iteration": iteration,
                            "total_iters": self.config.num_iters,
                            "episode": ep,
                            "total_episodes": self.config.episodes_per_iter,
                            "ply_count": b.ply_count,
                            "acting_player": acting_p,
                            "current_player": b.current_player,
                            "from_sq": f_sq,
                            "to_sq": t_sq,
                            "from_coord": SQ_TO_COORD[f_sq],
                            "to_coord": SQ_TO_COORD[t_sq],
                            "from_algebraic": sq_to_algebraic(f_sq),
                            "to_algebraic": sq_to_algebraic(t_sq),
                            "notation": f"{sq_to_algebraic(f_sq)}-{sq_to_algebraic(t_sq)}",
                            "squares": list(b.squares),
                            "buffer_size": len(self.replay_buffer),
                        })
                    if self.visual_delay > 0:
                        time.sleep(self.visual_delay)

                samples, winner, reason = worker.play_game(step_callback=_step_cb)
                self.replay_buffer.extend(samples)
                new_samples += len(samples)

                self.self_play_stats["total_games"] += 1
                if winner == 1:
                    self.self_play_stats["white_wins"] += 1
                    winner_str = "⚪ ขาวชนะ"
                elif winner == -1:
                    self.self_play_stats["black_wins"] += 1
                    winner_str = "⚫ ดำชนะ"
                else:
                    self.self_play_stats["draws"] += 1
                    winner_str = "🤝 เสมอ"

                print(
                    f"  Episode {ep}/{self.config.episodes_per_iter}: {winner_str} ({reason}) | +{len(samples)} samples "
                    f"(Total buffer: {len(self.replay_buffer)}) "
                    f"[⚪ {self.self_play_stats['white_wins']} | ⚫ {self.self_play_stats['black_wins']} | 🤝 {self.self_play_stats['draws']}]"
                )

                if self.on_episode_end:
                    self.on_episode_end(
                        iteration,
                        ep,
                        self.config.episodes_per_iter,
                        len(self.replay_buffer),
                        winner,
                        dict(self.self_play_stats),
                        reason,
                    )

                if self.visual_delay > 0:
                    time.sleep(1.5)  # Pause to let user observe final board and ending reason

            if self.stop_requested:
                break

            # 2. Optimization
            last_loss, last_p_loss, last_v_loss = 0.0, 0.0, 0.0
            if len(self.replay_buffer) >= self.config.batch_size:
                print(f"🧠 Training model for {self.config.epochs_per_iter} epochs...")
                for ep in range(1, self.config.epochs_per_iter + 1):
                    loss, p_loss, v_loss = self.train_epoch()
                    last_loss, last_p_loss, last_v_loss = loss, p_loss, v_loss
                    print(f"  Epoch {ep}: Total Loss = {loss:.4f} (Policy = {p_loss:.4f}, Value = {v_loss:.4f})")
                    if self.on_epoch_end:
                        self.on_epoch_end(iteration, ep, loss, p_loss, v_loss)

                history["loss"].append(last_loss)
                history["policy_loss"].append(last_p_loss)
                history["value_loss"].append(last_v_loss)

            # 3. Checkpoint & Arena Tournament
            current_checkpoint = os.path.join(
                self.config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt"
            )
            self.model.save_checkpoint(current_checkpoint)

            # Evaluate challenger vs current best model
            arena_win_rate = 0.0
            accepted = False
            if os.path.exists(best_model_path):
                print(f"⚔️ Evaluating in Arena vs Best Model ({self.config.arena_games} games)...")
                best_model = ThaiCheckersNet()
                best_model.load_checkpoint(best_model_path, device=self.device)

                challenger_fn = make_mcts_agent(self.model, sims=self.config.mcts_sims)
                best_fn = make_mcts_agent(best_model, sims=self.config.mcts_sims)

                arena = Arena(challenger_fn, best_fn)
                arena_res = arena.play_games(num_games=self.config.arena_games)
                arena_win_rate = arena_res["agent1_win_rate"]
                history["win_rate"].append(arena_win_rate)

                print(
                    f"  Arena Result: {arena_res['agent1_wins']}W - "
                    f"{arena_res['agent2_wins']}L - {arena_res['draws']}D "
                    f"(Win Rate: {arena_win_rate * 100:.1f}%)"
                )

                if arena_win_rate >= self.config.arena_threshold:
                    accepted = True
                    print(f"🏆 NEW BEST MODEL ACCEPTED! (Win rate {arena_win_rate*100:.1f}% >= {self.config.arena_threshold*100:.1f}%)")
                    self.model.save_checkpoint(best_model_path)
                else:
                    print("🛡️ Best model retained.")
            else:
                self.model.save_checkpoint(best_model_path)
                accepted = True

            if self.on_iter_end:
                self.on_iter_end(iteration, {
                    "loss": last_loss,
                    "policy_loss": last_p_loss,
                    "value_loss": last_v_loss,
                    "win_rate": arena_win_rate,
                    "accepted": accepted,
                    "buffer_size": len(self.replay_buffer),
                })

        return history
