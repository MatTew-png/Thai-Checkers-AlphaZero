"""
Minimax Expert Tutor & Promotion Defense Booster.
Generates expert demonstrations from Minimax and fine-tunes best_model.pt directly on MPS GPU.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from env.thai_checkers import Board, Player, Move, Piece, NUM_SQUARES, ACTION_SPACE_SIZE
from models.net import ThaiCheckersNet, get_device
from training.minimax_warmup import MinimaxWarmupWorker
from training.trainer import ReplayBuffer


def run_minimax_boost(num_games: int = 25, depth: int = 3, epochs: int = 8, lr: float = 4e-4):
    device = get_device()
    print(f"🚀 Initializing Minimax Expert Booster on {device}...")

    model = ThaiCheckersNet().to(device)
    best_path = os.path.join("checkpoints", "best_model.pt")
    if os.path.exists(best_path):
        print(f"📦 Loaded base weights from {best_path}")
        model.load_checkpoint(best_path, device=device)
    else:
        print("⚠️ Warning: best_model.pt not found, using initialized weights")

    # 1. Generate Expert Games from Minimax
    print(f"⚡ Generating {num_games} tactical games using Minimax (Depth {depth})...")
    worker = MinimaxWarmupWorker(depth=depth, temperature=0.3)
    buffer = ReplayBuffer(capacity=10000)

    start_time = time.time()
    for g in range(1, num_games + 1):
        records, winner, reason = worker.play_game(max_plies=100)
        for state_tensor, target_pi, acting_player in records:
            z = 0.0 if winner == 0 else (1.0 if winner == acting_player else -1.0)
            buffer.push((state_tensor, target_pi, z))

        w_str = "⚪ ขาวชนะ" if winner == 1 else ("⚫ ดำชนะ" if winner == -1 else "🤝 เสมอ")
        if g % 5 == 0 or g == num_games:
            print(f"  [Minimax {g}/{num_games}] {w_str} ({reason}) | Buffer: {len(buffer)} samples")

    gen_time = time.time() - start_time
    print(f"✅ Generated {len(buffer)} expert positions in {gen_time:.1f}s")

    # 2. Supervised Fine-Tuning on MPS GPU
    print(f"🧠 Fine-tuning Neural Network on {device} for {epochs} epochs...")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    batch_size = 64
    num_batches = max(1, len(buffer) // batch_size)

    for ep in range(1, epochs + 1):
        total_loss, total_pi, total_v = 0.0, 0.0, 0.0
        for _ in range(num_batches):
            states, target_pis, target_vs = buffer.sample(batch_size)
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.to(device)

            optimizer.zero_grad()
            out_pi, out_v = model(states)

            log_probs = F.log_softmax(out_pi, dim=1)
            pi_loss = -torch.mean(torch.sum(target_pis * log_probs, dim=1))
            v_loss = F.mse_loss(out_v, target_vs)
            loss = pi_loss + v_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pi += pi_loss.item()
            total_v += v_loss.item()

        avg_loss = total_loss / num_batches
        avg_pi = total_pi / num_batches
        avg_v = total_v / num_batches
        print(f"  Epoch {ep}/{epochs}: Total Loss = {avg_loss:.4f} (Policy = {avg_pi:.4f}, Value = {avg_v:.4f})")

    # 3. Save directly to checkpoints/best_model.pt
    model.save_checkpoint(best_path)
    print(f"💾 Successfully saved tuned weights to {best_path}!")


if __name__ == "__main__":
    run_minimax_boost(num_games=25, depth=3, epochs=8)
