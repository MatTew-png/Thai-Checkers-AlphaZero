"""One-command training entry point for Thai-Checkers-AlphaZero."""

import argparse
import torch
from models.net import ThaiCheckersNet, get_device
from training.trainer import Trainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser(description="Train Thai Checkers AlphaZero from scratch.")
    parser.add_argument("--num_iters", type=int, default=5, help="Number of training iterations")
    parser.add_argument("--episodes_per_iter", type=int, default=5, help="Self-play games per iteration")
    parser.add_argument("--mcts_sims", type=int, default=50, help="MCTS rollouts per move")
    parser.add_argument("--epochs_per_iter", type=int, default=4, help="Epochs per iteration")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for gradient updates")
    parser.add_argument("--arena_games", type=int, default=6, help="Games to evaluate challenger vs best")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing best_model.pt if available")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="Start training from scratch")
    args = parser.parse_args()

    import os
    device = get_device()
    print(f"🎮 Initializing Thai-Checkers-AlphaZero Training Pipeline")
    print(f"⚡ Device: {device} ({'Apple Silicon Metal' if device.type == 'mps' else device.type.upper()})")

    config = TrainerConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs_per_iter=args.epochs_per_iter,
        num_iters=args.num_iters,
        episodes_per_iter=args.episodes_per_iter,
        mcts_sims=args.mcts_sims,
        arena_games=args.arena_games,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = ThaiCheckersNet()
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if args.resume and os.path.exists(best_path):
        print(f"📦 Resuming from existing best model: {best_path}")
        model.load_checkpoint(best_path, device=device)
    else:
        print("🌱 Starting fresh training (no prior weights loaded).")

    trainer = Trainer(model=model, config=config, device=device)
    history = trainer.run_training_loop()
    print("\n✅ Training complete! Best model saved to checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
