"""Tournament evaluation runner between AlphaZero, Minimax, and Random agents."""

import argparse
import os
import torch

from env.thai_checkers import Board
from models.net import ThaiCheckersNet, get_device
from training.arena import Arena, make_mcts_agent, make_minimax_agent, make_random_agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Thai Checkers AlphaZero against baselines.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--opponent", type=str, choices=["minimax", "random"], default="minimax", help="Opponent type")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--mcts_sims", type=int, default=100, help="MCTS rollouts per move for AlphaZero")
    parser.add_argument("--minimax_depth", type=int, default=3, help="Minimax depth")
    parser.add_argument("--verbose", action="store_true", help="Print game progress")
    args = parser.parse_args()

    device = get_device()
    model = ThaiCheckersNet()

    if os.path.exists(args.model_path):
        print(f"📦 Loading weights from {args.model_path}...")
        model.load_checkpoint(args.model_path, device=device)
    else:
        print(f"⚠️ Checkpoint {args.model_path} not found. Using untrained initialization.")

    model.to(device)
    az_agent = make_mcts_agent(model, sims=args.mcts_sims)

    if args.opponent == "minimax":
        opp_agent = make_minimax_agent(depth=args.minimax_depth)
        opp_name = f"Minimax (depth={args.minimax_depth})"
    else:
        opp_agent = make_random_agent()
        opp_name = "Random Player"

    print(f"\n==========================================")
    print(f"⚔️ TOURNAMENT: AlphaZero ({args.mcts_sims} sims) vs {opp_name}")
    print(f"🎯 Total Games: {args.games} (alternating colors)")
    print(f"==========================================")

    arena = Arena(az_agent, opp_agent)
    res = arena.play_games(num_games=args.games, verbose=args.verbose)

    print("\n------------------------------------------")
    print(f"📊 Final Results:")
    print(f"   AlphaZero Wins: {res['agent1_wins']}")
    print(f"   {opp_name} Wins: {res['agent2_wins']}")
    print(f"   Draws: {res['draws']}")
    print(f"   AlphaZero Win Rate: {res['agent1_win_rate'] * 100:.1f}%")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
