"""Arena for Head-to-Head Model Evaluation and Elo Rating Tracking."""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np

from env.thai_checkers import Board, Player, Move
from models.net import ThaiCheckersNet
from mcts.mcts import MCTS, MCTSConfig
from baseline.minimax import MinimaxAgent, RandomAgent


class Arena:
    """Head-to-head tournament evaluator."""

    def __init__(self, agent1_fn: Callable[[Board], int], agent2_fn: Callable[[Board], int]):
        """
        agent1_fn: function(Board) -> action_id
        agent2_fn: function(Board) -> action_id
        """
        self.agent1_fn = agent1_fn
        self.agent2_fn = agent2_fn

    def play_game(self, agent1_as_p1: bool = True) -> Tuple[int, int]:
        """Plays a single game between the two agents.
        
        Returns:
            (winner, total_plies)
            winner: 1 if agent1 won, -1 if agent2 won, 0 if draw
        """
        board = Board()
        board.setup_initial_position()

        p1_agent = self.agent1_fn if agent1_as_p1 else self.agent2_fn
        p2_agent = self.agent2_fn if agent1_as_p1 else self.agent1_fn

        done = False
        winner: Optional[int] = None

        while not done:
            current_fn = p1_agent if board.current_player == Player.P1 else p2_agent
            action = current_fn(board)
            board.step(action)
            done, winner = board.check_game_over()

        # Map winner to agent1 (1), agent2 (-1), or draw (0)
        if winner == 0 or winner is None:
            return 0, board.ply_count
        if agent1_as_p1:
            return (1 if winner == Player.P1 else -1), board.ply_count
        else:
            return (1 if winner == Player.P2 else -1), board.ply_count

    def play_games(self, num_games: int = 20, verbose: bool = False) -> Dict[str, Any]:
        """Plays num_games between agent 1 and agent 2, alternating starting colors.
        
        Returns:
            Dict containing wins, losses, draws, win_rate.
        """
        a1_wins = 0
        a2_wins = 0
        draws = 0

        for i in range(num_games):
            # Alternate starting colors
            agent1_first = (i % 2 == 0)
            res, plies = self.play_game(agent1_as_p1=agent1_first)

            if res == 1:
                a1_wins += 1
                outcome_str = "Agent 1 Won"
            elif res == -1:
                a2_wins += 1
                outcome_str = "Agent 2 Won"
            else:
                draws += 1
                outcome_str = "Draw"

            if verbose:
                print(f"Game {i + 1}/{num_games} ({plies} plies): {outcome_str}")

        win_rate = (a1_wins + 0.5 * draws) / num_games
        return {
            "agent1_wins": a1_wins,
            "agent2_wins": a2_wins,
            "draws": draws,
            "total_games": num_games,
            "agent1_win_rate": win_rate,
        }


def make_mcts_agent(model: ThaiCheckersNet, sims: int = 100, temp: float = 0.0) -> Callable[[Board], int]:
    """Helper creating an action selector function from an MCTS model."""
    mcts = MCTS(model, MCTSConfig(num_simulations=sims))

    def agent(board: Board) -> int:
        return mcts.select_best_move(board, num_simulations=sims, temperature=temp)

    return agent


def make_minimax_agent(depth: int = 4) -> Callable[[Board], int]:
    """Helper creating an action selector function from a MinimaxAgent."""
    mm = MinimaxAgent(depth=depth)

    def agent(board: Board) -> int:
        move = mm.select_move(board)
        return move.action_id

    return agent


def make_random_agent() -> Callable[[Board], int]:
    """Helper creating a random action selector."""
    rnd = RandomAgent()

    def agent(board: Board) -> int:
        move = rnd.select_move(board)
        return move.action_id

    return agent
