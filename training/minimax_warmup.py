"""Expert Knowledge Warmup Generator using Minimax Self-Play.

Injects high-quality tactical and endgame demonstrations directly into the AlphaZero
Replay Buffer, bootstrapping pawn advancement, piece defense, and flying king trapping.
"""

from __future__ import annotations
import random
from typing import List, Tuple, Optional, Callable
import numpy as np

from env.thai_checkers import (
    Board,
    Player,
    Move,
    ACTION_SPACE_SIZE,
    NUM_SQUARES,
    canonicalize_action,
)
from baseline.minimax import evaluate_board, order_moves


class MinimaxWarmupWorker:
    def __init__(self, depth: int = 3, temperature: float = 0.4):
        self.depth = depth
        self.temperature = temperature

    def evaluate_moves(self, board: Board) -> List[Tuple[Move, float]]:
        """Evaluates all legal moves using minimax up to self.depth."""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return []

        scored_moves: List[Tuple[Move, float]] = []
        is_maximizing = (board.current_player == Player.P1)

        for move in legal_moves:
            clone_b = board.clone()
            clone_b.step(move)
            # Search remaining depth
            score = self._minimax(
                clone_b,
                depth=self.depth - 1,
                alpha=-float("inf"),
                beta=float("inf"),
                is_maximizing=(clone_b.current_player == Player.P1),
            )
            # From perspective of acting player
            player_score = score if is_maximizing else -score
            scored_moves.append((move, player_score))

        return scored_moves

    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> float:
        done, winner = board.check_game_over()
        if done:
            if winner == Player.P1:
                return 100000.0 + depth
            elif winner == Player.P2:
                return -100000.0 - depth
            return 0.0

        if depth <= 0:
            legal_moves = board.get_legal_moves()
            if not legal_moves or not legal_moves[0].is_capture:
                return evaluate_board(board)

        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return -100000.0 if board.current_player == Player.P1 else 100000.0

        ordered_moves = order_moves(board, legal_moves)

        if is_maximizing:
            max_eval = -float("inf")
            for m in ordered_moves:
                clone_b = board.clone()
                clone_b.step(m)
                ev = self._minimax(clone_b, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for m in ordered_moves:
                clone_b = board.clone()
                clone_b.step(m)
                ev = self._minimax(clone_b, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    break
            return min_eval

    def play_game(
        self,
        max_plies: int = 120,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], int, str]:
        """Plays one game between two Minimax agents with soft temperature exploration.

        Returns:
            Tuple of (episode_records, final_winner, reason):
                episode_records: List of (canonical_state_tensor, target_pi, acting_player)
                final_winner: 1, -1, or 0
                reason: game termination explanation
        """
        board = Board()
        board.setup_initial_position()

        episode_records: List[Tuple[np.ndarray, np.ndarray, int]] = []
        done = False
        winner: Optional[int] = None

        while not done and board.ply_count < max_plies:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                done, winner = board.check_game_over()
                break

            scored_moves = self.evaluate_moves(board)
            if not scored_moves:
                break

            # Build target policy pi (4096)
            canonical_board = board.get_canonical_form()
            state_tensor = canonical_board.get_tensor_representation()
            acting_player = board.current_player

            pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

            # Softmax or temperature-scaled distribution over move scores
            # Canonicalize actions for policy vector
            scores = np.array([s for _, s in scored_moves], dtype=np.float32)
            # Normalize scores to prevent overflow in exp
            scores = scores - np.max(scores)
            temp = max(self.temperature, 0.1)
            exp_scores = np.exp(scores / (temp * 50.0))  # Scale factor for heuristic score range
            probs = exp_scores / (np.sum(exp_scores) + 1e-8)

            # Map to actions for the policy target in canonical space
            for (m, _), prob in zip(scored_moves, probs):
                can_act = canonicalize_action(m.action_id, acting_player)
                pi[can_act] = prob

            # Normalize pi
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                first_can_act = canonicalize_action(scored_moves[0][0].action_id, acting_player)
                pi[first_can_act] = 1.0

            episode_records.append((state_tensor, pi, acting_player))

            # Select move for the actual board (sampling with probs)
            chosen_idx = int(np.random.choice(len(scored_moves), p=probs))
            chosen_move = scored_moves[chosen_idx][0]

            board.step(chosen_move)
            done, winner = board.check_game_over()

        final_winner = 0 if winner is None else winner
        reason = board.get_game_over_reason()
        return episode_records, final_winner, reason


def generate_minimax_warmup(
    num_games: int = 10,
    depth: int = 3,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generates warmup self-play games from Minimax experts.

    Args:
        num_games: Number of self-play games to simulate
        depth: Minimax search depth (3 is fast & tactically sound; 4 is deeper)
        progress_callback: Optional fn(game_idx, total_games, accumulated_samples, reason)

    Returns:
        List of (canonical_state_tensor, target_pi, outcome_z) ready for ReplayBuffer
    """
    worker = MinimaxWarmupWorker(depth=depth)
    all_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for g in range(1, num_games + 1):
        records, final_winner, reason = worker.play_game()

        game_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for state_tensor, target_pi, acting_player in records:
            if final_winner == 0:
                z = 0.0
            elif final_winner == acting_player:
                z = 1.0
            else:
                z = -1.0
            game_samples.append((state_tensor, target_pi, z))

        all_samples.extend(game_samples)

        if progress_callback:
            progress_callback(g, num_games, len(all_samples), reason)

    return all_samples
