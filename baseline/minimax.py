"""Alpha-Beta Minimax Baseline Agent for Thai Checkers.

Features:
- Alpha-Beta Pruning with move ordering (captures prioritized)
- Quiescence extension for capture sequences
- Strategic Thai Checkers heuristic evaluation (Piece values, advancement, center control, mobility)
"""

from __future__ import annotations
import random
from typing import Optional, Tuple, List
from env.thai_checkers import Board, Piece, Player, Move, NUM_SQUARES, SQ_TO_COORD


# Positional weights: Center squares have higher value
CENTER_WEIGHTS = [
    0, 0, 0, 0,    # Row 0
    1, 2, 2, 1,    # Row 1
    1, 4, 4, 1,    # Row 2
    2, 6, 6, 2,    # Row 3
    2, 6, 6, 2,    # Row 4
    1, 4, 4, 1,    # Row 5
    1, 2, 2, 1,    # Row 6
    0, 0, 0, 0,    # Row 7
]


def evaluate_board(board: Board) -> float:
    """Heuristic static evaluation of board from perspective of Player 1 (positive = P1 advantage)."""
    score = 0.0

    p1_pieces = 0
    p2_pieces = 0

    for sq in range(NUM_SQUARES):
        val = board.squares[sq]
        if val == Piece.EMPTY:
            continue

        r, c = SQ_TO_COORD[sq]
        center = CENTER_WEIGHTS[sq]

        if val == Piece.P1_PAWN:
            p1_pieces += 1
            # Pawn material + advancement bonus + center control
            score += 100.0 + (r * 6.0) + (center * 2.0)
            # Back row defense bonus (row 0 pawns guard against opponent king invasion)
            if r == 0:
                score += 5.0
        elif val == Piece.P1_KING:
            p1_pieces += 1
            # King material + center control
            score += 350.0 + (center * 4.0)
        elif val == Piece.P2_PAWN:
            p2_pieces += 1
            # Pawn material + advancement bonus towards row 0
            score -= 100.0 + ((7 - r) * 6.0) + (center * 2.0)
            if r == 7:
                score -= 5.0
        elif val == Piece.P2_KING:
            p2_pieces += 1
            score -= 350.0 + (center * 4.0)

    # Endgame bonus: if ahead in material, reward piece trade/captures
    if p1_pieces > p2_pieces:
        score += (8 - p2_pieces) * 10.0
    elif p2_pieces > p1_pieces:
        score -= (8 - p1_pieces) * 10.0

    return score


class MinimaxAgent:
    def __init__(self, depth: int = 4):
        self.depth = depth

    def select_move(self, board: Board) -> Move:
        """Selects best move using Alpha-Beta Minimax."""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")
        if len(legal_moves) == 1:
            return legal_moves[0]

        is_maximizing = (board.current_player == Player.P1)
        best_move = legal_moves[0]

        if is_maximizing:
            best_val = -float("inf")
            for move in legal_moves:
                clone_b = board.clone()
                clone_b.step(move)
                val = self._minimax(clone_b, self.depth - 1, -float("inf"), float("inf"))
                if val > best_val:
                    best_val = val
                    best_move = move
        else:
            best_val = float("inf")
            for move in legal_moves:
                clone_b = board.clone()
                clone_b.step(move)
                val = self._minimax(clone_b, self.depth - 1, -float("inf"), float("inf"))
                if val < best_val:
                    best_val = val
                    best_move = move

        return best_move

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        done, winner = board.check_game_over()
        if done:
            if winner == Player.P1:
                return 100000.0 + depth
            elif winner == Player.P2:
                return -100000.0 - depth
            return 0.0  # Draw

        if depth <= 0:
            # Quiescence check: if captures are pending, extend search by 1
            legal_moves = board.get_legal_moves()
            if not legal_moves or not legal_moves[0].is_capture:
                return evaluate_board(board)

        legal_moves = board.get_legal_moves()
        if not legal_moves:
            # Stalemate = loss for current player
            return -100000.0 if board.current_player == Player.P1 else 100000.0

        # Prioritize captures in move ordering
        legal_moves.sort(key=lambda m: m.is_capture, reverse=True)

        if board.current_player == Player.P1:
            max_eval = -float("inf")
            for move in legal_moves:
                clone_b = board.clone()
                clone_b.step(move)
                eval_val = self._minimax(clone_b, depth - 1, alpha, beta)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                clone_b = board.clone()
                clone_b.step(move)
                eval_val = self._minimax(clone_b, depth - 1, alpha, beta)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval


class RandomAgent:
    """Random benchmark agent."""

    def select_move(self, board: Board) -> Move:
        moves = board.get_legal_moves()
        if not moves:
            raise ValueError("No legal moves available.")
        return random.choice(moves)
