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

# Grandmaster Thai Checkers Pawn Advancement & Promotion Threat Table
# Back-rank pawns (row 0 for P1, row 7 for P2) are critical King shields (140.0).
# Approaching promotion (row 5: 280.0, row 6: 460.0) triggers immediate tactical priority.
P1_PAWN_ROW_SCORES = [140.0, 105.0, 115.0, 130.0, 170.0, 280.0, 460.0, 550.0]
P2_PAWN_ROW_SCORES = [550.0, 460.0, 280.0, 170.0, 130.0, 115.0, 105.0, 140.0]
KING_VALUE = 550.0


def evaluate_board(board: Board) -> float:
    """Grandmaster heuristic static evaluation from perspective of Player 1.

    Includes:
    - Deep pawn advancement & promotion threat curve
    - Back-rank gate protection (prevents giving away king squares)
    - Flying King mobility and dominance value (550.0)
    - King endgame hunting/chase bonus
    - Center control and material disparity amplification
    """
    score = 0.0
    p1_pieces = 0
    p2_pieces = 0
    p1_kings = 0
    p2_kings = 0
    p1_sqs: List[int] = []
    p2_sqs: List[int] = []

    for sq in range(NUM_SQUARES):
        val = board.squares[sq]
        if val == Piece.EMPTY:
            continue

        r, c = SQ_TO_COORD[sq]
        center = CENTER_WEIGHTS[sq]

        if val == Piece.P1_PAWN:
            p1_pieces += 1
            score += P1_PAWN_ROW_SCORES[r] + (center * 2.0)
            p1_sqs.append(sq)
        elif val == Piece.P1_KING:
            p1_pieces += 1
            p1_kings += 1
            score += KING_VALUE + (center * 4.0)
            p1_sqs.append(sq)
        elif val == Piece.P2_PAWN:
            p2_pieces += 1
            score -= P2_PAWN_ROW_SCORES[r] + (center * 2.0)
            p2_sqs.append(sq)
        elif val == Piece.P2_KING:
            p2_pieces += 1
            p2_kings += 1
            score -= KING_VALUE + (center * 4.0)
            p2_sqs.append(sq)

    # King endgame hunting: when ahead with a king, reward closing distance to enemy pieces
    if p1_kings > 0 and p1_pieces > p2_pieces and p2_sqs:
        min_dist = min(
            (abs(SQ_TO_COORD[k][0] - SQ_TO_COORD[e][0]) + abs(SQ_TO_COORD[k][1] - SQ_TO_COORD[e][1]))
            for k in p1_sqs if board.squares[k] == Piece.P1_KING
            for e in p2_sqs
        )
        score += (14 - min_dist) * 4.0
    elif p2_kings > 0 and p2_pieces > p1_pieces and p1_sqs:
        min_dist = min(
            (abs(SQ_TO_COORD[k][0] - SQ_TO_COORD[e][0]) + abs(SQ_TO_COORD[k][1] - SQ_TO_COORD[e][1]))
            for k in p2_sqs if board.squares[k] == Piece.P2_KING
            for e in p1_sqs
        )
        score -= (14 - min_dist) * 4.0

    # Material difference amplification
    score += (p1_pieces - p2_pieces) * 20.0

    return score


def order_moves(board: Board, moves: List[Move]) -> List[Move]:
    """Orders moves to maximize alpha-beta pruning efficiency."""
    def move_priority(m: Move) -> int:
        p = 0
        if m.is_capture:
            p += 1000
            if m.captured_sq is not None and Piece.is_king(board.squares[m.captured_sq]):
                p += 2000
        if m.promoted:
            p += 1500
        to_r, _ = SQ_TO_COORD[m.to_sq]
        if board.current_player == Player.P1 and to_r == 6:
            p += 400
        elif board.current_player == Player.P2 and to_r == 1:
            p += 400
        p += CENTER_WEIGHTS[m.to_sq] * 10
        return p

    return sorted(moves, key=move_priority, reverse=True)


class MinimaxAgent:
    def __init__(self, depth: int = 5):
        self.depth = depth

    def get_dynamic_depth(self, board: Board) -> int:
        total_pieces = sum(1 for sq in board.squares if sq != Piece.EMPTY)
        if total_pieces <= 6:
            return max(self.depth + 2, 7)
        elif total_pieces <= 10:
            return max(self.depth + 1, 6)
        return self.depth

    def select_move(self, board: Board) -> Move:
        """Selects best move using Grandmaster Alpha-Beta Minimax."""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")
        if len(legal_moves) == 1:
            return legal_moves[0]

        effective_depth = self.get_dynamic_depth(board)
        is_maximizing = (board.current_player == Player.P1)
        ordered_moves = order_moves(board, legal_moves)
        best_move = ordered_moves[0]

        if is_maximizing:
            best_val = -float("inf")
            for move in ordered_moves:
                clone_b = board.clone()
                clone_b.step(move)
                val = self._minimax(clone_b, effective_depth - 1, -float("inf"), float("inf"))
                if val > best_val:
                    best_val = val
                    best_move = move
        else:
            best_val = float("inf")
            for move in ordered_moves:
                clone_b = board.clone()
                clone_b.step(move)
                val = self._minimax(clone_b, effective_depth - 1, -float("inf"), float("inf"))
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

        ordered_moves = order_moves(board, legal_moves)

        if board.current_player == Player.P1:
            max_eval = -float("inf")
            for move in ordered_moves:
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
            for move in ordered_moves:
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
