"""Comprehensive unit tests for 100% authentic Thai Checkers rules."""

import pytest
import numpy as np
from env.thai_checkers import (
    Board,
    Piece,
    Player,
    Move,
    SQ_TO_COORD,
    COORD_TO_SQ,
    NUM_SQUARES,
)


def test_initial_board_setup():
    """Verify standard Thai Checkers setup: 8 vs 8 on 32 dark squares."""
    board = Board()
    board.setup_initial_position()

    # Check 32 squares
    assert len(board.squares) == 32

    # Check player 1 pawns (rows 0 and 1, squares 0..7)
    for sq in range(8):
        assert board.squares[sq] == Piece.P1_PAWN

    # Check empty middle squares (rows 2, 3, 4, 5, squares 8..23)
    for sq in range(8, 24):
        assert board.squares[sq] == Piece.EMPTY

    # Check player 2 pawns (rows 6 and 7, squares 24..31)
    for sq in range(24, 32):
        assert board.squares[sq] == Piece.P2_PAWN

    assert board.current_player == Player.P1
    assert board.continuation_sq is None

    # Starting legal moves for P1: only row 1 pawns (squares 4, 5, 6, 7) can move
    moves = board.get_legal_moves()
    assert len(moves) > 0
    for m in moves:
        r1, _ = SQ_TO_COORD[m.from_sq]
        r2, _ = SQ_TO_COORD[m.to_sq]
        assert r1 == 1
        assert r2 == 2
        assert not m.is_capture


def test_pawn_forward_only():
    """Thai Checkers rule: Pawns move only forward, never backward."""
    board = Board()
    # Place a single P1 pawn at row 3, col 3 (square 13)
    board.squares = [Piece.EMPTY] * NUM_SQUARES
    sq = COORD_TO_SQ[(3, 3)]
    board.squares[sq] = Piece.P1_PAWN
    board.current_player = Player.P1

    moves = board.get_legal_moves()
    # Can move to (4, 2) and (4, 4)
    destinations = [SQ_TO_COORD[m.to_sq] for m in moves]
    assert (4, 2) in destinations
    assert (4, 4) in destinations
    # Cannot move backward to (2, 2) or (2, 4)
    assert (2, 2) not in destinations
    assert (2, 4) not in destinations


def test_forced_capture_rule():
    """Thai Checkers rule: If ANY capture is available, ONLY captures are legal."""
    board = Board()
    board.squares = [Piece.EMPTY] * NUM_SQUARES

    # P1 pawn at (2, 2)
    p1_sq = COORD_TO_SQ[(2, 2)]
    board.squares[p1_sq] = Piece.P1_PAWN

    # P2 pawn at (3, 3) (can be captured by jumping to (4, 4))
    p2_sq = COORD_TO_SQ[(3, 3)]
    board.squares[p2_sq] = Piece.P2_PAWN

    # Another P1 pawn at (0, 0) with quiet moves available
    p1_free_sq = COORD_TO_SQ[(0, 0)]
    board.squares[p1_free_sq] = Piece.P1_PAWN

    board.current_player = Player.P1
    legal_moves = board.get_legal_moves()

    # Must only contain capture moves!
    assert len(legal_moves) == 1
    assert legal_moves[0].is_capture
    assert legal_moves[0].from_sq == p1_sq
    assert legal_moves[0].to_sq == COORD_TO_SQ[(4, 4)]
    assert legal_moves[0].captured_sq == p2_sq


def test_pawn_multi_jump():
    """Thai Checkers rule: If pawn captures and can capture again forward, must continue."""
    board = Board()
    board.squares = [Piece.EMPTY] * NUM_SQUARES

    # P1 pawn at (0, 0)
    p1_sq = COORD_TO_SQ[(0, 0)]
    board.squares[p1_sq] = Piece.P1_PAWN

    # First enemy at (1, 1)
    e1_sq = COORD_TO_SQ[(1, 1)]
    board.squares[e1_sq] = Piece.P2_PAWN

    # Second enemy at (3, 3)
    e2_sq = COORD_TO_SQ[(3, 3)]
    board.squares[e2_sq] = Piece.P2_PAWN

    board.current_player = Player.P1

    # First jump: from (0, 0) to (2, 2)
    first_moves = board.get_legal_moves()
    assert len(first_moves) == 1
    board.step(first_moves[0])

    # Enemy 1 removed
    assert board.squares[e1_sq] == Piece.EMPTY
    assert board.squares[COORD_TO_SQ[(2, 2)]] == Piece.P1_PAWN

    # Turn should NOT have switched to P2 because a second capture is available!
    assert board.current_player == Player.P1
    assert board.continuation_sq == COORD_TO_SQ[(2, 2)]

    # Second jump MUST be from (2, 2) to (4, 4)
    second_moves = board.get_legal_moves()
    assert len(second_moves) == 1
    assert second_moves[0].to_sq == COORD_TO_SQ[(4, 4)]
    board.step(second_moves[0])

    # After second jump, no more captures -> turn switches to P2
    assert board.squares[e2_sq] == Piece.EMPTY
    assert board.continuation_sq is None
    assert board.current_player == Player.P2


def test_pawn_promotion_ends_turn():
    """Thai Checkers rule: When reaching back rank, pawn promotes and turn ENDS immediately."""
    board = Board()
    board.squares = [Piece.EMPTY] * NUM_SQUARES

    # P1 pawn at (5, 1)
    p1_sq = COORD_TO_SQ[(5, 1)]
    board.squares[p1_sq] = Piece.P1_PAWN

    # P2 pawn at (6, 2)
    e1_sq = COORD_TO_SQ[(6, 2)]
    board.squares[e1_sq] = Piece.P2_PAWN

    board.current_player = Player.P1
    legal_moves = board.get_legal_moves()
    assert len(legal_moves) == 1
    move = legal_moves[0]

    # Jumps to (7, 3) which is row 7 (promotion rank)
    assert move.to_sq == COORD_TO_SQ[(7, 3)]
    assert move.promoted

    board.step(move)

    # Pawn promoted to P1_KING
    assert board.squares[COORD_TO_SQ[(7, 3)]] == Piece.P1_KING
    # Turn ends immediately!
    assert board.continuation_sq is None
    assert board.current_player == Player.P2


def test_flying_king_movement():
    """Thai Checkers rule: Flying King can move any number of empty squares along diagonal."""
    board = Board()
    board.squares = [Piece.EMPTY] * NUM_SQUARES

    # P1 King at center (3, 3)
    k_sq = COORD_TO_SQ[(3, 3)]
    board.squares[k_sq] = Piece.P1_KING
    board.current_player = Player.P1

    moves = board.get_legal_moves()
    destinations = [SQ_TO_COORD[m.to_sq] for m in moves]

    # Ray (+1, +1): (4, 4), (5, 5), (6, 6), (7, 7)
    for coord in [(4, 4), (5, 5), (6, 6), (7, 7)]:
        assert coord in destinations

    # Ray (-1, -1): (2, 2), (1, 1), (0, 0)
    for coord in [(2, 2), (1, 1), (0, 0)]:
        assert coord in destinations


def test_flying_king_capture_any_empty_landing():
    """Thai Checkers rule: King can land on ANY empty square beyond captured enemy along diagonal ray."""
    board = Board()
    board.squares = [Piece.EMPTY] * NUM_SQUARES

    # King at (1, 1)
    k_sq = COORD_TO_SQ[(1, 1)]
    board.squares[k_sq] = Piece.P1_KING

    # Enemy at (3, 3)
    e_sq = COORD_TO_SQ[(3, 3)]
    board.squares[e_sq] = Piece.P2_PAWN

    board.current_player = Player.P1
    legal_moves = board.get_legal_moves()

    # King can jump enemy at (3, 3) and land on (4, 4), (5, 5), (6, 6), or (7, 7)
    destinations = [SQ_TO_COORD[m.to_sq] for m in legal_moves]
    assert len(destinations) == 4
    assert (4, 4) in destinations
    assert (5, 5) in destinations
    assert (6, 6) in destinations
    assert (7, 7) in destinations


def test_canonical_form_rotation():
    """Verify that canonical board rotation accurately flips P2 perspective to P1."""
    board = Board()
    board.setup_initial_position()
    board.current_player = Player.P2

    canonical = board.get_canonical_form()
    assert canonical.current_player == Player.P1

    # In canonical form, P2 pawns become +1 pawns on rows 0 and 1
    for sq in range(8):
        assert canonical.squares[sq] == Piece.P1_PAWN

    # P1 pawns become -1 pawns on rows 6 and 7
    for sq in range(24, 32):
        assert canonical.squares[sq] == Piece.P2_PAWN


def test_algebraic_notation():
    """Verify algebraic notation conversions (A-H, 1-8)."""
    from env.thai_checkers import (
        coord_to_algebraic,
        algebraic_to_coord,
        sq_to_algebraic,
        algebraic_to_sq,
    )

    # (0, 0) is A1
    assert coord_to_algebraic(0, 0) == "A1"
    assert algebraic_to_coord("A1") == (0, 0)
    assert algebraic_to_coord("a1") == (0, 0)

    # (1, 1) is B2
    assert coord_to_algebraic(1, 1) == "B2"
    assert algebraic_to_coord("b2") == (1, 1)

    # (7, 7) is H8
    assert coord_to_algebraic(7, 7) == "H8"
    assert algebraic_to_coord("H8") == (7, 7)

    # Square indices: sq 0 is (0, 0) -> A1
    assert sq_to_algebraic(0) == "A1"
    assert algebraic_to_sq("A1") == 0

    # Move notation
    m1 = Move(from_sq=algebraic_to_sq("B2"), to_sq=algebraic_to_sq("A3"))
    assert m1.notation == "B2-A3"

    m2 = Move(from_sq=algebraic_to_sq("B2"), to_sq=algebraic_to_sq("D4"), is_capture=True)
    assert m2.notation == "B2xD4"
