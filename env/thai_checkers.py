"""Thai Checkers (หมากฮอสไทย) Engine.

Exact 100% Thai Checkers Rules:
1. 8x8 Board, 32 playable dark squares where (row + col) % 2 == 0.
   - Top-left corner (7, 0) is light (non-playable), so (0, 0) is dark (playable).
   - 8 pieces per player at start:
     - Player 1 (Red / White): rows 0 and 1 (squares 0..7). Moves UP (row increases).
     - Player 2 (Black / Blue): rows 6 and 7 (squares 24..31). Moves DOWN (row decreases).
2. Forced Capture (กฎบังคับกิน):
   - If any capture is available, the player MUST capture.
   - If multiple captures are possible, player may choose any valid capture path.
3. Multi-Jump (การกินต่อเนื่อง):
   - If a piece captures and has another capture available from the landing square,
     it must continue capturing until no further captures can be made.
4. Promotion (เข้าฮอส):
   - When a pawn reaches the opponent's back rank (row 7 for P1, row 0 for P2),
     it promotes to King immediately and the turn ENDS IMMEDIATELY (cannot continue jumping).
5. Flying King (ฮอสบิน):
   - Can move diagonally in all 4 directions across any number of empty squares.
   - Can jump over an enemy piece along diagonal line of sight and land on ANY empty
     square beyond it along that ray.
   - If another capture is available from that landing square, King must continue capturing.
6. Action Space:
   - 1024 discrete actions: action_id = from_sq * 32 + to_sq (0 <= from_sq, to_sq < 32).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import numpy as np


class Piece:
    EMPTY = 0
    P1_PAWN = 1
    P1_KING = 2
    P2_PAWN = -1
    P2_KING = -2

    @staticmethod
    def is_p1(p: int) -> bool:
        return p > 0

    @staticmethod
    def is_p2(p: int) -> bool:
        return p < 0

    @staticmethod
    def is_king(p: int) -> bool:
        return abs(p) == 2

    @staticmethod
    def is_pawn(p: int) -> bool:
        return abs(p) == 1

    @staticmethod
    def owner(p: int) -> int:
        if p > 0:
            return 1
        if p < 0:
            return -1
        return 0


class Player:
    P1 = 1
    P2 = -1


NUM_SQUARES = 32
ACTION_SPACE_SIZE = 1024  # 32 * 32

# Map 32 playable squares (dark squares where (row + col) % 2 == 0)
SQ_TO_COORD: List[Tuple[int, int]] = []
for r in range(8):
    for c in range(8):
        if (r + c) % 2 == 0:
            SQ_TO_COORD.append((r, c))

COORD_TO_SQ: Dict[Tuple[int, int], int] = {coord: i for i, coord in enumerate(SQ_TO_COORD)}

# Standard Algebraic Notation: Columns A-H, Rows 1-8
FILES = "ABCDEFGH"
RANKS = "12345678"


def coord_to_algebraic(r: int, c: int) -> str:
    """Converts (row, col) to standard algebraic notation e.g. (0, 0) -> 'A1'."""
    return f"{FILES[c]}{RANKS[r]}"


def algebraic_to_coord(notation: str) -> Tuple[int, int]:
    """Converts notation e.g. 'A1' or 'b2' to 0-indexed (row, col)."""
    s = notation.strip().upper()
    if len(s) != 2 or s[0] not in FILES or s[1] not in RANKS:
        raise ValueError(f"Invalid algebraic notation: {notation}")
    c = FILES.index(s[0])
    r = RANKS.index(s[1])
    return r, c


def sq_to_algebraic(sq: int) -> str:
    """Converts square index (0..31) to algebraic notation e.g. 0 -> 'A1'."""
    r, c = SQ_TO_COORD[sq]
    return coord_to_algebraic(r, c)


def algebraic_to_sq(notation: str) -> int:
    """Converts algebraic notation e.g. 'A1' to square index (0..31)."""
    r, c = algebraic_to_coord(notation)
    if (r, c) not in COORD_TO_SQ:
        raise ValueError(f"Square {notation} is a light square (non-playable in Thai Checkers).")
    return COORD_TO_SQ[(r, c)]

# Precomputed ray directions for diagonal steps: (dr, dc)
DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# Precompute rays and 1-step neighbors for each playable square
SQ_RAYS: Dict[int, Dict[Tuple[int, int], List[int]]] = {}
SQ_NEIGHBORS: Dict[int, Dict[Tuple[int, int], Optional[int]]] = {}

for sq, (r, c) in enumerate(SQ_TO_COORD):
    SQ_RAYS[sq] = {}
    SQ_NEIGHBORS[sq] = {}
    for dr, dc in DIRECTIONS:
        ray: List[int] = []
        step = 1
        while True:
            nr, nc = r + step * dr, c + step * dc
            if 0 <= nr < 8 and 0 <= nc < 8 and (nr, nc) in COORD_TO_SQ:
                ray.append(COORD_TO_SQ[(nr, nc)])
                step += 1
            else:
                break
        SQ_RAYS[sq][(dr, dc)] = ray
        SQ_NEIGHBORS[sq][(dr, dc)] = ray[0] if len(ray) > 0 else None


@dataclass(frozen=True)
class Move:
    from_sq: int
    to_sq: int
    captured_sq: Optional[int] = None
    is_capture: bool = False
    promoted: bool = False

    @property
    def action_id(self) -> int:
        return self.from_sq * 32 + self.to_sq

    @property
    def notation(self) -> str:
        sep = "x" if self.is_capture else "-"
        return f"{sq_to_algebraic(self.from_sq)}{sep}{sq_to_algebraic(self.to_sq)}"

    @staticmethod
    def from_action_id(action_id: int) -> Tuple[int, int]:
        return action_id // 32, action_id % 32


def canonicalize_action(action_id: int, current_player: int) -> int:
    """Converts a real board action ID to canonical action ID (180-degree rotation for P2)."""
    if current_player == Player.P1:
        return action_id
    from_sq = action_id // 32
    to_sq = action_id % 32
    return (31 - from_sq) * 32 + (31 - to_sq)


def decanonicalize_action(canonical_action_id: int, current_player: int) -> int:
    """Converts a canonical action ID back to real board action ID."""
    return canonicalize_action(canonical_action_id, current_player)


class Board:
    def __init__(self):
        # 32 playable squares
        self.squares: List[int] = [Piece.EMPTY] * NUM_SQUARES
        self.current_player: int = Player.P1
        # If in middle of a multi-jump, continuation_sq holds the piece square that MUST continue
        self.continuation_sq: Optional[int] = None
        self.halfmove_clock: int = 0
        self.ply_count: int = 0
        self.history: List[int] = []  # State hashes for 3-fold repetition check
        self.setup_initial_position()

    def setup_initial_position(self) -> None:
        """Sets up the standard Thai Checkers starting position (8 vs 8)."""
        self.squares = [Piece.EMPTY] * NUM_SQUARES
        # Player 1 (Red / White): rows 0 and 1 (squares 0..7)
        for sq in range(8):
            self.squares[sq] = Piece.P1_PAWN
        # Player 2 (Black / Blue): rows 6 and 7 (squares 24..31)
        for sq in range(24, 32):
            self.squares[sq] = Piece.P2_PAWN

        self.current_player = Player.P1
        self.continuation_sq = None
        self.halfmove_clock = 0
        self.ply_count = 0
        self.history = [self.position_hash()]

    def clone(self) -> Board:
        b = Board.__new__(Board)
        b.squares = list(self.squares)
        b.current_player = self.current_player
        b.continuation_sq = self.continuation_sq
        b.halfmove_clock = self.halfmove_clock
        b.ply_count = self.ply_count
        b.history = list(self.history)
        return b

    def position_hash(self) -> int:
        """Hash representing board layout and active player for repetition checking."""
        return hash((tuple(self.squares), self.current_player, self.continuation_sq))

    def get_legal_moves(self) -> List[Move]:
        """Returns all legal moves for the current player, enforcing Thai Checkers rules.
        
        Rules:
        1. If continuation_sq is set (mid-multijump), only captures from continuation_sq are considered.
        2. Forced Capture: If ANY capture move exists, ONLY capture moves are legal.
        """
        captures: List[Move] = []
        quiet_moves: List[Move] = []

        if self.continuation_sq is not None:
            # Must continue jump with this specific piece
            c_moves = self._get_captures_for_square(self.continuation_sq)
            return c_moves

        # Search captures and quiet moves for all pieces belonging to current_player
        for sq in range(NUM_SQUARES):
            piece = self.squares[sq]
            if piece == Piece.EMPTY or Piece.owner(piece) != self.current_player:
                continue

            # Check captures
            sq_captures = self._get_captures_for_square(sq)
            captures.extend(sq_captures)

            # Check quiet moves only if we haven't found any captures yet
            # (If captures exist anywhere on board, quiet moves will be discarded anyway)
            if not captures:
                sq_quiets = self._get_quiet_moves_for_square(sq)
                quiet_moves.extend(sq_quiets)

        # Forced Capture Rule: If any capture is available, must capture!
        if captures:
            return captures
        return quiet_moves

    def _get_quiet_moves_for_square(self, sq: int) -> List[Move]:
        """Returns legal non-capturing moves for a piece at sq."""
        moves: List[Move] = []
        piece = self.squares[sq]
        r, c = SQ_TO_COORD[sq]

        if Piece.is_pawn(piece):
            # Pawn moves forward diagonally 1 step
            # P1 moves UP (dr = +1), P2 moves DOWN (dr = -1)
            forward_dr = 1 if self.current_player == Player.P1 else -1
            for dc in (-1, 1):
                target_sq = SQ_NEIGHBORS[sq].get((forward_dr, dc))
                if target_sq is not None and self.squares[target_sq] == Piece.EMPTY:
                    promoted = self._will_promote(piece, target_sq)
                    moves.append(Move(from_sq=sq, to_sq=target_sq, promoted=promoted))

        elif Piece.is_king(piece):
            # Flying King moves in all 4 diagonal directions any number of empty squares
            for direction, ray in SQ_RAYS[sq].items():
                for target_sq in ray:
                    if self.squares[target_sq] == Piece.EMPTY:
                        moves.append(Move(from_sq=sq, to_sq=target_sq))
                    else:
                        # Blocked by another piece
                        break

        return moves

    def _get_captures_for_square(self, sq: int) -> List[Move]:
        """Returns legal capture moves originating from sq."""
        captures: List[Move] = []
        piece = self.squares[sq]
        r, c = SQ_TO_COORD[sq]

        if Piece.is_pawn(piece):
            # Pawn captures forward diagonally by jumping over adjacent enemy piece
            forward_dr = 1 if self.current_player == Player.P1 else -1
            for dc in (-1, 1):
                enemy_sq = SQ_NEIGHBORS[sq].get((forward_dr, dc))
                if enemy_sq is not None and self.squares[enemy_sq] != Piece.EMPTY:
                    if Piece.owner(self.squares[enemy_sq]) != self.current_player:
                        # Check landing square (2 steps forward in this diagonal)
                        landing_r, landing_c = r + 2 * forward_dr, c + 2 * dc
                        if 0 <= landing_r < 8 and 0 <= landing_c < 8 and (landing_r, landing_c) in COORD_TO_SQ:
                            landing_sq = COORD_TO_SQ[(landing_r, landing_c)]
                            if self.squares[landing_sq] == Piece.EMPTY:
                                promoted = self._will_promote(piece, landing_sq)
                                captures.append(
                                    Move(
                                        from_sq=sq,
                                        to_sq=landing_sq,
                                        captured_sq=enemy_sq,
                                        is_capture=True,
                                        promoted=promoted,
                                    )
                                )

        elif Piece.is_king(piece):
            # Flying King captures along any of the 4 diagonals:
            # Moves along ray until it finds an enemy piece.
            # In Thai Checkers: King MUST land on the immediate square right behind the captured enemy!
            for direction, ray in SQ_RAYS[sq].items():
                enemy_sq: Optional[int] = None
                enemy_idx: int = -1

                for idx, step_sq in enumerate(ray):
                    sq_val = self.squares[step_sq]
                    if sq_val == Piece.EMPTY:
                        continue
                    elif Piece.owner(sq_val) == self.current_player:
                        # Friendly piece blocks ray
                        break
                    else:
                        # Enemy piece encountered
                        enemy_sq = step_sq
                        enemy_idx = idx
                        break

                if enemy_sq is not None:
                    # In Thai Checkers, King must land immediately behind the captured piece (first square)
                    if enemy_idx + 1 < len(ray):
                        landing_sq = ray[enemy_idx + 1]
                        if self.squares[landing_sq] == Piece.EMPTY:
                            captures.append(
                                Move(
                                    from_sq=sq,
                                    to_sq=landing_sq,
                                    captured_sq=enemy_sq,
                                    is_capture=True,
                                    promoted=False,
                                )
                            )

        return captures

    def _will_promote(self, piece: int, to_sq: int) -> bool:
        """Checks if a piece moving to to_sq will be promoted to King."""
        if not Piece.is_pawn(piece):
            return False
        r, _ = SQ_TO_COORD[to_sq]
        if Piece.owner(piece) == Player.P1 and r == 7:
            return True
        if Piece.owner(piece) == Player.P2 and r == 0:
            return True
        return False

    def step(self, action: int | Move) -> Tuple[Board, float, bool, Dict]:
        """Executes a move/action on the board.
        
        Args:
            action: either an integer action_id (0..1023) or a Move object.
            
        Returns:
            (self, reward, done, info)
            reward is from the perspective of the player who just acted:
                +1.0 if player won
                -1.0 if player lost
                 0.0 for draw or in-progress
        """
        if isinstance(action, int):
            from_sq, to_sq = Move.from_action_id(action)
            legal_moves = self.get_legal_moves()
            matching = [m for m in legal_moves if m.from_sq == from_sq and m.to_sq == to_sq]
            if not matching:
                raise ValueError(
                    f"Illegal action {action} ({from_sq}->{to_sq}) for player {self.current_player}."
                )
            move = matching[0]
        else:
            move = action

        piece = self.squares[move.from_sq]
        acting_player = self.current_player

        # 1. Move piece
        self.squares[move.to_sq] = piece
        self.squares[move.from_sq] = Piece.EMPTY

        # 2. If capture, remove captured piece
        if move.is_capture and move.captured_sq is not None:
            self.squares[move.captured_sq] = Piece.EMPTY
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.ply_count += 1

        # 3. Check promotion:
        # Thai Checkers Rule: When promoted to King, the turn ENDS immediately!
        if move.promoted:
            # Transform to King
            self.squares[move.to_sq] = (
                Piece.P1_KING if acting_player == Player.P1 else Piece.P2_KING
            )
            # Turn ends immediately
            self.continuation_sq = None
            self.current_player = -self.current_player

        elif move.is_capture:
            # 4. Check for multi-jump continuation:
            further_captures = self._get_captures_for_square(move.to_sq)
            if further_captures:
                # Must continue jumping with this piece in the same turn!
                self.continuation_sq = move.to_sq
                # Active player does NOT switch!
            else:
                # Multi-jump sequence ended
                self.continuation_sq = None
                self.current_player = -self.current_player
        else:
            # Quiet move ends turn
            self.continuation_sq = None
            self.current_player = -self.current_player

        # Add new state hash to history
        current_hash = self.position_hash()
        self.history.append(current_hash)

        # 5. Check game over
        done, winner = self.check_game_over()
        reward = 0.0
        if done:
            if winner == acting_player:
                reward = 1.0
            elif winner == -acting_player:
                reward = -1.0
            else:
                reward = 0.0  # Draw

        return self, reward, done, {"winner": winner, "acting_player": acting_player}

    def check_game_over(self) -> Tuple[bool, Optional[int]]:
        """Determines if the game is over and returns (is_done, winner).
        
        Winner is:
            1: Player 1 won
           -1: Player 2 won
            0: Draw
         None: Game is still active
        """
        # 1. Check piece counts
        p1_pieces = sum(1 for s in self.squares if Piece.is_p1(s))
        p2_pieces = sum(1 for s in self.squares if Piece.is_p2(s))

        if p1_pieces == 0:
            return True, Player.P2
        if p2_pieces == 0:
            return True, Player.P1

        # 2. Check legal moves for current player
        # Thai Checkers Rule: A player with no legal moves loses (stalemate = loss)
        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            # Current player cannot move -> Current player loses!
            return True, -self.current_player

        # 3. Check 3-fold repetition
        curr_hash = self.history[-1] if self.history else self.position_hash()
        if self.history.count(curr_hash) >= 3:
            return True, 0  # Draw

        # 4. Check 50-move rule (100 plies without a capture) or maximum turn limit (160 plies)
        if self.halfmove_clock >= 100 or self.ply_count >= 200:
            return True, 0  # Draw

        # 5. Endgame 1 King vs 1 King draw check
        if p1_pieces == 1 and p2_pieces == 1:
            p1_kings = sum(1 for s in self.squares if s == Piece.P1_KING)
            p2_kings = sum(1 for s in self.squares if s == Piece.P2_KING)
            if p1_kings == 1 and p2_kings == 1 and self.halfmove_clock >= 32:
                return True, 0  # Thai counting draw

        return False, None

    def get_game_over_reason(self) -> str:
        """Returns a clear, human-readable Thai explanation of why the game ended."""
        p1_pieces = sum(1 for s in self.squares if Piece.is_p1(s))
        p2_pieces = sum(1 for s in self.squares if Piece.is_p2(s))

        if p1_pieces == 0:
            return "กินหมากขาวหมดกระดาน"
        if p2_pieces == 0:
            return "กินหมากดำหมดกระดาน"

        legal_moves = self.get_legal_moves()
        if len(legal_moves) == 0:
            trapped_side = "⚪ ขาว" if self.current_player == Player.P1 else "⚫ ดำ"
            return f"{trapped_side}ไม่มีตาเดิน (โดนขังหมากจนมุม)"

        curr_hash = self.history[-1] if self.history else self.position_hash()
        if self.history.count(curr_hash) >= 3:
            return "เสมอกัน (หมากซ้ำตำแหน่งเดิม 3 ครั้ง)"

        if self.halfmove_clock >= 100:
            return "เสมอกัน (กฎ 50 ตา ไม่มีการกิน)"
        if self.ply_count >= 200:
            return "เสมอกัน (ครบเพดาน 200 ตา)"

        if p1_pieces == 1 and p2_pieces == 1:
            p1_kings = sum(1 for s in self.squares if s == Piece.P1_KING)
            p2_kings = sum(1 for s in self.squares if s == Piece.P2_KING)
            if p1_kings == 1 and p2_kings == 1 and self.halfmove_clock >= 32:
                return "เสมอกัน (ฮอสเดี่ยวดวลฮอสเดี่ยวครบก้าว)"

        return "จบเกม"

    def get_legal_action_mask(self) -> np.ndarray:
        """Returns a boolean array of size 1024 indicating legal action indices."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for move in self.get_legal_moves():
            mask[move.action_id] = True
        return mask

    def get_canonical_legal_action_mask(self) -> np.ndarray:
        """Returns a boolean array of size 1024 indicating legal actions in canonical coordinates."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for move in self.get_legal_moves():
            can_act = canonicalize_action(move.action_id, self.current_player)
            mask[can_act] = True
        return mask

    def get_canonical_form(self) -> Board:
        """Returns the board from the perspective of the current player.
        
        For Player 1: returns a clone unchanged.
        For Player 2: rotates board 180 degrees (mapping (r, c) -> (7-r, 7-c)),
                      inverts piece colors (* -1), and maps continuation_sq.
        This allows the neural network to always play as Player 1 (+1).
        """
        if self.current_player == Player.P1:
            return self.clone()

        canonical = Board.__new__(Board)
        canonical.squares = [Piece.EMPTY] * NUM_SQUARES

        # 180-degree rotation maps dark square (r, c) to (7-r, 7-c)
        for sq in range(NUM_SQUARES):
            piece = self.squares[sq]
            if piece != Piece.EMPTY:
                r, c = SQ_TO_COORD[sq]
                rot_r, rot_c = 7 - r, 7 - c
                rot_sq = COORD_TO_SQ[(rot_r, rot_c)]
                # Invert owner: P2 pieces become positive (+1, +2), P1 pieces negative (-1, -2)
                canonical.squares[rot_sq] = -piece

        canonical.current_player = Player.P1
        if self.continuation_sq is not None:
            r, c = SQ_TO_COORD[self.continuation_sq]
            canonical.continuation_sq = COORD_TO_SQ[(7 - r, 7 - c)]
        else:
            canonical.continuation_sq = None

        canonical.halfmove_clock = self.halfmove_clock
        canonical.ply_count = self.ply_count
        canonical.history = []
        return canonical

    def get_tensor_representation(self) -> np.ndarray:
        """Converts the canonical board into a (6, 8, 8) float32 numpy tensor.
        
        Planes:
        0: My pawns (+1)
        1: My kings (+2)
        2: Opponent pawns (-1)
        3: Opponent kings (-2)
        4: Continuation square mask (1.0 at continuation_sq if mid-jump, else 0)
        5: Game progress plane (ply_count / 100.0)
        """
        tensor = np.zeros((6, 8, 8), dtype=np.float32)

        for sq in range(NUM_SQUARES):
            piece = self.squares[sq]
            r, c = SQ_TO_COORD[sq]
            if piece == Piece.P1_PAWN:
                tensor[0, r, c] = 1.0
            elif piece == Piece.P1_KING:
                tensor[1, r, c] = 1.0
            elif piece == Piece.P2_PAWN:
                tensor[2, r, c] = 1.0
            elif piece == Piece.P2_KING:
                tensor[3, r, c] = 1.0

        if self.continuation_sq is not None:
            cr, cc = SQ_TO_COORD[self.continuation_sq]
            tensor[4, cr, cc] = 1.0

        tensor[5, :, :] = min(1.0, self.ply_count / 100.0)
        return tensor

    def render_ascii(self) -> str:
        """Renders an elegant ASCII board with standard algebraic notation (A-H, 1-8)."""
        symbol_map = {
            Piece.EMPTY: " . ",
            Piece.P1_PAWN: " ○ ",  # White / P1 Pawn
            Piece.P1_KING: " ◎ ",  # White / P1 King
            Piece.P2_PAWN: " ● ",  # Black / P2 Pawn
            Piece.P2_KING: " ◉ ",  # Black / P2 King
        }
        lines = []
        lines.append("   +---+---+---+---+---+---+---+---+")
        for r in range(7, -1, -1):
            rank_num = r + 1
            row_str = f" {rank_num} |"
            for c in range(8):
                if (r + c) % 2 == 0:
                    sq = COORD_TO_SQ[(r, c)]
                    row_str += symbol_map[self.squares[sq]] + "|"
                else:
                    row_str += "   |"  # Light square (unusable)
            lines.append(row_str)
            lines.append("   +---+---+---+---+---+---+---+---+")
        lines.append("     A   B   C   D   E   F   G   H  ")
        lines.append(
            f"Active: {'Player 1 (○/◎)' if self.current_player == 1 else 'Player 2 (●/◉)'} "
            f"| Ply: {self.ply_count} | Halfmove: {self.halfmove_clock}"
        )
        if self.continuation_sq is not None:
            r, c = SQ_TO_COORD[self.continuation_sq]
            alg = coord_to_algebraic(r, c)
            lines.append(f"⚡ Forced Multi-jump continuation from {alg} [Sq {self.continuation_sq}]")
        return "\n".join(lines)


class ThaiCheckersEnv:
    """Gym-like environment wrapper for Thai Checkers AlphaZero."""

    def __init__(self):
        self.board = Board()

    def reset(self) -> np.ndarray:
        self.board.setup_initial_position()
        return self.board.get_canonical_form().get_tensor_representation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        _, reward, done, info = self.board.step(action)
        canonical_state = self.board.get_canonical_form().get_tensor_representation()
        return canonical_state, reward, done, info

    def get_legal_action_mask(self) -> np.ndarray:
        return self.board.get_legal_action_mask()

    def get_legal_moves(self) -> List[Move]:
        return self.board.get_legal_moves()

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        return self.board.check_game_over()

    def clone(self) -> ThaiCheckersEnv:
        env = ThaiCheckersEnv.__new__(ThaiCheckersEnv)
        env.board = self.board.clone()
        return env
