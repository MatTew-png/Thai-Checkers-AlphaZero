import os
import argparse
import re
from typing import Optional, Tuple

from env.thai_checkers import (
    Board,
    Player,
    Piece,
    SQ_TO_COORD,
    COORD_TO_SQ,
    Move,
    sq_to_algebraic,
    algebraic_to_sq,
)
from models.net import ThaiCheckersNet, get_device
from mcts.mcts import MCTS, MCTSConfig
from baseline.minimax import MinimaxAgent


def parse_move_input(input_str: str) -> Optional[Tuple[int, int]]:
    """Parses user input like 'B2 A3', 'b2-a3', 'b2xa4', '2 0 3 1', or '4 8' into (from_sq, to_sq)."""
    cleaned = re.sub(r"[-xX➔>]+", " ", input_str.strip())
    parts = cleaned.split()

    if len(parts) == 2:
        # Check if entered as algebraic notation e.g. "B2", "A3"
        try:
            return algebraic_to_sq(parts[0]), algebraic_to_sq(parts[1])
        except (ValueError, KeyError):
            pass

        # Check if entered as square indices 0..31: from_sq to_sq
        try:
            f_sq, t_sq = int(parts[0]), int(parts[1])
            if 0 <= f_sq < 32 and 0 <= t_sq < 32:
                return f_sq, t_sq
        except ValueError:
            pass

    elif len(parts) == 4:
        # User entered coordinates (r1, c1, r2, c2)
        try:
            r1, c1, r2, c2 = map(int, parts)
            if (r1, c1) in COORD_TO_SQ and (r2, c2) in COORD_TO_SQ:
                return COORD_TO_SQ[(r1, c1)], COORD_TO_SQ[(r2, c2)]
        except ValueError:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Play Thai Checkers against AlphaZero in terminal.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to weights")
    parser.add_argument("--sims", type=int, default=150, help="MCTS rollouts per move")
    parser.add_argument("--side", type=str, choices=["1", "2"], default="1", help="Play as 1 (White, first) or 2 (Black, second)")
    args = parser.parse_args()

    human_player = Player.P1 if args.side == "1" else Player.P2
    device = get_device()

    model = ThaiCheckersNet()
    if os.path.exists(args.model_path):
        print(f"📦 Loading AlphaZero weights from {args.model_path}...")
        model.load_checkpoint(args.model_path, device=device)
    else:
        print("ℹ️ Checkpoint not found. Using untrained neural network weights.")

    model.to(device)
    mcts = MCTS(model, MCTSConfig(num_simulations=args.sims))

    board = Board()
    board.setup_initial_position()

    print("\n" + "=" * 50)
    print("   🇹🇭 ยินดีต้อนรับสู่ THAI-CHECKERS-ALPHAZERO CLI 🇹🇭   ")
    print("=" * 50)
    print("กติกาสำคัญ: เบี้ยเดิน/กินหน้าเท่านั้น | บังคับกิน | เข้าฮอสหยุดทันที | ฮอสบินยาว")
    print("วิธีป้อนตาเดิน: 'r1 c1 r2 c2' เช่น '1 1 2 0' หรือพิมพ์ 'sq1 sq2' เช่น '4 8'\n")

    while True:
        print("\n" + board.render_ascii())

        done, winner = board.check_game_over()
        if done:
            print("\n" + "=" * 50)
            if winner == 0:
                print("🤝 เสมอกัน! (Draw Game)")
            elif winner == human_player:
                print("🎉 ยินดีด้วย! คุณเอาชนะ AI ได้สำเร็จ! (HUMAN WINS)")
            else:
                print("🤖 AlphaZero ชนะเกมนี้! (AI WINS)")
            print("=" * 50)
            break

        legal_moves = board.get_legal_moves()
        is_human_turn = (board.current_player == human_player)

        # AI Board Evaluation
        canonical = board.get_canonical_form()
        _, val = model.predict(canonical.get_tensor_representation())
        ai_eval_score = val if board.current_player != human_player else -val
        eval_bar = f"AI Evaluation: {ai_eval_score:+.2f} ({'AI Advantage' if ai_eval_score > 0 else 'Human Advantage'})"

        if is_human_turn:
            print(f"\n👉 ตาของคุณ ({'Player 1 ○' if human_player == 1 else 'Player 2 ●'}) | {eval_bar}")
            if legal_moves and legal_moves[0].is_capture:
                print("⚡ [กฎบังคับกิน] คุณต้องเลือกตาเดินที่กินหมากเท่านั้น!")

            print("ตาเดินที่ถูกกติกา (เลือกหมายเลข หรือพิมพ์เช่น 'B2 A3' หรือ 'B2-A3'):")
            for idx, m in enumerate(legal_moves):
                cap_str = f" [กิน {sq_to_algebraic(m.captured_sq)}]" if m.is_capture else ""
                prom_str = " [เข้าฮอส 👑]" if m.promoted else ""
                print(f"  [{idx + 1}] {m.notation:<7} ({sq_to_algebraic(m.from_sq)} -> {sq_to_algebraic(m.to_sq)}) [Sq {m.from_sq}->{m.to_sq}]{cap_str}{prom_str}")

            while True:
                user_input = input("\nกรอกตาเดิน (เช่น 'B2 A3', '1-N', 'q' ออก): ").strip()
                if user_input.lower() == 'q':
                    print("ออกจากเกมเรียบร้อย")
                    return

                # If user typed move number
                if user_input.isdigit() and 1 <= int(user_input) <= len(legal_moves):
                    chosen_move = legal_moves[int(user_input) - 1]
                    break

                coords = parse_move_input(user_input)
                if coords:
                    f_sq, t_sq = coords
                    matches = [m for m in legal_moves if m.from_sq == f_sq and m.to_sq == t_sq]
                    if matches:
                        chosen_move = matches[0]
                        break

                print("❌ ตาเดินไม่ถูกต้อง หรือผิดกติกาบังคับกิน กรุณาลองใหม่อีกครั้ง (เช่น B2 A3 หรือพิมพ์หมายเลข)")

            board.step(chosen_move)

        else:
            print(f"\n🤖 AlphaZero กำลังคิดด้วย MCTS ({args.sims} rollouts)...")
            best_action = mcts.select_best_move(board, num_simulations=args.sims, temperature=0.0)
            from_sq, to_sq = Move.from_action_id(best_action)
            best_m = next((m for m in legal_moves if m.action_id == best_action), None)
            notation_str = best_m.notation if best_m else f"{sq_to_algebraic(from_sq)}-{sq_to_algebraic(to_sq)}"
            print(f"🤖 AI เดิน: {notation_str} ({sq_to_algebraic(from_sq)} ➔ {sq_to_algebraic(to_sq)}) [Sq {from_sq}->{to_sq}]")
            board.step(best_action)


if __name__ == "__main__":
    main()
