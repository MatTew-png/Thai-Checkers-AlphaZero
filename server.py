import os
import threading
import asyncio
from typing import Optional, Dict, Any, List
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.thai_checkers import (
    Board,
    Player,
    Piece,
    Move,
    SQ_TO_COORD,
    COORD_TO_SQ,
    NUM_SQUARES,
    sq_to_algebraic,
)
from models.net import ThaiCheckersNet, get_device
from mcts.mcts import MCTS, MCTSConfig
from baseline.minimax import MinimaxAgent
from training.trainer import Trainer, TrainerConfig

app = FastAPI(title="Thai Checkers AlphaZero API")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

    def threadsafe_broadcast(self, data: dict):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_json(data), self.loop)

ws_manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    ws_manager.loop = asyncio.get_running_loop()
    print("✅ Startup event ran: loop set to", ws_manager.loop)

# Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui", "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global game state & AI model
device = get_device()
model = ThaiCheckersNet()
checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
if os.path.exists(checkpoint_path):
    print(f"📦 Loaded model from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path, device=device)
model.to(device)

game_board = Board()
game_board.setup_initial_position()

class TrainStartRequest(BaseModel):
    num_iters: int = 5
    episodes_per_iter: int = 5
    mcts_sims: int = 50
    epochs_per_iter: int = 4
    batch_size: int = 64
    arena_games: int = 4
    visual_delay: float = 0.05
    resume: bool = True

class TrainingState:
    def __init__(self):
        self.is_training: bool = False
        self.trainer: Optional[Trainer] = None
        self.thread: Optional[threading.Thread] = None
        self.current_iter: int = 0
        self.total_iters: int = 0
        self.current_episode: int = 0
        self.total_episodes: int = 0
        self.buffer_size: int = 0
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "win_rate": [],
        }

training_state = TrainingState()


class HumanMoveRequest(BaseModel):
    from_sq: int
    to_sq: int


class AIMoveRequest(BaseModel):
    simulations: int = 150
    engine: str = "alphazero"  # "alphazero" or "minimax"
    temperature: float = 0.0


def serialize_board_state(board: Board) -> Dict[str, Any]:
    """Serializes the board for frontend consumption."""
    legal_moves = board.get_legal_moves()
    serialized_moves = []
    for m in legal_moves:
        r1, c1 = SQ_TO_COORD[m.from_sq]
        r2, c2 = SQ_TO_COORD[m.to_sq]
        serialized_moves.append({
            "from_sq": m.from_sq,
            "to_sq": m.to_sq,
            "from_coord": [r1, c1],
            "to_coord": [r2, c2],
            "from_algebraic": sq_to_algebraic(m.from_sq),
            "to_algebraic": sq_to_algebraic(m.to_sq),
            "notation": m.notation,
            "is_capture": m.is_capture,
            "captured_sq": m.captured_sq,
            "promoted": m.promoted,
            "action_id": m.action_id,
        })

    done, winner = board.check_game_over()

    # Model evaluation
    canonical = board.get_canonical_form()
    _, val = model.predict(canonical.get_tensor_representation())
    eval_p1 = val if board.current_player == Player.P1 else -val

    return {
        "squares": board.squares,
        "current_player": board.current_player,
        "continuation_sq": board.continuation_sq,
        "ply_count": board.ply_count,
        "halfmove_clock": board.halfmove_clock,
        "is_game_over": done,
        "winner": winner,
        "legal_moves": serialized_moves,
        "has_forced_capture": bool(legal_moves and legal_moves[0].is_capture),
        "evaluation_p1": float(eval_p1),
    }


@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"status": "Thai Checkers AlphaZero API running. UI index.html not yet built."}
    return FileResponse(index_path)


@app.get("/api/state")
def get_state():
    return serialize_board_state(game_board)


@app.post("/api/reset")
def reset_game():
    global game_board
    game_board = Board()
    game_board.setup_initial_position()
    return serialize_board_state(game_board)


@app.post("/api/human_move")
def play_human_move(req: HumanMoveRequest):
    global game_board
    done, _ = game_board.check_game_over()
    if done:
        raise HTTPException(status_code=400, detail="Game is already over.")

    legal_moves = game_board.get_legal_moves()
    matching = [m for m in legal_moves if m.from_sq == req.from_sq and m.to_sq == req.to_sq]
    if not matching:
        raise HTTPException(status_code=400, detail="Illegal move.")

    game_board.step(matching[0])
    return serialize_board_state(game_board)


@app.post("/api/ai_move")
def play_ai_move(req: AIMoveRequest):
    global game_board
    done, _ = game_board.check_game_over()
    if done:
        raise HTTPException(status_code=400, detail="Game is already over.")

    legal_moves = game_board.get_legal_moves()
    if not legal_moves:
        raise HTTPException(status_code=400, detail="No legal moves available.")

    candidate_thoughts: List[Dict[str, Any]] = []

    if req.engine == "minimax":
        mm = MinimaxAgent(depth=3)
        chosen_move = mm.select_move(game_board)
        chosen_action = chosen_move.action_id
    else:
        # AlphaZero MCTS
        mcts = MCTS(model, MCTSConfig(num_simulations=req.simulations))
        probs, root = mcts.get_action_probs(
            game_board, num_simulations=req.simulations, temperature=req.temperature
        )
        chosen_action = int(np.argmax(probs))

        # Extract top MCTS candidate moves for UI visualization
        if root.children:
            sorted_actions = sorted(
                root.children.keys(), key=lambda a: root.children[a].visit_count, reverse=True
            )
            total_visits = sum(root.children[a].visit_count for a in sorted_actions)
            for a in sorted_actions[:5]:
                node = root.children[a]
                f_sq, t_sq = Move.from_action_id(a)
                candidate_thoughts.append({
                    "from_sq": f_sq,
                    "to_sq": t_sq,
                    "from_coord": SQ_TO_COORD[f_sq],
                    "to_coord": SQ_TO_COORD[t_sq],
                    "from_algebraic": sq_to_algebraic(f_sq),
                    "to_algebraic": sq_to_algebraic(t_sq),
                    "notation": f"{sq_to_algebraic(f_sq)}-{sq_to_algebraic(t_sq)}",
                    "visits": node.visit_count,
                    "percentage": (node.visit_count / max(1, total_visits)) * 100,
                    "q_value": node.value,
                })

    from_sq, to_sq = Move.from_action_id(chosen_action)
    game_board.step(chosen_action)

    resp = serialize_board_state(game_board)
    resp["ai_move"] = {
        "from_sq": from_sq,
        "to_sq": to_sq,
        "from_coord": SQ_TO_COORD[from_sq],
        "to_coord": SQ_TO_COORD[to_sq],
        "from_algebraic": sq_to_algebraic(from_sq),
        "to_algebraic": sq_to_algebraic(to_sq),
        "notation": f"{sq_to_algebraic(from_sq)}-{sq_to_algebraic(to_sq)}",
    }
    resp["mcts_candidates"] = candidate_thoughts
    return resp


# ----------------------------------------------------
# Real-Time Training Endpoints & WebSocket
# ----------------------------------------------------

@app.websocket("/ws/training")
async def websocket_training_endpoint(websocket: WebSocket):
    if ws_manager.loop is None or not ws_manager.loop.is_running():
        ws_manager.loop = asyncio.get_running_loop()
    await ws_manager.connect(websocket)
    # Send initial status
    await websocket.send_json({
        "type": "init",
        "is_training": training_state.is_training,
        "current_iter": training_state.current_iter,
        "total_iters": training_state.total_iters,
        "current_episode": training_state.current_episode,
        "total_episodes": training_state.total_episodes,
        "buffer_size": training_state.buffer_size,
        "history": training_state.history,
        "device": str(device),
    })
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/api/train/status")
def get_training_status():
    return {
        "is_training": training_state.is_training,
        "current_iter": training_state.current_iter,
        "total_iters": training_state.total_iters,
        "current_episode": training_state.current_episode,
        "total_episodes": training_state.total_episodes,
        "buffer_size": training_state.buffer_size,
        "history": training_state.history,
        "device": str(device),
    }


@app.post("/api/train/start")
def start_training(req: TrainStartRequest):
    global training_state
    if training_state.is_training:
        raise HTTPException(status_code=400, detail="Training is already running.")

    def run_trainer():
        global model, training_state
        training_state.is_training = True
        training_state.total_iters = req.num_iters
        training_state.total_episodes = req.episodes_per_iter

        ws_manager.threadsafe_broadcast({
            "type": "log",
            "message": f"🚀 เริ่มต้นการฝึกฝน AI บน {device} (Iterations: {req.num_iters}, Sims: {req.mcts_sims})"
        })

        train_model = ThaiCheckersNet()
        best_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
        if req.resume and os.path.exists(best_path):
            train_model.load_checkpoint(best_path, device=device)
            ws_manager.threadsafe_broadcast({
                "type": "log",
                "message": f"📦 โหลดสมองเดิมจาก {best_path} สำเร็จ! กำลังฝึกต่อยอด..."
            })
        else:
            ws_manager.threadsafe_broadcast({
                "type": "log",
                "message": "🌱 เริ่มต้นโครงข่ายประสาทเทียมใหม่จากศูนย์ (Scratch)"
            })

        config = TrainerConfig(
            lr=1e-3,
            batch_size=req.batch_size,
            epochs_per_iter=req.epochs_per_iter,
            num_iters=req.num_iters,
            episodes_per_iter=req.episodes_per_iter,
            mcts_sims=req.mcts_sims,
            arena_games=req.arena_games,
            checkpoint_dir="checkpoints",
        )

        def on_step_cb(data: dict):
            training_state.current_iter = data["iteration"]
            training_state.current_episode = data["episode"]
            training_state.buffer_size = data["buffer_size"]
            ws_manager.threadsafe_broadcast({
                "type": "step",
                "data": data,
            })

        def on_episode_end_cb(iter_num, ep_num, total_eps, buf_size):
            training_state.current_iter = iter_num
            training_state.current_episode = ep_num
            training_state.buffer_size = buf_size
            ws_manager.threadsafe_broadcast({
                "type": "episode_end",
                "iteration": iter_num,
                "episode": ep_num,
                "total_episodes": total_eps,
                "buffer_size": buf_size,
            })

        def on_epoch_end_cb(iter_num, ep_num, loss, p_loss, v_loss):
            ws_manager.threadsafe_broadcast({
                "type": "epoch_end",
                "iteration": iter_num,
                "epoch": ep_num,
                "loss": loss,
                "policy_loss": p_loss,
                "value_loss": v_loss,
            })

        def on_iter_end_cb(iter_num, info):
            training_state.history["loss"].append(info["loss"])
            training_state.history["policy_loss"].append(info["policy_loss"])
            training_state.history["value_loss"].append(info["value_loss"])
            training_state.history["win_rate"].append(info["win_rate"])
            ws_manager.threadsafe_broadcast({
                "type": "iter_end",
                "iteration": iter_num,
                "info": info,
                "history": training_state.history,
            })
            if info.get("accepted"):
                try:
                    model.load_checkpoint(best_path, device=device)
                    ws_manager.threadsafe_broadcast({
                        "type": "log",
                        "message": "🏆 โมเดลเวอร์ชันใหม่ผ่านการทดสอบ! อัปเดตสมองในโหมดเล่นเกมเรียบร้อยแล้ว"
                    })
                except Exception as e:
                    print(f"Error reloading model: {e}")

        trainer = Trainer(
            model=train_model,
            config=config,
            device=device,
            on_step=on_step_cb,
            on_episode_end=on_episode_end_cb,
            on_epoch_end=on_epoch_end_cb,
            on_iter_end=on_iter_end_cb,
            visual_delay=req.visual_delay,
        )
        training_state.trainer = trainer

        try:
            hist = trainer.run_training_loop()
            training_state.history = hist
            if os.path.exists(best_path):
                model.load_checkpoint(best_path, device=device)
            ws_manager.threadsafe_broadcast({
                "type": "training_complete",
                "history": training_state.history,
            })
            ws_manager.threadsafe_broadcast({
                "type": "log",
                "message": "✅ การฝึกฝนเสร็จสิ้นเรียบร้อยแล้ว!"
            })
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"❌ Exception in run_trainer: {err}")
            ws_manager.threadsafe_broadcast({
                "type": "log",
                "message": f"❌ เกิดข้อผิดพลาดในการเทรน: {err}"
            })
        finally:
            training_state.is_training = False
            training_state.trainer = None

    t = threading.Thread(target=run_trainer, daemon=True)
    training_state.thread = t
    t.start()
    return {"status": "started", "num_iters": req.num_iters, "simulations": req.mcts_sims}


@app.post("/api/train/stop")
def stop_training():
    global training_state
    if not training_state.is_training or not training_state.trainer:
        return {"status": "not_running"}
    training_state.trainer.stop_requested = True
    ws_manager.threadsafe_broadcast({
        "type": "log",
        "message": "🛑 ได้รับคำสั่งหยุดการเทรน... กำลังบันทึก Checkpoint อย่างปลอดภัย"
    })
    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
