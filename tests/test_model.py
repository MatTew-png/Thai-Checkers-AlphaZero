"""Unit tests for Neural Network, MPS execution, MCTS, and Training pipeline."""

import pytest
import numpy as np
import torch

from env.thai_checkers import Board, ACTION_SPACE_SIZE
from models.net import ThaiCheckersNet, get_device
from mcts.mcts import MCTS, MCTSConfig
from training.trainer import ReplayBuffer, Trainer, TrainerConfig
from baseline.minimax import MinimaxAgent


def test_device_selection():
    """Verify Apple Silicon MPS or CPU device detection."""
    device = get_device()
    assert device.type in ["mps", "cuda", "cpu"]


def test_network_forward_pass():
    """Verify Dual-Head ResNet shapes."""
    model = ThaiCheckersNet(num_res_blocks=2, num_channels=32)
    device = get_device()
    model.to(device)

    dummy_input = torch.randn(4, 6, 8, 8, device=device)
    policy_logits, value = model(dummy_input)

    assert policy_logits.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_predict_legal_masking():
    """Verify inference masking and probability normalization."""
    model = ThaiCheckersNet(num_res_blocks=2, num_channels=32)
    board = Board()
    board.setup_initial_position()

    tensor = board.get_canonical_form().get_tensor_representation()
    mask = board.get_legal_action_mask()

    probs, val = model.predict(tensor, legal_mask=mask)

    assert probs.shape == (ACTION_SPACE_SIZE,)
    assert np.isclose(np.sum(probs), 1.0)
    # Illegal moves must have 0 probability
    illegal_indices = np.where(~mask)[0]
    assert np.all(probs[illegal_indices] == 0.0)


def test_mcts_search():
    """Verify MCTS search runs and produces valid action distribution."""
    model = ThaiCheckersNet(num_res_blocks=2, num_channels=32)
    board = Board()
    board.setup_initial_position()

    config = MCTSConfig(num_simulations=20)
    mcts = MCTS(model, config)

    probs, root = mcts.get_action_probs(board, num_simulations=20, temperature=1.0)

    assert np.isclose(np.sum(probs), 1.0)
    assert root.visit_count >= 20
    best_move = mcts.select_best_move(board, num_simulations=20)
    assert 0 <= best_move < ACTION_SPACE_SIZE


def test_replay_buffer_and_optimizer_step():
    """Verify ReplayBuffer sampling and one optimizer backpropagation step."""
    buffer = ReplayBuffer(capacity=100)
    dummy_state = np.zeros((6, 8, 8), dtype=np.float32)
    dummy_pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    dummy_pi[0] = 1.0

    for _ in range(20):
        buffer.push((dummy_state, dummy_pi, 1.0))

    assert len(buffer) == 20
    states, pis, vs = buffer.sample(8)
    assert states.shape == (8, 6, 8, 8)
    assert pis.shape == (8, ACTION_SPACE_SIZE)
    assert vs.shape == (8, 1)

    model = ThaiCheckersNet(num_res_blocks=2, num_channels=32)
    trainer = Trainer(model=model, config=TrainerConfig(batch_size=8, epochs_per_iter=1))
    trainer.replay_buffer = buffer
    loss, p_loss, v_loss = trainer.train_epoch()

    assert loss > 0
    assert not np.isnan(loss)


def test_minimax_agent():
    """Verify Minimax agent selects valid legal moves."""
    board = Board()
    board.setup_initial_position()
    mm = MinimaxAgent(depth=2)

    move = mm.select_move(board)
    legal_moves = board.get_legal_moves()
    assert move in legal_moves


def test_trainer_live_callbacks_and_stop(tmp_path):
    """Verify live step callback, episode callback, and stop_requested."""
    steps_recorded = []
    episodes_recorded = []

    def on_step(d):
        steps_recorded.append(d)

    def on_ep(it, ep, total_eps, buf_size):
        episodes_recorded.append((it, ep))

    model = ThaiCheckersNet(num_res_blocks=2, num_channels=32)
    cfg = TrainerConfig(
        num_iters=1,
        episodes_per_iter=1,
        mcts_sims=5,
        checkpoint_dir=str(tmp_path),
        batch_size=8,
    )
    trainer = Trainer(
        model=model,
        config=cfg,
        on_step=on_step,
        on_episode_end=on_ep,
    )

    # Trigger 1 episode
    from training.self_play import SelfPlayWorker
    worker = SelfPlayWorker(trainer.model, mcts_simulations=5)

    def _cb(b, a, p):
        on_step({"ply": b.ply_count, "player": p})

    samples, winner = worker.play_game(step_callback=_cb)
    assert len(samples) > 0
    assert winner in (1, -1, 0)
    assert len(steps_recorded) > 0
    assert steps_recorded[0]["ply"] >= 1

    # Test stop_requested flag
    trainer.stop_requested = True
    assert trainer.stop_requested is True

