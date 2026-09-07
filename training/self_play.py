"""Self-Play Data Generator for Thai Checkers AlphaZero.

Generates training samples (state, pi, z) through games played by the network against itself
guided by MCTS and exploration noise.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable
import numpy as np

from env.thai_checkers import Board, Player
from models.net import ThaiCheckersNet
from mcts.mcts import MCTS, MCTSConfig


class SelfPlayWorker:
    def __init__(
        self,
        model: ThaiCheckersNet,
        mcts_simulations: int = 100,
        use_pcr: bool = True,
        pcr_fast_ratio: float = 0.75,
        pcr_fast_sims: Optional[int] = None,
    ):
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.use_pcr = use_pcr
        self.pcr_fast_ratio = pcr_fast_ratio
        self.pcr_fast_sims = pcr_fast_sims or max(15, mcts_simulations // 4)
        self.config = MCTSConfig(
            num_simulations=mcts_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=1.0,
        )
        self.mcts = MCTS(self.model, self.config)

    def play_game(
        self,
        temp_threshold: int = 14,
        step_callback: Optional[Callable[[Board, int, int], None]] = None,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int, str]:
        """Plays a single game of self-play and returns training examples and final winner.
        
        Args:
            temp_threshold: Ply count before temperature drops from 1.0 to 0.2 (more exploitation)
            step_callback: Optional callback invoked after each move
            
        Returns:
            Tuple of (training_samples, winner, reason):
                training_samples: List of (canonical_state_tensor, target_policy_pi, game_outcome_z)
                winner: 1 (White wins), -1 (Black wins), 0 (Draw)
                reason: Description of how the game terminated
        """
        board = Board()
        board.setup_initial_position()

        # History stores: (canonical_tensor, pi_distribution, active_player, is_full_search)
        episode_data: List[Tuple[np.ndarray, np.ndarray, int, bool]] = []

        done = False
        winner: Optional[int] = None

        while not done:
            temp = 1.0 if board.ply_count < temp_threshold else 0.2

            # KataGo Playout Cap Randomization (PCR)
            is_full_search = True
            if self.use_pcr and self.mcts_simulations > 20:
                if np.random.rand() < self.pcr_fast_ratio:
                    current_sims = self.pcr_fast_sims
                    is_full_search = False
                else:
                    current_sims = self.mcts_simulations
                    is_full_search = True
            else:
                current_sims = self.mcts_simulations

            canonical_board = board.get_canonical_form()
            state_tensor = canonical_board.get_tensor_representation()

            pi, _ = self.mcts.get_action_probs(
                board,
                num_simulations=current_sims,
                temperature=temp,
                add_dirichlet_noise=True,
            )

            # Record step
            episode_data.append((state_tensor, pi, board.current_player, is_full_search))

            # Sample action from pi
            action_indices = np.where(pi > 0)[0]
            if len(action_indices) == 0:
                # No legal moves, terminate
                done, winner = board.check_game_over()
                break

            action_probs = pi[action_indices]
            action_probs = action_probs / np.sum(action_probs)
            chosen_action = int(np.random.choice(action_indices, p=action_probs))

            # Execute move
            acting_player = board.current_player
            board.step(chosen_action)
            if step_callback is not None:
                step_callback(board, chosen_action, acting_player)
            done, winner = board.check_game_over()

        final_winner = 0 if winner is None else winner
        reason = board.get_game_over_reason()

        # Build training tuples with final outcome z
        training_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for state_tensor, target_pi, acting_player, _ in episode_data:
            if final_winner == 0:
                z = 0.0
            elif final_winner == acting_player:
                z = 1.0
            else:
                z = -1.0
            training_samples.append((state_tensor, target_pi, z))

        return training_samples, final_winner, reason
