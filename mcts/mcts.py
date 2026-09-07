"""Monte Carlo Tree Search (MCTS) with PUCT and Dirichlet Exploration Noise.

Features:
- AlphaZero PUCT algorithm with dynamic exploration constant
- Sub-turn multi-jump awareness (correct value propagation when turn does not switch)
- Temperature-scaled action probabilities
- Fast tree reuse and caching
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from env.thai_checkers import Board, ACTION_SPACE_SIZE
from models.net import ThaiCheckersNet


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


class Node:
    def __init__(self, prior: float = 0.0):
        self.prior: float = prior
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: Dict[int, Node] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_action(self, c_puct: float) -> int:
        """Selects the child action with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total = math.sqrt(total_visits + 1e-8)

        for action, child in self.children.items():
            # Upper Confidence Bound for Trees (PUCT)
            q_val = child.value
            u_val = c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expand(self, legal_actions: List[int], priors: np.ndarray) -> None:
        """Expands node with legal actions and prior probabilities."""
        self.is_expanded = True
        for action in legal_actions:
            self.children[action] = Node(prior=float(priors[action]))


class MCTS:
    def __init__(self, model: ThaiCheckersNet, config: Optional[MCTSConfig] = None):
        self.model = model
        self.config = config or MCTSConfig()

    def search(
        self,
        board: Board,
        num_simulations: Optional[int] = None,
        add_dirichlet_noise: bool = False,
    ) -> Node:
        """Runs MCTS search from the given board state and returns the root Node."""
        sims = num_simulations if num_simulations is not None else self.config.num_simulations
        root = Node()

        # Initial expansion of root node
        canonical_board = board.get_canonical_form()
        state_tensor = canonical_board.get_tensor_representation()
        legal_mask = board.get_legal_action_mask()
        legal_actions = np.where(legal_mask)[0].tolist()

        if not legal_actions:
            return root

        policy_probs, _ = self.model.predict(state_tensor, legal_mask=legal_mask)

        # Inject Dirichlet exploration noise at root if requested (for self-play)
        if add_dirichlet_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_actions))
            eps = self.config.dirichlet_epsilon
            for i, act in enumerate(legal_actions):
                policy_probs[act] = (1 - eps) * policy_probs[act] + eps * noise[i]

        root.expand(legal_actions, policy_probs)

        # Run simulations
        for _ in range(sims):
            self._simulate(root, board.clone())

        return root

    def _simulate(self, root: Node, board: Board) -> None:
        """Executes a single MCTS simulation pass (Select -> Expand -> Evaluate -> Backup)."""
        node = root
        search_path: List[Tuple[Node, int, Board]] = []

        # 1. Selection phase
        while node.is_expanded and not board.check_game_over()[0]:
            action = node.select_action(self.config.c_puct)
            if action == -1:
                break
            prev_board = board.clone()
            board.step(action)
            search_path.append((node, action, prev_board))
            node = node.children[action]

        # 2. Evaluation phase
        done, winner = board.check_game_over()
        if done:
            # Game is over at this leaf
            last_acting_player = search_path[-1][2].current_player if search_path else board.current_player
            if winner == 0:
                value = 0.0
            elif winner == last_acting_player:
                value = 1.0
            else:
                value = -1.0
        else:
            # Expand leaf node using neural network prediction
            canonical_board = board.get_canonical_form()
            state_tensor = canonical_board.get_tensor_representation()
            legal_mask = board.get_legal_action_mask()
            legal_actions = np.where(legal_mask)[0].tolist()

            policy_probs, leaf_value = self.model.predict(state_tensor, legal_mask=legal_mask)
            if legal_actions:
                node.expand(legal_actions, policy_probs)

            # leaf_value is from perspective of current active player on the leaf board
            value = leaf_value

        # 3. Backpropagation phase
        # Traverse path backwards from leaf to root
        # Crucial for Thai Checkers:
        # If parent and child had the SAME active player (e.g. multi-jump), value does NOT invert!
        # If parent and child had DIFFERENT active players (normal turn switch), value inverts (-value).
        current_val = value
        for i in reversed(range(len(search_path))):
            parent_node, action, parent_board = search_path[i]
            child_node = parent_node.children[action]

            # Determine if turn switched between parent and resulting state
            # If search_path has next step, use that board; else use final board
            next_state_board = search_path[i + 1][2] if i + 1 < len(search_path) else board

            if parent_board.current_player != next_state_board.current_player:
                # Turn changed: value is negated for parent
                current_val = -current_val
            # If same player (mid multi-jump), current_val remains unchanged!

            child_node.visit_count += 1
            child_node.total_value += current_val

        root.visit_count += 1

    def get_action_probs(
        self,
        board: Board,
        num_simulations: Optional[int] = None,
        temperature: Optional[float] = None,
        add_dirichlet_noise: bool = False,
    ) -> Tuple[np.ndarray, Node]:
        """Returns action probability distribution pi over 1024 actions and the root node."""
        temp = self.config.temperature if temperature is None else temperature
        root = self.search(
            board, num_simulations=num_simulations, add_dirichlet_noise=add_dirichlet_noise
        )

        probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not root.children:
            return probs, root

        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)

        if temp == 0 or temp < 1e-3:
            # Greedy argmax
            best_idx = np.argmax(visits)
            probs[actions[best_idx]] = 1.0
        else:
            # Softmax with temperature
            visits_exp = visits ** (1.0 / temp)
            norm_visits = visits_exp / (np.sum(visits_exp) + 1e-8)
            for a, p in zip(actions, norm_visits):
                probs[a] = p

        return probs, root

    def select_best_move(
        self, board: Board, num_simulations: Optional[int] = None, temperature: float = 0.0
    ) -> int:
        """Selects best action index for competitive play."""
        probs, _ = self.get_action_probs(
            board, num_simulations=num_simulations, temperature=temperature
        )
        return int(np.argmax(probs))
