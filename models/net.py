"""Dual-Head Deep Residual Convolutional Neural Network for Thai Checkers AlphaZero.

Architecture:
- Input: (B, 6, 8, 8) tensor
  Planes:
  0: My pawns
  1: My kings
  2: Opponent pawns
  3: Opponent kings
  4: Continuation square mask
  5: Game progress indicator
- Backbone: 5 Residual Blocks with 128 filters
- Policy Head: outputs raw logits over 1,024 discrete actions
- Value Head: outputs board evaluation V in [-1.0, +1.0]
- Supports Apple Silicon GPU (MPS), CUDA, and CPU fallback.
"""

from __future__ import annotations
import os
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Selects best available device (Apple Silicon MPS -> CUDA -> CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ThaiCheckersNet(nn.Module):
    def __init__(self, num_res_blocks: int = 5, num_channels: int = 128, action_size: int = 1024):
        super().__init__()
        self.action_size = action_size

        # Input convolutional block
        self.in_conv = nn.Conv2d(6, num_channels, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(num_channels)

        # Residual backbone
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy Head (p(s, a))
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value Head (v(s) in [-1, 1])
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch, 6, 8, 8)
        Returns:
            policy_logits: (batch, 1024)
            value: (batch, 1) in [-1, 1]
        """
        out = F.relu(self.in_bn(self.in_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self, state_tensor: np.ndarray, legal_mask: Optional[np.ndarray] = None, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, float]:
        """Inference helper for single board state.
        
        Args:
            state_tensor: (6, 8, 8) numpy float32 array
            legal_mask: (1024,) boolean numpy array of legal moves
            device: torch device
        Returns:
            policy_probs: (1024,) normalized probability vector
            value: scalar float in [-1.0, 1.0]
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        t = torch.from_numpy(state_tensor).unsqueeze(0).float().to(device)
        logits, v = self.forward(t)
        logits = logits.squeeze(0).cpu().numpy()
        value = float(v.squeeze(0).cpu().item())

        if legal_mask is not None:
            # Mask illegal actions with -inf
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                return np.zeros(self.action_size, dtype=np.float32), value
            masked_logits = np.full_like(logits, -1e9)
            masked_logits[legal_mask] = logits[legal_mask]
            # Numerically stable softmax over legal moves
            max_val = np.max(masked_logits[legal_mask])
            exp_logits = np.exp(masked_logits[legal_mask] - max_val)
            probs = np.zeros(self.action_size, dtype=np.float32)
            probs[legal_mask] = exp_logits / np.sum(exp_logits)
            return probs, value
        else:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return probs, value

    def save_checkpoint(self, filepath: str) -> None:
        """Saves model weights to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath: str, device: Optional[torch.device] = None) -> None:
        """Loads model weights from disk."""
        if device is None:
            device = get_device()
        self.load_state_dict(torch.load(filepath, map_location=device))
        self.to(device)
