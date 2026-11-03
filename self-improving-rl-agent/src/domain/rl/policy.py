"""Policy network — actor-critic for PPO with ResNet backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple

class ResBlock(nn.Module):
    """Residual block to prevent vanishing gradients and improve depth."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))

class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    Implements a Residual backbone with LayerNorm for high stability.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()

        # Professional Backbone: Linear -> LayerNorm -> ResBlocks
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden)
        )

        # Actor head — outputs action logits
        self.actor = nn.Linear(hidden, action_dim)

        # Critic head — outputs state value V(s)
        self.critic = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Professional orthogonal initialization
                gain = 0.01 if m == self.actor else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        x = self.input_layer(state)
        x = self.res_blocks(x)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        logits, values = self(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
