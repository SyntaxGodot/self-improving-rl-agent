"""Rollout buffer for on-policy PPO training with GAE."""
from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class Transition:
    state:      torch.Tensor
    action:     int
    log_prob:   torch.Tensor
    value:      torch.Tensor
    reward:     float
    done:       bool

class RolloutBuffer:
    """
    Stores a full episode rollout for PPO update.
    Implements Generalized Advantage Estimation (GAE) to reduce variance.
    """

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ):
        self.transitions.append(
            Transition(state, action, log_prob, value, reward, done)
        )

    def compute_gae(self, next_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE and Returns.
        GAE reduces variance compared to Monte-Carlo while maintaining a reasonable bias.
        """
        values = torch.stack([t.value for t in self.transitions])
        rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # Calculate GAE backward
        # Delta = r + gamma * V(s') * (1-done) - V(s)
        next_v = next_value.detach()
        for t in reversed(range(len(self.transitions))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v * mask - values[t].detach()
            advantages[t] = last_gae_lam = delta + gamma * lam * mask * last_gae_lam
            next_v = values[t]
            
        returns = advantages + values.detach()
        return returns, advantages

    def to_tensors(self, gamma: float = 0.99, lam: float = 0.95, next_value: torch.Tensor = torch.tensor(0.0)):
        returns, advantages = self.compute_gae(next_value, gamma, lam)
        
        states    = torch.stack([t.state    for t in self.transitions])
        actions   = torch.tensor([t.action  for t in self.transitions], dtype=torch.long)
        log_probs = torch.stack([t.log_prob for t in self.transitions])
        values    = torch.stack([t.value    for t in self.transitions])
        
        # Standardize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, returns, advantages

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)
