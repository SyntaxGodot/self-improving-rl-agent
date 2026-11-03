"""PPO Agent — orchestrates policy, memory, and self-improving updates."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any
import numpy as np

from src.domain.rl.policy import PolicyNetwork
from src.domain.rl.memory import RolloutBuffer
from src.domain.entities.state import State
from src.domain.entities.action import Action, ACTION_SPACE_SIZE

class PPOAgent:
    """
    Self-Improving Proximal Policy Optimization agent.
    Features GAE, ResNet backbone, and automatic hyperparameter mutation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ):
        self.device     = torch.device(device)
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.ppo_epochs = ppo_epochs
        self.gamma      = gamma
        self.lam        = lam

        self.policy  = PolicyNetwork(state_dim, action_dim, hidden).to(self.device)
        self.buffer  = RolloutBuffer()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Professional Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000)

        self.total_updates = 0
        self.reward_history = []
        self.best_avg_reward = -float("inf")

    def act(self, state: State) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        state_vec = torch.tensor(state.to_vector(), dtype=torch.float32).to(self.device)
        action_idx, log_prob, value = self.policy.act(state_vec)
        return Action.from_index(action_idx), log_prob, value

    def store(self, state: State, action: Action, log_prob, value, reward: float, done: bool):
        state_vec = torch.tensor(state.to_vector(), dtype=torch.float32)
        self.buffer.add(state_vec, action.action_type.value, log_prob, value, reward, done)

    def update(self, last_value: torch.Tensor = torch.tensor(0.0)) -> dict:
        """Run PPO update with GAE and Self-Improving logic."""
        if len(self.buffer) == 0:
            return {}

        # 1. Compute GAE Returns and Advantages
        returns, advantages = self.buffer.to_tensors(self.gamma, self.lam, last_value.to(self.device))
        
        states      = torch.stack([t.state for t in self.buffer.transitions]).to(self.device)
        actions     = torch.tensor([t.action for t in self.buffer.transitions], dtype=torch.long).to(self.device)
        old_log_probs = torch.stack([t.log_prob for t in self.buffer.transitions]).detach().to(self.device)
        returns     = returns.to(self.device)
        advantages  = advantages.to(self.device)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for _ in range(self.ppo_epochs):
            log_probs, values, entropy = self.policy.evaluate(states, actions)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Total loss (with entropy bonus for exploration)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"]  += value_loss.item()
            metrics["entropy"]     += entropy.mean().item()

        self.scheduler.step()
        self.buffer.clear()
        self.total_updates += 1
        
        n = self.ppo_epochs
        return {k: v / n for k, v in metrics.items()}

    # ── Self-Improving Meta-Learning ─────────────────────────────────────────────

    def evolve_hyperparameters(self, current_avg_reward: float):
        """
        Automatic Hyperparameter Mutation.
        If reward plateaus, mutate gamma or clip_eps to explore new policy spaces.
        """
        self.reward_history.append(current_avg_reward)
        if len(self.reward_history) < 10: return

        # Check for plateau (last 10 episodes have < 1% improvement)
        recent = self.reward_history[-10:]
        improvement = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)
        
        if improvement < 0.01:
            logger.info("Plateau detected. Mutating hyperparameters for self-improvement...")
            # Mutate gamma (discount factor) and clip_eps
            self.gamma = np.clip(self.gamma * np.random.uniform(0.95, 1.05), 0.9, 0.999)
            self.clip_eps = np.clip(self.clip_eps * np.random.uniform(0.9, 1.1), 0.1, 0.3)
            logger.info(f"New params: gamma={self.gamma:.4f}, clip_eps={self.clip_eps:.4f}")

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "updates": self.total_updates}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.total_updates = ckpt.get("updates", 0)
