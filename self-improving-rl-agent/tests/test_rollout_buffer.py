import torch

from src.domain.rl.memory import RolloutBuffer


def _add_transition(buffer, value, reward, done):
    buffer.add(
        state=torch.tensor([value], dtype=torch.float32),
        action=0,
        log_prob=torch.tensor(0.0),
        value=torch.tensor(value, dtype=torch.float32),
        reward=reward,
        done=done,
    )


def test_terminal_transition_does_not_bootstrap_from_next_value():
    buffer = RolloutBuffer()
    _add_transition(buffer, value=0.5, reward=1.0, done=True)

    returns, advantages = buffer.compute_gae(
        next_value=torch.tensor(10.0), gamma=0.99, lam=0.95
    )

    assert torch.allclose(returns, torch.tensor([1.0]))
    assert torch.allclose(advantages, torch.tensor([0.5]))


def test_to_tensors_returns_aligned_batches_and_normalized_advantages():
    buffer = RolloutBuffer()
    _add_transition(buffer, value=0.0, reward=1.0, done=False)
    _add_transition(buffer, value=0.0, reward=0.0, done=True)

    states, actions, log_probs, returns, advantages = buffer.to_tensors()

    assert states.shape == (2, 1)
    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert returns.shape == (2,)
    assert advantages.shape == (2,)
    assert torch.isclose(advantages.mean(), torch.tensor(0.0))
