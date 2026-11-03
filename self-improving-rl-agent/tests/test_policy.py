import torch

from src.domain.rl.policy import PolicyNetwork


def test_policy_forward_returns_action_logits_and_values():
    policy = PolicyNetwork(state_dim=4, action_dim=3, hidden=16)
    states = torch.zeros(2, 4)

    logits, values = policy(states)

    assert logits.shape == (2, 3)
    assert values.shape == (2,)


def test_policy_action_and_evaluation_outputs_are_compatible():
    policy = PolicyNetwork(state_dim=4, action_dim=3, hidden=16)
    state = torch.zeros(4)

    action, log_prob, value = policy.act(state)
    log_probs, values, entropy = policy.evaluate(
        state.unsqueeze(0), torch.tensor([action])
    )

    assert 0 <= action < 3
    assert log_prob.shape == torch.Size([])
    assert value.shape == torch.Size([1])
    assert log_probs.shape == torch.Size([1])
    assert values.shape == torch.Size([1])
    assert entropy.shape == torch.Size([1])
