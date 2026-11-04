## Summary

Add focused PPO component tests and a GitHub Actions test matrix.

## Motivation / Problem

The project had only smoke tests. Rollout-buffer behavior and policy tensor contracts were not covered by automated validation.

## Changes

- Test terminal-transition GAE behavior.
- Test rollout tensor shapes and advantage normalization.
- Test policy action and evaluation outputs.
- Run tests on Python 3.10, 3.11, and 3.12.
- Install PyTorch explicitly in CI.

## How to test

```bash
python -m pytest self-improving-rl-agent/tests -q
```

## Checklist

- [ ] Tests pass
- [x] Documentation updated where applicable
- [x] No breaking changes
- [ ] Linked issue, if applicable
