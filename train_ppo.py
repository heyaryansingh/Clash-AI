"""
PPO Training Script
TODO:
  1. Rollout episodes in env
  2. Collect (state, action, reward, log_prob, value)
  3. Update with PPOAgent
"""

import torch
from src.policy.rl_ppo import PolicyNet

def main():
    # TODO: connect to env
    state_dim = 128  # placeholder
    n_actions = 100

    model = PolicyNet(state_dim, n_actions)
    print("PPO policy model initialized.")

if __name__ == "__main__":
    main()
