"""
Behavior Cloning Training Script
TODO:
  1. Load dataset of (state, action)
  2. Train BehaviorCloneNet
  3. Save checkpoints
"""

import torch
from src.policy.imitation import BehaviorCloneNet

def main():
    # TODO: load dataset
    state_dim = 128  # placeholder
    n_cards = 8
    grid_size = 100

    model = BehaviorCloneNet(state_dim, n_cards, grid_size)
    print("Behavior cloning model initialized.")

if __name__ == "__main__":
    main()
