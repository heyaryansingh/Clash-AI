import torch
import torch.nn as nn

class BehaviorCloneNet(nn.Module):
    def __init__(self, state_dim:int, n_cards:int=8, grid_size:int=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.card_head = nn.Linear(128, n_cards)
        self.place_head = nn.Linear(128, grid_size)

    def forward(self, x):
        x = self.fc(x)
        return {
            "card_logits": self.card_head(x),
            "place_logits": self.place_head(x)
        }
