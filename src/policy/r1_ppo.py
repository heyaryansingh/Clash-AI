import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.fc(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

class PPOAgent:
    def __init__(self, policy:PolicyNet, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def act(self, state):
        logits, value = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
