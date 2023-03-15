# minje-seok/Git-RL/env
import torch

class GridEnv():
    def __init__(self):
        self.state = torch.tensor([-1, -1])
        self.action = -1
        self.reward = 0
