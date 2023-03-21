import numpy as np

from env import grid
from env.grid import GridEnv

env = GridEnv()

class agent:
    def __init__(self, env):
        self.state = np.zeros(4,4)
        self.next_state = self.state
        self.action = -1
        self.reward = 0
        self.env = env

    def iteration_step(self):
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                self.next_state[i, j] = 0.25 * (env.step(self.state[i, j], 'r') + self.state[i, j]) + \
                                   0.25 * (env.step(self.state[i, j], 'l') + self.state[i, j]) + \
                                   0.25 * (env.step(self.state[i, j], 'u') + self.state[i, j]) + \
                                   0.25 * (env.step(self.state[i, j], 'd') + self.state[i, j])

        self.state = self.next_state
        return self.state