import numpy as np

from env import grid
from env.grid import GridEnv

env = GridEnv()

class DP:
    def __init__(self):
        self.tabular = np.zeros(4, 4)
        self.next_tabular = self.tabular
        self.env = env

    def iteration_step(self):
        for i in range(self.tabular.shape[0]):
            for j in range(self.tabular.shape[1]):
                self.next_state[i, j] = 0.25 * (env.step('r') + self.state[i, j]) + \
                                   0.25 * (env.step('l') + self.state[i, j]) + \
                                   0.25 * (env.step('u') + self.state[i, j]) + \
                                   0.25 * (env.step('d') + self.state[i, j])

        self.state = self.next_state
        return self.state

agent = DP()
print(agent.state)