import numpy as np

from env import grid
from env.grid import GridEnv

env = GridEnv()

class DP:
    def __init__(self):
        self.tabular = np.zeros(4, 4)
        self.pre_tabular = self.tabular

    def iteration_step(self):
        for i in range(self.tabular.shape[0]):
            for j in range(self.tabular.shape[1]):
                self.pre_tabular = self.tabular
                env.create_grid(i, j)
                self.tabular[i, j] = 0.25 * (env.step('r') + self.pre_tabular[i, j]) + \
                                   0.25 * (env.step('l') + self.pre_tabular[i, j]) + \
                                   0.25 * (env.step('u') + self.pre_tabular[i, j]) + \
                                   0.25 * (env.step('d') + self.pre_tabular[i, j])
                env.reset()

        self.pre_tabular = self.tabular

    def show(self):
        return self.tabular


agent = DP()
print(agent.state)