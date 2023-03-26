import numpy as np

from env import grid
from env.grid import GridEnv



class DP:
    def __init__(self, env):
        self.tabular = np.zeros((4, 4))
        self.pre_tabular = self.tabular
        self.env = env

    def iteration_step(self):
        for i in range(self.tabular.shape[0]):
            for j in range(self.tabular.shape[1]):
                self.pre_tabular = self.tabular
                for a in ["r", "l", "u", "d"]:
                    print(a)
                    self.env.reset()
                    self.env.create_grid(i, j)
                    print(self.env.state)
                    state, reward, done = self.env.step(a)
                    self.tabular[i][j] += 0.25 * (reward + self.pre_tabular[i][j])


        self.pre_tabular = self.tabular

    def show(self):
        print(self.tabular)

env = GridEnv()
agent = DP(env)
agent.iteration_step()
