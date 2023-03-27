import numpy as np

from env import grid
from env.grid import GridEnv



class DP:
    def __init__(self):
        self.state_V = np.zeros((4, 4))
        self.next_state_V = np.zeros((4, 4))
        self.env = GridEnv()

    def policy_evaluation(self):
        for i in range(self.state_V.shape[0]):
            for j in range(self.state_V.shape[1]):
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    continue

                next_state_V = 0
                for a in ['u', 'd', 'l', 'r']:
                    self.env.create_grid(i, j)
                    next_state, reward, done = self.env.step(a)
                    x, y = np.where(next_state == 1)
                    pos_x = int(x)
                    pos_y = int(y)
                    next_state_V += 0.25 * (reward + self.state_V[pos_x][pos_y])
                    self.env.reset()
                self.next_state_V[i][j] = next_state_V
        self.state_V = self.next_state_V.copy()

    def show(self):
        print(self.next_state_V)

agent = DP()
agent.policy_evaluation()
agent.show()
agent.policy_evaluation()
agent.show()
agent.policy_evaluation()
agent.show()