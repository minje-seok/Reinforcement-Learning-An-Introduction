import numpy as np

from env import grid
from env.grid import GridEnv



class DP:
    def __init__(self):
        self.V = np.zeros((4, 4))
        self.next_V = self.V.copy()
        self.env = GridEnv()

    def policy_evaluation(self, gamma=1.0, theta=0.0000000001):
        # gamma: discount factor
        # theta: threshold to stop iteration
        # delta: difference between V[i][j] and next_V[i][j](= new_V)
        iter = 0
        while True:
            iter += 1
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    new_V = 0
                    for a in ['u', 'd', 'l', 'r']:
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(a)
                        x, y = np.where(next_state == 1)
                        pos_x = int(x)
                        pos_y = int(y)
                        new_V += 0.25 * (reward + gamma * self.V[pos_x][pos_y])
                        self.env.reset()
                    self.next_V[i][j] = new_V
                    delta = max(delta, abs(self.V[i][j] - new_V))
            self.V = self.next_V.copy()

            if delta < theta:
                print('Policy Evaluation - iter:{0}'.format(iter))
                print(self.next_V)
                break

    def policy_improvement(self):

        return


agent = DP()
agent.policy_evaluation()

