'''
github.com/minje-seok/Git-RL/env/grid.py

"Environment - Grid"

[[ init, 0, 0, 0 ],
[ 0, -1, 0, 0 ],
[ 0, 0, 0, 0 ],
[ 0, 0, -1, dst ]]

'''
import torch
import numpy as np

print("{0}(CUDA available: {1})".format(torch.cuda.get_device_name(0), torch.cuda.is_available()))
device = torch.cuda.device(0)

class GridEnv():
    def __init__(self):
        self.state = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, -1, 99]])
        self.action = -1
        self.reward = 0


    def step(self, action):
        x, y = np.where(self.state == 1)
        pos_x = int(x)
        pos_y = int(y)
        if action == 'r':
            if self.state[pos_x][pos_y + 1] == -1:
                self.reward -= 1
                return self.state, self.reward

            elif pos_y < (self.state.shape[0]-1):
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    return self.state, self.reward

                self.state[pos_x][pos_y + 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward

        elif action == 'l':
            if self.state[pos_x][pos_y - 1] == -1:
                self.reward -= 1
                return self.state, self.reward

            elif pos_y > 0:
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    return self.state, self.reward

                self.state[pos_x][pos_y - 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward

        if action == 'u':
            if self.state[pos_x - 1][pos_y] == -1:
                self.reward -= 1
                return self.state, self.reward

            elif pos_x > 0:
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    return self.state, self.reward

                self.state[pos_x - 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward

        elif action == 'd':
            if self.state[pos_x + 1][pos_y] == -1:
                self.reward -= 1
                return self.state, self.reward

            elif pos_x < (self.state.shape[0] - 1):
                if self.state[pos_x + 1][pos_y] == 99:
                    self.reward += 99
                    return self.state, self.reward

                self.state[pos_x + 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward

    def cur_state(self):
        print('------------')
        print(self.state, 'reward: ',self.reward)

env = GridEnv()
print()
for i in range(1):
    next_state, reward = env.step('r')
    print(next_state, 'reward: ', reward)
    next_state, reward = env.step('r')
    print(next_state, 'reward: ', reward)
    next_state, reward = env.step('r')
    print(next_state, 'reward: ', reward)
    next_state, reward = env.step('d')
    print(next_state, 'reward: ', reward)
    next_state, reward = env.step('d')
    print(next_state, 'reward: ', reward)
    next_state, reward = env.step('d')
    print(next_state, 'reward: ', reward)