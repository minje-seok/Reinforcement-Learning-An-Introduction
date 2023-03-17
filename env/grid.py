'''
github.com/minje-seok/Git-RL/env/grid.py

"Environment - Grid"

[[ init, 0, 0, 0 ],
[ 0, 0, 0, 0 ],
[ 0, 0, 0, 0 ],
[ 0, 0, 0, dst ]]

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
                               [0, -1, 0, 99]])
        self.action = -1
        self.reward = 0


    def step(self, action):
        x, y = np.where(self.state == 1)
        print(x, y)
        pos_x = int(x)
        pos_y = int(y)
        print(pos_x, pos_y)
        if action == 'r':
            if pos_x < self.state.shape[0] | self.state[pos_x][pos_y+1] != -1:
                self.state[pos_x + 1] = 1
                self.state[pos_x] = 0
                self.reward -= 1
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 1
                    return
            else:
                self.reward -= 1
                return

env = GridEnv()

for i in range(3):
    env.step('r')
    print(env.state)
