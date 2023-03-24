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
        self.init_state = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        self.state = self.init_state
        self.init_action = -1
        self.action = self.init_action
        self.reward = 0
        self.done = 0
        self.state_space = self.state.shape[0]
        self.action_space = 4

    def create_grid(self, i, j):
        self.state = self.init_state
        self.state[i][j] = 1
        return self.state, self.reward, self.done

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    def step(self, action):
        x, y = np.where(self.state == 1)
        pos_x = int(x)
        pos_y = int(y)
        print(self.state)
        if action == 'r':
            if self.state[pos_x][pos_y + 1] == -1:
                self.reward -= 1
                return self.state, self.reward, self.done

            elif pos_y < (self.state.shape[0]-1):
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x][pos_y + 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward, self.done

        elif action == 'l':
            if self.state[pos_x][pos_y - 1] == -1:
                self.reward -= 1
                return self.state, self.reward, self.done

            elif pos_y > 0:
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x][pos_y - 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward, self.done

        if action == 'u':
            if self.state[pos_x - 1][pos_y] == -1:
                self.reward -= 1
                return self.state, self.reward, self.done

            elif pos_x > 0:
                if self.state[pos_x][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x - 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward, self.done

        elif action == 'd':
            if self.state[pos_x + 1][pos_y] == -1:
                self.reward -= 1
                return self.state, self.reward, self.done

            elif pos_x < (self.state.shape[0] - 1):
                if self.state[pos_x + 1][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x + 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                return self.state, self.reward, self.done

    def reset(self):
        self.state = self.init_state
        self.action = self.init_action
        self.reward = 0
        self.done = 0
        return self.state

    def show(self):
        print(self.state, 'reward: ', self.reward, ', done: ', self.done)


# for i in range(1):
#     next_state, reward, done = env.step('r')
#     print(next_state, 'reward: ', reward)
#     next_state, reward, done = env.step('r')
#     print(next_state, 'reward: ', reward)
#     next_state, reward, done = env.step('r')
#     print(next_state, 'reward: ', reward)
#     next_state, reward, done = env.step('d')
#     print(next_state, 'reward: ', reward)
#     next_state, reward, done = env.step('d')
#     print(next_state, 'reward: ', reward)
#     next_state, reward, done = env.step('d')
#     print(next_state, 'reward: ', reward)