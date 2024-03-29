
import torch
import numpy as np

# print("{0}(CUDA available: {1})".format(torch.cuda.get_device_name(0), torch.cuda.is_available()))
# device = torch.cuda.device(0)

class GridEnv():
    '''
    github.com/minje-seok/Git-RL/env/grid.py

    "Environment - Grid"
    Creates an environment with a 4 by 4 array(default) as state, with the first and last elements defined as init state
    and terminal state, respectively(resizeable through self.init_state).

    [[ init, 0, 0, 0 ],
    [ 0, 0, 0, 0 ],
    [ 0, 0, 0, 0 ],
    [ 0, 0, 0, dst ]]

    Args:
        state(np.array): A numpy array in which the information of the current location is represented by 1.
        action(int): top down left right (discrete). If the state in the direction of the action is blocked or at the end of the array, the position is maintained.
        reward(int): -1 when moving to any state except init state and terminal state. If the terminal state is reached, 99 is paid.
    '''
    def __init__(self, row=4, col=4):
        self.init_state = np.zeros((row, col))
        self.state = self.init_state
        self.init_action = -1
        self.action = self.init_action
        self.reward = 0
        self.done = 0
        self.state_space = self.state.shape[0]
        self.action_space = 4
        self.policy = np.ones((4, 4)) / self.action_space

    def create_grid(self, i, j):
        self.state = self.init_state
        self.state[i][j] = 1
        # print("---create_grid[{0}][{1}]---".format(i, j))
        return self.state, self.reward, self.done

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    def get_current_state(self):
        x, y = np.where(self.state)
        pos_x = int(x)
        pos_y = int(y)
        return pos_x, pos_y

    def step(self, action):
        x, y = np.where(self.state == 1)
        pos_x = int(x)
        pos_y = int(y)
        if action == 0: # up
            try:
                self.state[pos_x - 1][pos_y]
            except IndexError:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            if self.state[pos_x - 1][pos_y] == -1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            elif pos_x == 0:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            else:
                if self.state[pos_x - 1][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x - 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

        elif action == 1: # down
            try:
                self.state[pos_x + 1][pos_y]
            except IndexError:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            if self.state[pos_x + 1][pos_y] == -1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            elif pos_x == self.state.shape[0] - 1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            else:
                if self.state[pos_x + 1][pos_y] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x + 1][pos_y] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

        elif action == 2: # left
            try:
                self.state[pos_x][pos_y - 1]
            except IndexError:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            if self.state[pos_x][pos_y - 1] == -1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            elif pos_y == 0:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            else:
                if self.state[pos_x][pos_y - 1] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x][pos_y - 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

        elif action == 3: # right
            try:
                self.state[pos_x][pos_y + 1]
            except IndexError:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            if self.state[pos_x][pos_y + 1] == -1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            elif pos_y == self.state.shape[0] - 1:
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

            else:
                if self.state[pos_x][pos_y + 1] == 99:
                    self.reward += 99
                    self.done = 1
                    return self.state, self.reward, self.done

                self.state[pos_x][pos_y + 1] = 1
                self.state[pos_x][pos_y] = 0
                self.reward -= 1
                self.done = 0
                return self.state, self.reward, self.done

    def reset(self):
        # self.state = self.init_state
        x, y = np.where(self.state == 1)
        pos_x = int(x)
        pos_y = int(y)
        self.state[pos_x][pos_y] = 0
        self.action = self.init_action
        self.reward = 0
        self.done = 0
        return self.state

    def show(self):
        print(self.state, 'reward: ', self.reward, ', done: ', self.done)
