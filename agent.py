import numpy as np
import random

from env import grid
from env.grid import GridEnv

row = 4
col = 4
env = GridEnv(row, col)
random_policy = np.full((row, col, env.action_space), [[0.25, 0.25, 0.25, 0.25]])
ppp = [0.25, 0.25, 0.25, 0.25]

class DP:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.V = np.zeros((4, 4))
        self.next_V = self.V.copy()

        self.in_place_V = np.zeros((4, 4))

    def choose_random_value(self, arr):
        random_index = random.randint(0, len(arr) - 1)
        return arr[random_index]

    def policy_evaluation(self, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function for a given policy using 1-array.

        Args:
            env (class): The environment to get element such as state and reward.
            policy (np.array): A 2D array representing the policy (same shape as env's state)
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.

        Returns:
            value function (np.array): The computed value function for given policy.
        '''

        iter = 0
        # Repeat until the difference between the old and new value functions is less than theta.
        while True:
            iter += 1
            delta = 0
            print('---------------------------------------------', iter)
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    # Calculate excluding start and terminal state.
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.V[i][j]
                    new_V = 0
                    t = self.policy.copy()

                    # # Find index of max action probability in policy.
                    # max_val = np.max(self.policy[i][j])
                    # max_indices = np.where(self.policy[i][j] == max_val)[0]

                    max_val = np.max(t[i][j])
                    max_indices = np.where(t[i][j] == max_val)[0]

                    p = [0, 0, 0, 0]
                    # If number of max action is more than 1, choose randomly.
                    if len(max_indices) >= 1:
                        for i in max_indices.tolist():
                            p[i] = 1 / int(len(max_indices))
                    else:
                        p[max_indices] = 1


                    # Calculated a new value function according to the policy.
                    # At GridEnv, policy is equiprobable random policy (all actions equally likely).
                    for action, action_prob in enumerate(self.policy[i][j].astype(np.float64)): #
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()

                        # action probability * (immediate reward + discount factor * next state's value function)
                        new_V += action_prob * (reward + gamma * self.V[x][y])
                        # print(reward, action, action_prob, self.V[x][y])
                        self.env.reset()
                    # print(new_V)
                    
                    # Policy evaluation: Store computed new value function to self.next_V.
                    self.next_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))
            
            # Copy self.next_V to the existing self.V.
            self.V = self.next_V.copy()

            # If the delta is smaller than theta, the loop is stopped with break.
            if delta < theta:
                print('Policy Evaluation - iter:{0}'.format(iter))
                print(self.next_V, '\n')
                break

        return self.next_V
    
    def in_place_policy_evaluation(self, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function for a given policy using only 1-array.

        Args:
            env (class): The environment to get element such as state and reward.
            policy (np.array): A 2D array representing the policy (same shape as env's state)
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.

        Returns:
            value function (np.array): The computed value function for given policy.
        '''
        iter = 0
        # Repeat until the difference between the old and new value functions is less than theta.
        while True:
            iter += 1
            delta = 0
            for i in range(self.in_place_V.shape[0]):
                for j in range(self.in_place_V.shape[1]):
                    # Calculate excluding start and terminal state.
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.in_place_V[i][j]
                    new_V = 0
                    
                    # Calculated a new value function according to the policy. 
                    # At GridEnv, policy is equiprobable random policy (all actions equally likely).
                    for action, action_prob in enumerate(self.policy):
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()
                        
                        # action probability * (immediate reward + discount factor * next state's value function)
                        new_V += action_prob * (reward + gamma * self.in_place_V[x][y])
                        self.env.reset()
                                                
                    # In-place policy evaluation: Update the computed new value function to self.V.
                    self.in_place_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))
                    
            # If the delta is smaller than theta, the loop is stopped with break.
            if delta < theta:
                print('Policy Evaluation(in-place) - iter:{0}'.format(iter))
                print(self.in_place_V, '\n')
                break

        return self.in_place_V

    def policy_improvement(self):
        V = self.in_place_policy_evaluation()

        return

agent = DP(env, random_policy)
agent.policy_evaluation()
# agent.in_place_policy_evaluation()
