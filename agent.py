import numpy as np
import random

from env import grid
from env.grid import GridEnv

row = 4
col = 4
env = GridEnv(row, col)
random_policy = np.full((row, col, env.action_space), [[0.25, 0.25, 0.25, 0.25]])
random_policy[0][0] = [0, 0, 0, 0]
random_policy[-1][-1] = [0, 0, 0, 0]

class DP:
    def __init__(self, env, row, col, policy):
        self.env = env
        self.row = row
        self.col = col
        self.policy = policy
        self.V = np.zeros((self.row, self.col))
        self.next_V = self.V.copy()

        self.in_place_V = np.zeros((self.row, self.col))

        self.Q = np.zeros((self.row, self.col, env.action_space))


    def find_max_indices(self, arr):
        max_val = np.max(arr)
        max_indices = np.where(arr == max_val)[0]
        return max_indices

    def choose_random_value(self, arr):
        random_index = random.randint(0, len(arr) - 1)
        return arr[random_index]

    def make_policy(self, max_indices):
        arr = [0, 0, 0, 0]
        if len(max_indices) > 1:
            for i in max_indices:
                arr[i] = 1 / int(len(max_indices))
        else:
            arr[max_indices[0]] = 1
        return arr

    def show_updated_policy(self, policy):
        shape = (self.row, self.col, 1)
        arr = np.zeros(shape).tolist()

        for i in range(self.row):
            for j in range(self.col):
                if (i == 0 and j == 0) or (i == env.action_space - 1 and j == env.action_space - 1):
                    continue

                if policy[i][j][0] != 0:
                    arr[i][j].insert(-1, '∧')
                if policy[i][j][1] != 0:
                    arr[i][j].insert(-1, '∨')
                if policy[i][j][2] != 0:
                    arr[i][j].insert(-1, '<')
                if policy[i][j][3] != 0:
                    arr[i][j].insert(-1, '>')

                arr[i][j].remove(0)

        for i in range(len(arr)):
            print(arr[i])


    def policy_evaluation(self, update = 0, iter_num=0, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function for a given policy using 1-array.

        Args:
            env (class): The environment to get element such as state and reward.
            policy (np.array): A 2D array representing the policy (same shape as env's state)
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.
            iter_num (int): The number of iterations, If it is 0 (default), it iterates until stopped by theta
                            or iterates as many times as its value.

        Returns:
            value function (np.array): The computed value function for given policy.
        '''

        iter = 0
        
        # Repeat until the difference between the old and new value functions is less than theta.
        while True:
            iter += 1
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    # Calculate excluding start and terminal state.
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.V[i][j]
                    new_V = 0

                    if update == 0:
                        self.policy[i][j] = [0.25, 0.25, 0.25, 0.25]

                    # Calculated a new value function according to the policy.
                    # At GridEnv, policy is equiprobable random policy (all actions equally likely).
                    for action, action_prob in enumerate(self.policy[i][j]): # self.policy[i][j].astype(np.float64)
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()

                        # action probability * (immediate reward + discount factor * next state's value function)
                        new_V += action_prob * (reward + gamma * self.V[x][y])
                        self.env.reset()

                    # Policy evaluation: Store computed new value function to self.next_V.
                    self.next_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))
            
            # Copy self.next_V to the existing self.V.
            self.V = self.next_V.copy()

            # If the delta is smaller than theta, the loop is stopped with break.
            if delta < theta or iter == iter_num:
                print('Policy Evaluation - iter: {0}'.format(iter))
                print(self.next_V, '\n')
                break

        return self.V
    
    def in_place_policy_evaluation(self, iter_num=0, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function for a given policy using only 1-array.

        Args:
            env (class): The environment to get element such as state and reward.
            policy (np.array): A 2D array representing the policy (same shape as env's state)
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.
            iter_num (int): The number of iterations, If it is 0 (default), it iterates until stopped by theta
                            or iterates as many times as its value.

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
                    for action, action_prob in enumerate(self.policy[i][j]):
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()
                        
                        # action probability * (immediate reward + discount factor * next state's value function)
                        new_V += action_prob * (reward + gamma * self.in_place_V[x][y])
                        self.env.reset()

                    # In-place policy evaluation: Update the computed new value function to self.in_place_V.
                    self.in_place_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))
                    
            # If the delta is smaller than theta, the loop is stopped with break.
            if delta < theta or iter == iter_num:
                print('Policy Evaluation(in-place) - iter: {0}'.format(iter))
                print(self.in_place_V, '\n')
                break

        return self.in_place_V

    def greedy_policy_improvement(self, gamma=1.0):
        '''
        Perform greedy policy improvement to choose the best action-value function for a given policy and update policy.

        Returns:
            policy (np.array): A policy that contains a probability value for the max action value..
        '''
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                # Calculate excluding start and terminal state.
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    continue

                # Calculate action-value function according to the policy.
                Q = []
                for action, action_prob in enumerate(self.policy[i][j]):
                    self.env.create_grid(i, j)
                    next_state, reward, done = self.env.step(action)
                    x, y = self.env.get_current_state()

                    # action-value function = immediate reward + discount factor * next state's value function
                    Q.append(reward + gamma * self.V[x][y])
                    self.env.reset()

                self.Q[i][j] = Q

                # Update greedy policy according to calculated self.Q.
                max_Q_indices = self.find_max_indices(self.Q[i][j]).tolist()
                self.policy[i][j] = self.make_policy(max_Q_indices)

        print('Greedy Policy Improvement')
        self.show_updated_policy(self.policy)
        
        return self.policy

<<<<<<< HEAD
        return self.policy
=======
>>>>>>> 39d070e4a64daa410a3c03ba2ac0537b3185774e

    def policy_iteration(self, num=3):
        iter = 0
        no_change = 0

        while True:
            iter += 1
            past_policy = self.policy.copy()

            print('*** Policy Iteration - iter: {0} ***'.format(iter))
            self.policy_evaluation(update=0, iter_num=1)
            self.greedy_policy_improvement()

            if np.array_equal(past_policy, self.policy):
                no_change += 1
                print('no_change: {0}\n\n'.format(no_change))
            else:
                print('\n')

            if no_change == num:
                break

    def value_iteration(self):
        # Find current policy and index of max action.
        max_indices = self.find_max_indices(self.policy[i][j]).tolist()
        policy = self.make_policy(max_indices)
        max_V = self.choose_random_value(policy)

agent = DP(env, row, col, random_policy)
# agent.policy_evaluation()
# agent.greedy_policy_improvement()
<<<<<<< HEAD
agent.policy_iteration()
=======
agent.policy_iteration()
>>>>>>> 39d070e4a64daa410a3c03ba2ac0537b3185774e
