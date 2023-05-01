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

        # for policy evaluation
        self.V = np.zeros((self.row, self.col))
        self.next_V = np.zeros((self.row, self.col))

        # for in-place policy evaluation
        self.in_place_V = np.zeros((self.row, self.col))

        # for policy improvement
        self.Q = np.zeros((self.row, self.col, env.action_space))

        # for value iteration
        self.max_V = np.zeros((self.row, self.col))
        self.next_max_V = np.zeros((self.row, self.col))

    # Find the index of max value in array
    def find_max_indices(self, arr):
        max_val = np.max(arr)
        max_indices = np.where(arr == max_val)[0]
        return max_indices

    # Update policy greedily based on max value function
    def make_policy(self, max_indices):
        arr = [0, 0, 0, 0]
        if len(max_indices) > 1:
            for i in max_indices:
                arr[i] = 1 / int(len(max_indices))
        else:
            arr[max_indices[0]] = 1
        return arr

    # Visualize the current policy with arrows
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

        for i in range(self.row):
            print(arr[i])


    def policy_evaluation(self, update=False, iter_num=0, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function according to the given policy using 1-array.

        Args:
            update (bool): The flag, apply random policy everytime if 0, otherwise apply improved policy.
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.
            iter_num (int): The number of iterations, iterates until stopped by theta if its value is 0 (default),
                            otherwise iterates as many times as its value.

        Returns:
            value function (np.array): The computed value function according to the given policy.
        '''

        iter = 0
        
        # Repeat until the difference between the old and new value functions is less than theta
        while True:
            iter += 1
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    # Exclude calculating start and terminal state
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.V[i][j]
                    new_V = 0

                    # If the policy is not updating, then keep equiprobable random
                    if update == False:
                        self.policy[i][j] = [0.25, 0.25, 0.25, 0.25]

                    # Calculate a new value function according to the policy
                    for action, action_prob in enumerate(self.policy[i][j]):
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()

                        # new state-value function = action probability * (immediate reward + discount factor * next state-value function)
                        new_V += action_prob * (reward + gamma * self.V[x][y])
                        self.env.reset()

                    # Policy evaluation: Store computed new value function in another array, self.next_V
                    self.next_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))

            # Break the loop when the delta is smaller than theta or iter is reached to iter_num
            if delta < theta or iter == iter_num:
                print('Policy Evaluation - iter: {0}'.format(iter))
                print(self.next_V, '\n')
                break

            # Copy self.next_V to self.V which means that the state value-function is updated
            self.V = self.next_V.copy()
        return self.V
    
    def in_place_policy_evaluation(self, update=False, iter_num=0, gamma=1.0, theta=1e-10):
        '''
        Perform policy evaluation to compute the value function according to the given policy using only 1-array.

        Args:
            update (bool): The flag, apply random policy everytime if 0, otherwise apply improved policy.
            gamma (float): The discount factor, should be between 0 and 1.
            theta (float): The convergence threshold.
            iter_num (int): The number of iterations, iterates until stopped by theta if its value is 0 (default),
                            otherwise iterates as many times as its value.

        Returns:
            value function (np.array): The computed value function according to the given policy.
        '''

        iter = 0
        
        # Repeat until the difference between the old and new value functions is less than theta
        while True:
            iter += 1
            delta = 0
            for i in range(self.in_place_V.shape[0]):
                for j in range(self.in_place_V.shape[1]):
                    # Exclude calculating start and terminal state
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.in_place_V[i][j]
                    new_V = 0

                    # If the policy is not updating, then keep equiprobable random
                    if update == False:
                        self.policy[i][j] = [0.25, 0.25, 0.25, 0.25]

                    # Calculate a new value function according to the policy
                    for action, action_prob in enumerate(self.policy[i][j]):
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()
                        
                        # new state-value function = action probability * (immediate reward + discount factor * next state-value function)
                        new_V += action_prob * (reward + gamma * self.in_place_V[x][y])
                        self.env.reset()

                    # In-place policy evaluation: Store the computed new value function in same array, self.in_place_V
                    self.in_place_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))

            # Break the loop when the delta is smaller than theta or iter is reached to iter_num
            if delta < theta or iter == iter_num:
                print('Policy Evaluation(in-place) - iter: {0}'.format(iter))
                print(self.in_place_V, '\n')
                break

            # There are no syntax related to copying like policy evaluation

        return self.in_place_V

    def greedy_policy_improvement(self, in_place=False, show_policy=True, gamma=1.0):
        '''
        Perform greedy policy improvement to choose the best action-value function according to the given policy and update policy.

        Args:
            in_place (bool): The flag, determines whether policy evaluation is in_place method or not.
            show_policy (int): The flag, determines whether to show policy indicated by arrows.
            gamma (float): The discount factor, should be between 0 and 1.

        Returns:
            policy (np.array): A policy that contains a probability which is the maximal.
        '''
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                # Exclude calculating start and terminal state
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    continue

                # Calculate action-value function according to the policy
                Q = []
                for action, action_prob in enumerate(self.policy[i][j]):
                    self.env.create_grid(i, j)
                    next_state, reward, done = self.env.step(action)
                    x, y = self.env.get_current_state()

                    # action-value function = immediate reward + discount factor * next state-value function
                    if in_place == True:
                        Q.append(reward + gamma * self.in_place_V[x][y])
                    else:
                        Q.append(reward + gamma * self.V[x][y])
                    self.env.reset()

                self.Q[i][j] = Q

                # Update greedy policy according to calculated self.Q
                max_Q_indices = self.find_max_indices(self.Q[i][j]).tolist()
                self.policy[i][j] = self.make_policy(max_Q_indices)

        print('Greedy Policy Improvement')
        # Show the self.policy as an arrow to visualize easily not float
        if show_policy == 1:
            self.show_updated_policy(self.policy)
        
        return self.policy

    def policy_iteration(self, evaluation_num, update=False, show_policy=1, converge_num=3):
        '''
        Perform policy iteration until the policy converges.

        Args:
             evaluation_num (int): The number of iterations at policy evaluation, iterates until stopped by theta if its value is 0 (default),
                                   otherwise iterates as many times as its value.
             update (bool): The flag, apply random policy everytime if 0, otherwise apply improved policy.
             converge_num (int): The convergence threshold, is the number of iterations that no longer change at policy.

        Returns:
            value function (np.array): The computed value function according to the given policy.
        '''
        iter = 0
        no_change = 0

        while True:
            iter += 1
            # Save past policy to compare later
            past_policy = self.policy.copy()

            # Execute policy evaluation and policy improvement
            print('----- Policy Iteration - iter: {0} -----'.format(iter))
            self.policy_evaluation(update=update, iter_num=(iter if not update else evaluation_num))
            self.greedy_policy_improvement(show_policy=show_policy)
            print('')

            # Compare current policy and past policy
            if np.array_equal(past_policy, self.policy):
                no_change += 1
                print('* There are no change at policy for {0} times\n'.format(no_change))
            else:
                no_change = 0
                print('')

            if no_change == converge_num:
                break

        return self.V

    def value_iteration(self, converge_num=3, gamma=1):
        '''
        Perform value iteration until the value function converges.

        Args:
             converge_num (int): The convergence threshold, is the number of iterations that no longer change at value function.
             gamma (float): The discount factor, should be between 0 and 1.

        return:
            value function (np.array): The computed value function according to the given policy.
        '''

        iter = 0
        no_change = 0

        while True:
            iter += 1

            print('----- Value Iteration - iter: {0} -----'.format(iter))
            for i in range(self.max_V.shape[0]):
                for j in range(self.max_V.shape[1]):
                    # Exclude calculating start and terminal state
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    new_V = []

                    # Calculate a new max state-value function
                    for action, action_prob in enumerate(self.policy[i][j]):
                        self.env.create_grid(i, j)
                        next_state, reward, done = self.env.step(action)
                        x, y = self.env.get_current_state()

                        # state-value function = immediate reward + discount factor * next state-value function
                        new_V.append((reward + gamma * self.max_V[x][y]))
                        self.env.reset()

                    # new state-value function = max(all possible calculated state-value functions)
                    self.next_max_V[i][j] = np.max(new_V)

            # Compare current state-value function and past state-value function
            if np.array_equal(self.max_V, self.next_max_V):
                print(self.next_max_V, '\n')
                no_change += 1
                print('* There are no change at value function for {0} times'.format(no_change))
                print('\n')
            else:
                no_change = 0
                print(self.next_max_V, '\n')

            if no_change == converge_num:
                break

            # Copy self.next_max_V to self.max_V which means that the state value-function is updated
            self.max_V = self.next_max_V.copy()

        return self.max_V


agent = DP(env, row, col, random_policy)
# agent.policy_evaluation(iter_num=0) # converge at iter 426
# agent.in_place_policy_evaluation(iter_num=0) # converge at iter 272
# agent.greedy_policy_improvement(in_place=False) # Show updated policy based on value function performed in policy evaluation

# agent.policy_iteration(update=True, evaluation_num=0) # policy converges at iter 3
# agent.value_iteration() # value function converges at iter 3
