import numpy as np

from env import grid
from env.grid import GridEnv

env = GridEnv()
random_policy = [0.25, 0.25, 0.25, 0.25]

class DP:
    def __init__(self):
        self.V = np.zeros((4, 4))
        self.next_V = self.V.copy()

        self.in_place_V = np.zeros((4, 4))

    def policy_evaluation(self, env, policy, gamma=1.0, theta=1e-10):
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
        # Repeat until the difference between the old value function and the new value function is smaller than theta, the specified threshold.
        while True:
            iter += 1
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    # Excluding the start state and terminal state, the value function is calculated.
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.V[i][j]
                    new_V = 0
                    # A new value function is calculated according to the probability of each action.
                    for action, action_prob in enumerate(policy):
                        env.create_grid(i, j)
                        next_state, reward, done = env.step(action)
                        x, y = env.get_current_state()
                        new_V += action_prob * (reward + gamma * self.V[x][y])
                        env.reset()
                    self.next_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))
            self.V = self.next_V.copy()
            
            # If the delta is smaller than theta, the loop is stopped with break.
            if delta < theta:
                print('Policy Evaluation - iter:{0}'.format(iter))
                print(self.next_V, '\n')
                break

        return self.next_V
    def in_place_policy_evaluation(self, env, policy, gamma=1.0, theta=1e-10):
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
        while True:
            iter += 1
            delta = 0
            for i in range(self.in_place_V.shape[0]):
                for j in range(self.in_place_V.shape[1]):
                    if (i == 0 and j == 0) or (i == 3 and j == 3):
                        continue

                    old_V = self.in_place_V[i][j]
                    new_V = 0
                    for action, action_prob in enumerate(policy):
                        env.create_grid(i, j)
                        next_state, reward, done = env.step(action)
                        x, y = env.get_current_state()
                        new_V += action_prob * (reward + gamma * self.in_place_V[x][y])
                        env.reset()
                    self.in_place_V[i][j] = new_V
                    delta = max(delta, abs(old_V - new_V))

            if delta < theta:
                print('Policy Evaluation(in-place) - iter:{0}'.format(iter))
                print(self.in_place_V, '\n')
                break

        return self.in_place_V

    def policy_improvement(self):

        return


agent = DP()
agent.policy_evaluation(env, random_policy)
agent.in_place_policy_evaluation(env, random_policy)
