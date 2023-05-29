import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use('ggplot')

EPISODES = 1000000
env = gym.make("Blackjack-v1", natural=False, sab=False) # render_mode='human'

class MC:
    def __init__(self):
        self.V = defaultdict(float)
        self.N = defaultdict(int)

    def first_visit_MC(self, states, actions, rewards):
        # First, we initialize the empty value table as a dictionary for storing the values of each state
        # Using collections.defaultdict can specify the value according to each key, if int(), default value of any key is 0
        returns = 0

        # Calculate returns from end to beginning of EPISODE
        for t in range(len(states) - 1, 0, -1):
            R = rewards[t] # last
            S = states[t] #

            returns += R

            # Check if the episode is visited for the first time,
            # if yes, take the average of returns and assign the value of the state as an average of returns
            if S not in states[:t]:
        #         self.V[S] += returns
        #         self.N[S] += 1
        #
        # for S in self.V.keys():
        #     self.V[S] = self.V[S] / self.N[S]
                self.N[S] += 1
                self.V[S] += (returns - self.V[S]) / self.N[S]

    # Plot the state-value function
    def plot_blackjack(self, V, ax1, ax2):
        player_sum = np.arange(12, 21 + 1)
        dealer_show = np.arange(1, 10 + 1)
        usable_ace = np.array([False, True])
        state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

        for i, player in enumerate(player_sum):
            for j, dealer in enumerate(dealer_show):
                for k, ace in enumerate(usable_ace):
                    state_values[i, j, k] = V[player, dealer, ace]

        X, Y = np.meshgrid(player_sum, dealer_show)

        ax1.plot_wireframe(X, Y, state_values[:, :, 0])
        ax2.plot_wireframe(X, Y, state_values[:, :, 1])

        for ax in ax1, ax2:
            ax.set_zlim(-1, 1)
            ax.set_ylabel('player sum')
            ax.set_xlabel('dealer showing')
            ax.set_zlabel('state-value')

        plt.show()

    # Define fixed policy function. If the score is greater than or equal to 20 we stand (0) else we hit (1)
    def fixed_policy(self, state):
        score, dealer_score, usable_ace = state
        return 0 if score >= 20 else 1

agent = MC()

for _ in range(EPISODES):
    # Initialize list to store state, action, reward during each EPISODE
    states, actions, rewards = [], [], []
    returns = 0

    state, _ = env.reset() # (The player's current sum, The value of dealer's one showing card(1-10 where 1 is ace), Whether the player holds a usable ace (0 or 1)

    # 0: Stick, 1: Hit
    while True:
        states.append(state)

        # Select an action according to policy
        action = agent.fixed_policy(state)
        actions.append(action)

        # Perform the selected action in the env according to policy and get next state, reward, done
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)

        state = next_state

        # Break if the state is a terminal state
        if done:
            break

    # Perform first-visit MC to estimate value function
    agent.first_visit_MC(states, actions, rewards)
# print(agent.V.popitem(), '\n')

env.close()

fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')
agent.plot_blackjack(agent.V, axes[0], axes[1])

