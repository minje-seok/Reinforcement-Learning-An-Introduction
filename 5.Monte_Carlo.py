from collections import defaultdict
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
import gymnasium as gym

import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use('ggplot')

EPISODES = 10000
env = gym.make("Blackjack-v1", natural=False, sab=False) # render_mode='human'

class CustomBlackjackEnv(BlackjackEnv):
    def __init__(self, initial_state=None):
        super().__init__()
        self.initial_state = initial_state

    def reset(self, init_state=False):
        if init_state is not None:
            self.initial_state = init_state
        return super().reset()

    def _reset(self):
        if self.initial_state is not None:
            self.player, self.dealer, _ = self.initial_state
            self.initial_state = None
        return super()._reset()

class MC:
    def __init__(self):
        # Initialize the empty value table as a dictionary for storing the values of each state
        # Using collections.defaultdict can specify the value according to each key, default value of any key is 0
        self.V = defaultdict(float) # state-value function
        self.N = defaultdict(int) # number of occurrences of each state
        self.Q = defaultdict(float)


    def execute_first_visit(self, states, actions, rewards):
        returns = 0

        # Calculate returns from end to beginning of the single EPISODE
        for t in range(len(states) - 1, 0, -1):
            R = rewards[t]
            S = states[t]

            returns += R # Accumulated reward

            # Check if the episode is visited for the first time,
            # if yes, take the average of returns and assign the value of the state as an average of returns
            if S not in states[:t]:
                self.N[S] += 1
                self.V[S] += (returns - self.V[S]) / self.N[S]

    def execute_exploring_starts(self, states, actions, rewards):
        returns = 0

        # Calculate returns from end to beginning of the single EPISODE
        for t in range(len(states) - 1, 0, -1):
            R = rewards[t]
            S = states[t]

            returns += R # Accumulated reward

            # Check if the episode is visited for the first time,
            # if yes, take the average of returns and assign the value of the state as an average of returns
            if S not in states[:t]:
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


    def first_visit_MC(self):
        agent = MC()

        for _ in range(EPISODES):
            # Initialize list to store state, action, reward during each EPISODE
            states, actions, rewards = [], [], []
            state, _ = env.reset()

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
            agent.execute_first_visit(states, actions, rewards)

        env.close()

        fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
        axes[0].set_title('value function without usable ace')
        axes[1].set_title('value function with usable ace')
        agent.plot_blackjack(agent.V, axes[0], axes[1])

    def MC_exploring_starts(self):
        agent = MC()

        for _ in range(EPISODES):
            # Initialize list to store state, action, reward during each EPISODE
            states, actions, rewards = [], [], []
            state, _ = env.reset()
            initial_state = (15, 10, False)
            obs = env.reset(init_state=initial_state)
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
            agent.execute_exploring_starts(states, actions, rewards)

        env.close()

        fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
        axes[0].set_title('value function without usable ace')
        axes[1].set_title('value function with usable ace')
        agent.plot_blackjack(agent.V, axes[0], axes[1])

agent = MC()
agent.first_visit_MC()
