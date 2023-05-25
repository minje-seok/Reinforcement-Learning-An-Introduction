from collections import defaultdict

import gymnasium as gym
import  matplotlib.pyplot as plt

EPISODES = 3

env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode='human')

def first_visit_MC(states, actions, rewards):
    # First, we initialize the empty value table as a dictionary for storing the values of each state
    value_table = defaultdict(float)
    N = defaultdict(int)
    returns = 0

    for t in range(len(states) - 1, -1, -1):
        R = rewards[t]
        S = states[t]

        returns += R

        # Now to perform first visit MC, we check if the episode is visited for the first time, if yes,
        # we simply take the average of returns and assign the value of the state as an average of returns

        if S not in states[:t]:
            N[S] += 1
            value_table[S] += (returns - value_table[S]) / N[S]

    return value_table

# Define fixed policy function. If the score is greater than or equal to 20 we stand (0) else we hit (1)
def perform_policy(state):
    score, dealer_score, usable_ace = state
    return 0 if score >= 20 else 1

for _ in range(EPISODES):
    # Initialize list to store state, action, reward
    states, actions, rewards = [], [], []

    state, _ = env.reset() # (The player's current sum, The value of dealer's one showing card(1-10 where 1 is ace), Whether the player holds a usable ace (0 or 1)

    # 0: Stick, 1: Hit
    while True:
        states.append(state)

        # Select an action according to policy

        action = perform_policy(state)
        actions.append(action)

        # Perform the selected action in the env according to policy and get next state, reward, done
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)

        state = next_state

        # Break if the state is a terminal state
        if done:
            break

    # Perform first-visit MC to estimate value function
    print(states)
    print(actions)
    print(rewards)

env.close()