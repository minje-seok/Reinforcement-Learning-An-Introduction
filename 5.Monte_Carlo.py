import gymnasium as gym
import  matplotlib.pyplot as plt

EPISODES = 3

env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode='human')


for _ in range(EPISODES):
    state, _  = env.reset() # (The player's current sum, The value of dealer's one showing card(1-10 where 1 is ace), Whether the player holds a usable ace (0 or 1)
    done = False
    # 0: Stick, 1: Hit
    while not done:
        action = env.action_space.sample()
        print(action)
        next_state, reward, done, _, _ = env.step(action)

        print(next_state, reward, done)
        state = next_state

env.close()