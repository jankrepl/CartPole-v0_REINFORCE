"""Solving the CartPole-v0 environment with the REINFORCE method """

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

import gym
import matplotlib.pyplot as plt

from foo import *

# PARAMETERS
number_of_episodes = 5000

step_size_initial = 1
step_size_decay = 1

# INITIALIZATION

evol = []
env = gym.make('CartPole-v0')
my_policy = Policy()
step_size = step_size_initial

for e in range(number_of_episodes):
    s_old = env.reset()
    done = False

    episode_database = []
    t = 0
    while not done:
        t += 1

        a = my_policy.choose_action(s_old)

        s_new, r, done, _ = env.step(a)

        episode_database.append(tuple((s_old, a)))

        s_old = s_new

    evol.append(t)
    if t != 200:
        print('Updating my policy...')
        for i, pair in enumerate(episode_database):
            estimated_return = t - i - 1
            my_policy.update(pair[0], pair[1], estimated_return, step_size)

        step_size *= step_size_decay

    print('Episode ' + str(e) + ' ended in ' + str(t) + ' time steps')
    print('Current step size: ' + str(step_size))

plt.plot(evol)
plt.show()
