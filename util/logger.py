import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
from util.parameters import Parameters

class Figure:
    def test(self, env, agent):  # test for the whole episode
        obs, prev_obs = env.reset(), None  # obs is a tuple and prev_obs is None
        reward_sum = 0
        reward_history = []
        params = Parameters()
        for t in range(params.t):
            # env.render()

            # d_obs = agent.pre_process(np.array(obs), np.array(prev_obs))
            d_obs = env.pre_process(obs, prev_obs)

            with torch.no_grad():
                action, action_prob = agent(d_obs)

            prev_obs = obs  # if obs is a tuple, prev_obs is a tuple; if obs is a matrix, prev_obs is a matrix
            obs, reward, done, _, info = env.step(
                agent.convert_action(action))  # now obs is outputted as a matrix

            reward_sum += reward
            reward_history.append(reward)

            if done:
                break

        #print('test')
        #print(reward_sum)
        return reward_sum

    def figure(self):  # draw the figures
        print('figure')