import gym
import numpy as np
import torch as th

class CartPoleEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def pre_process(self, state, _):
        return th.FloatTensor(state).unsqueeze(0)