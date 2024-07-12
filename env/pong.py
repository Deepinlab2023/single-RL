import gym
import numpy as np
import torch

class PongEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def state_to_tensor(self, It):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector. See Karpathy's post: http://karpathy.github.io/2016/05/31/rl/ """
        if It is None:
        #if len(I) == 0:
            return torch.zeros(1, 6000)
        if len(It) == 2:  # It is a tuple or a matrix
            I = It[0]
        else:
            I = It
        I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2,::2,0] # downsample by factor of 2.
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, x, prev_x):
        aa = self.state_to_tensor(x)
        bb = self.state_to_tensor(prev_x)
        return aa - bb
