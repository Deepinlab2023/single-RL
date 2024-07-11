import torch as th
import numpy as np
from env.cartpole import CartPoleEnv
from env.pong import PongEnv

# Helper function to convert numpy arrays to tensors
def to_tensor(x):
    if isinstance(x, np.ndarray):
        return th.from_numpy(x).float()
    return x

def initialize_environment(env):
    if 'PongNoFrameskip-v4' in env.env_name:
        state_dim = 6000
        action_dim = 2
        action_offset = 2
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        action_offset = 0

    return state_dim, action_dim, action_offset