import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
from agent.ppo import PPO
from util.train import PPOtrainer
from env.pong import PongEnv
def main():
    # Environment Initialization
    env_name = 'PongNoFrameskip-v4'
    gpu = False
    nb_episodes = 128 #100000
    batch_size = 8 #24576
    env = PongEnv(env_name)
    env.reset()

    agent_ppo = PPO()
    trainer_ppo = PPOtrainer()

    #train
    result = trainer_ppo.train(env, agent_ppo, nb_episodes, batch_size)
    np.save('PPO_training_result.npy', result)
    print('training done')
    env.close()

if __name__ == "__main__":
    main()

