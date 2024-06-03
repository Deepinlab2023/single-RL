import numpy as np
from agent.ppo import PPO
from train.train import PPOtrainer
from env.pong import PongEnv
from util.parameters import Parameters
from util.logger import Figure

def main_ppo():
    # Environment Initialization
    env_name = 'PongNoFrameskip-v4'
    #gpu = False
    params = Parameters()
    nb_episodes = params.nb_episodes  # 100000
    batch_size = params.batch_size  # 24576
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
    main_ppo()

