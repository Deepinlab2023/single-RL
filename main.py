import numpy as np
from agent.ppo import PPO
from train.train import PPOtrainer
from env.pong import PongEnv
from env.cartpole import CartPoleEnv
from util.parameters import Parameters
from util.logger import Figure
from util.benchmarker import Utils
import os

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
    
def run_experiment():
    env_name = 'CartPole-v1'
    env = CartPoleEnv(env_name)
    env.reset()

    params = Parameters()
    nb_episodes = params.nb_episodes
    batch_size = params.batch_size
    num_trials = params.num_trials

    Load_save_result = params.Load_save_result

    if Load_save_result is False or not os.path.isfile('all_train_returns.npy') or not os.path.isfile('all_test_returns.npy'):
        all_train_returns = []
        all_test_returns = []

        for trial in range(num_trials):
            print(f"Trial: {trial+1}")
            agent_ppo = PPO()
            trainer_ppo = PPOtrainer(env_name)

            train_rewards, test_rewards = trainer_ppo.train(env, agent_ppo, nb_episodes, batch_size)
            all_train_returns.append(train_rewards)
            all_test_returns.append(test_rewards)
        np.save('all_train_returns.npy', all_train_returns)
        np.save('all_test_returns.npy', all_test_returns)
    else:
        all_train_returns = np.load('all_train_returns.npy')
        all_test_returns = np.load('all_test_returns.npy')

    utils = Utils()
    average_returns, max_return, max_return_ci, individual_returns = utils.benchmark_plot(all_train_returns, all_test_returns, params.test_interval)
    print(f"Average Return: {average_returns}")
    print(f"Max Return: {max_return}")
    print(f"Max Return 95% CI: {max_return_ci}")
    print(f"Individual Returns: {individual_returns}")
    print("Completed experiment")

def main():
    run_experiment()

if __name__ == "__main__":
    main()
    #main_ppo()

