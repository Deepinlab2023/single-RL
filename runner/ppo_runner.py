import os
import numpy as np
from util.benchmarker import Utils
from util.parameters import ParametersPPO
from agent.ppo import PPO
from train.train import PPOtrainer

class PPOrunner():
    def __init__(self, env):
        self.env = env
        
    def run_experiment(self):
        params = ParametersPPO()
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
                trainer_ppo = PPOtrainer(self.env.env_name)

                train_rewards, test_rewards = trainer_ppo.train(self.env, agent_ppo, nb_episodes, batch_size)
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

