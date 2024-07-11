import numpy as np
import os

from util.parameters import ParametersA2C
from train.train import A2Ctrainer
from helpers.a2c_helper import initialize_environment
from util.benchmarker import Utils

class A2Crunner:
    def __init__(self, env):
        self.env = env

    def run_experiment(self):
        params = ParametersA2C(self.env.env_name)
        trainer_a2c = A2Ctrainer()
        num_trials = params.num_trials

        state_dim, action_dim, action_offset = initialize_environment(self.env)
        self.env.reset()

        train_params = {
            'env': self.env,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'action_offset': action_offset,
            'gamma': params.config_A2C['gamma'],
            'actor_hidden_dim': params.config_A2C['actor_hidden_dim'],
            'critic_hidden_dim': params.config_A2C['critic_hidden_dim'],
            'value_dim': params.config_A2C['value_dim'],
            'alpha': params.config_A2C['alpha'],
            'beta': params.config_A2C['beta'],
            'num_training_episodes': params.config_A2C['num_training_episodes'],
            'num_batch_episodes': params.config_A2C['num_batch_episodes'],
            't_max': params.config_A2C['t_max'],
            'tau': params.config_A2C['tau'],
            'test_interval': params.config_A2C['test_interval'],
            'num_test_episodes': params.config_A2C['num_test_episodes']
        }

        Load_save_result = params.Load_save_result
        
        if Load_save_result is False or not os.path.isfile('all_train_returns.npy') or not os.path.isfile('all_test_returns.npy'):
            all_train_returns = []
            all_test_returns = []

            for trial in range(num_trials):
                print(f"Trial: {trial+1}")
                train_rewards, test_rewards = trainer_a2c.train(**train_params)
                all_train_returns.append(train_rewards)
                all_test_returns.append(test_rewards)
            np.save('all_train_returns.npy', all_train_returns)
            np.save('all_test_returns.npy', all_test_returns)
        else:
            all_train_returns = np.load('all_train_returns.npy')
            all_test_returns = np.load('all_test_returns.npy')

        utils = Utils()
        average_returns, max_return, max_return_ci, individual_returns = utils.benchmark_plot(all_train_returns, all_test_returns, train_params['test_interval'])
        print(f"Average Return: {average_returns}")
        print(f"Max Return: {max_return}")
        print(f"Max Return 95% CI: {max_return_ci}")
        print(f"Individual Returns: {individual_returns}")
        print("Completed experiment")

        self.env.close()