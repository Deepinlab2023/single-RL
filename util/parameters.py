import math

class ParametersPPO:
    def __init__(self):
        self.nb_episodes = 1000  # or 500
        self.batch_size = 128  # or 64
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.layers1_num = 4  # CartPole state space dimension
        self.layers2_num = 64  # or 128, hidden_layer
        self.out_num = 2  # CartPole action space dimension
        self.lr = 3e-4  # learning rate for actor network
        self.lr_c = 0.001  # learning rate for critic network
        self.ep = 10
        self.t = 500  # or 1000 the max time steps
        self.training_times = 10
        self.save_episode = 50
        self.test_episode = 25
        self.test_trials = 10  # test 10 times and get the average result
        self.test_interval = 10  # test every 10 episodes
        self.num_trials = 5

class ParametersA2C:
    def __init__(self):
        self.num_trials = 5
        self.training_value = 2000
        self.batch_size = 10

        self.config_A2C = {
            'gamma': 0.99,
            'actor_hidden_dim': 256,
            'critic_hidden_dim': 256,
            'value_dim': 1,
            'alpha': 1e-3,
            'beta': 1e-3,
            'num_training_episodes': math.ceil(self.training_value / self.batch_size),
            'num_batch_episodes': self.batch_size,
            't_max': 1000,
            'tau': 0.005,
            'test_interval': 20,
            'num_test_episodes': 10
        }