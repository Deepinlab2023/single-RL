class Parameters:
    def __init__(self):
        self.nb_episodes = 1000
        self.batch_size = 64
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.layers1_num = 4  # CartPole state space dimension
        self.layers2_num = 128
        self.out_num = 2  # CartPole action space dimension
        self.lr = 3e-4
        self.ep = 10
        self.t = 1000
        self.training_times = 10
        self.save_episode = 50
        self.test_episode = 25
        self.test_interval = 10
