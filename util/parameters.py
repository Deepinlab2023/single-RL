class Parameters:
    def __init__(self,):
        self.nb_episodes = 4000  # 100000, 4000 is enough for Pong
        self.batch_size = 24576  # 24576, 1536
        # PPO
        self.gamma = 0.99
        self.eps_clip = 0.1
        self.layers1_num = 6000
        self.layers2_num = 512
        self.out_num = 2
        # training
        self.lr = 1e-3
        self.ep = 10
        self.t = 190000
        self.training_times = 5
        self.save_episode = 5  # save model every 5 episodes
        self.test_episode = 10  # test every 10 episodes
