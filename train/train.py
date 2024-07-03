import random
import gym
import numpy as np
from PIL import Image
import torch as th
from torch.nn import functional as F
from torch import nn
from util.parameters import Parameters
from util.logger import Figure

class PPOtrainer:
    def __init__(self, env_name):
        self.env_name = env_name

    def train(self, env, agent, nb_episodes, batch_size):
        tester = Figure()
        params = Parameters()
        opt = th.optim.Adam(agent.parameters(), lr=params.lr)
        reward_sum_running_avg = None
        reward_sum_running_avg_history = []
        training_results = []
        test_results = []

        for it in range(nb_episodes):
            d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
            episode_rewards = 0

            for ep in range(params.ep):
                obs, prev_obs = env.reset(), None
                for t in range(params.t):
                    d_obs = env.pre_process(obs, prev_obs)

                    with th.no_grad():
                        action, action_prob = agent(d_obs)

                    prev_obs = obs
                    obs, reward, done, _ = env.step(agent.convert_action(action, self.env_name))

                    d_obs_history.append(d_obs)
                    action_history.append(action)
                    action_prob_history.append(action_prob)
                    reward_history.append(reward)

                    episode_rewards += reward

                    if done:
                        reward_sum = sum(reward_history[-t:])
                        reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                        print(
                            'Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (
                            it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                        reward_sum_running_avg_history.append(reward_sum_running_avg)
                        break

            training_results.append(episode_rewards / params.ep) # Average reward per episode

            # compute advantage
            R = 0
            discounted_rewards = []

            for r in reward_history[::-1]:
                if self.env_name == 'Pong-v0':
                   if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum 
                R = r + agent.gamma * R
                discounted_rewards.insert(0, R)

            discounted_rewards = th.FloatTensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

            # update policy
            for _ in range(params.training_times):
                idxs = random.sample(range(len(action_history)), batch_size)
                d_obs_batch = th.cat([d_obs_history[idx] for idx in idxs], 0)
                action_batch = th.LongTensor([action_history[idx] for idx in idxs])
                action_prob_batch = th.FloatTensor([action_prob_history[idx] for idx in idxs])
                advantage_batch = th.FloatTensor([discounted_rewards[idx] for idx in idxs])

                opt.zero_grad()
                loss = agent(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
                loss.backward()
                opt.step()

            if it % params.test_interval == 0:
                test_reward = tester.test(env, agent, self.env_name)
                test_results.append(test_reward)
                print('Training reward for episode %d: %.2f' % (it, test_reward))

            if it % params.save_episode == 0:
                # th.save(agent.state_dict(), 'params.ckpt')
                pass

        return training_results, test_results

class DQNtrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("DQNtrainer")

class A2Ctrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("A2Ctrainer")