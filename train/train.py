import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
from util.parameters import Parameters
from util.logger import Figure

class PPOtrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        tester = Figure()
        params = Parameters()
        opt = torch.optim.Adam(agent.parameters(), lr=params.lr)
        reward_sum_running_avg = None
        reward_sum_running_avg_history = []
        training_results = []
        for it in range(nb_episodes):
            d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
            for ep in range(params.ep):
                obs, prev_obs = env.reset(), None  # obs is a tuple and prev_obs is None
                for t in range(params.t):
                    # env.render()

                    # d_obs = agent.pre_process(np.array(obs), np.array(prev_obs))
                    d_obs = env.pre_process(obs, prev_obs)

                    with torch.no_grad():
                        action, action_prob = agent(d_obs)

                    prev_obs = obs  # if obs is a tuple, prev_obs is a tuple; if obs is a matrix, prev_obs is a matrix
                    obs, reward, done, _, info = env.step(
                        agent.convert_action(action))  # now obs is outputted as a matrix

                    d_obs_history.append(d_obs)
                    action_history.append(action)
                    action_prob_history.append(action_prob)
                    reward_history.append(reward)

                    if done:
                        reward_sum = sum(reward_history[-t:])
                        reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                        print(
                            'Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (
                            it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                        reward_sum_running_avg_history.append(reward_sum_running_avg)
                        break

            # compute advantage
            R = 0
            discounted_rewards = []

            for r in reward_history[::-1]:
                if r != 0: R = 0  # scored/lost a point in pong, so reset reward sum
                R = r + agent.gamma * R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.FloatTensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

            # update policy
            for _ in range(params.training_times):
                #batch_size = 24576
                idxs = random.sample(range(len(action_history)), batch_size)
                d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
                action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
                action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
                advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])

                opt.zero_grad()
                loss = agent(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
                loss.backward()
                opt.step()

            if it % params.test_episode == 0:  # test every 10 episodes
                training_reward = tester.test(env, agent)
                print('Training reward for episode %d: %.2f' % (it, training_reward))
                training_results.append(training_reward)

            if it % params.save_episode == 0:  # save every 5 episodes
                torch.save(agent.state_dict(), 'params.ckpt')

        #return reward_sum_running_avg_history, training_results
        return training_results

class DQNtrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("DQNtrainer")

class A2Ctrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("A2Ctrainer")