import random
import gym
import numpy as np
from PIL import Image
import torch as th
from torch.nn import functional as F
from torch import nn
from torch.distributions import Categorical

from agent.a2c import A2CActor
from critics.a2c import A2CCritic
from helpers.a2c_helper import to_tensor
from helpers.a2c_bp import BatchTraining
from util.parameters import Parameters
from util.logger import Figure
from util.a2c_logger import a2c_test

class PPOtrainer:
    def __init__(self, env_name):
        self.env_name = env_name

    def train(self, env, agent, nb_episodes, batch_size):
        tester = Figure()
        params = Parameters()
        opt = th.optim.Adam([
            {'params': agent.layers.parameters(), 'lr': params.lr},
            {'params': agent.critic.parameters(), 'lr': params.lr_c}
        ])
        #opt = th.optim.Adam(agent.layers.parameters(), lr=params.lr)
        #opt_c = th.optim.Adam(agent.critic.parameters(), lr=params.lr_c)
        reward_sum_running_avg = None
        reward_sum_running_avg_history = []
        training_results = []
        test_results = []

        for it in range(nb_episodes):
            d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
            state_val_history = []
            done_history = []
            episode_rewards = 0

            for ep in range(params.ep):
                obs, prev_obs = env.reset(), None
                obs = obs[0]
                for t in range(params.t):
                    d_obs = env.pre_process(obs, prev_obs)

                    with th.no_grad():
                        action, action_prob = agent(d_obs)

                    state_val = agent.forward_critic(d_obs)
                    prev_obs = obs
                    obs, reward, done, truncated, _ = env.step(agent.convert_action(action, self.env_name))

                    d_obs_history.append(d_obs)
                    action_history.append(action)
                    action_prob_history.append(action_prob)
                    reward_history.append(reward)
                    state_val_history.append(state_val)
                    done_history.append(done)

                    episode_rewards += reward

                    if done:
                        reward_sum = sum(reward_history[-t:])
                        reward_sum_running_avg = 0.99 * reward_sum_running_avg + 0.01 * reward_sum if reward_sum_running_avg else reward_sum
                        #print(
                        #    'Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (
                        #    it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                        reward_sum_running_avg_history.append(reward_sum_running_avg)
                        break

            training_results.append(episode_rewards / params.ep) # Average reward per episode

            # compute advantage
            R = 0
            discounted_rewards = []

            for r, d in zip(reward_history[::-1], done_history[::-1]):
                if self.env_name == 'Pong-v0':
                    if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
                if d is True:
                    R = 0  # if terminal, R=0
                R = r + agent.gamma * R
                discounted_rewards.insert(0, R)

            # Normalizing the rewards
            discounted_rewards = th.FloatTensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
            assert(len(discounted_rewards) == len(state_val_history))
            advantage_history = []
            for i_adv in range(len(discounted_rewards)):
                adv = discounted_rewards[i_adv] - state_val_history[i_adv]
                advantage_history.append(adv)
            assert(len(advantage_history) == len(discounted_rewards))

            # update policy
            for _ in range(params.training_times):
                idxs = random.sample(range(len(action_history)), batch_size)
                d_obs_batch = th.cat([d_obs_history[idx] for idx in idxs], 0)
                action_batch = th.LongTensor([action_history[idx] for idx in idxs])
                action_prob_batch = th.FloatTensor([action_prob_history[idx] for idx in idxs])
                #advantage_batch = th.FloatTensor([discounted_rewards[idx] for idx in idxs])
                advantage_batch = th.FloatTensor([advantage_history[idx] for idx in idxs])
                state_val_batch = th.FloatTensor([state_val_history[idx] for idx in idxs])
                discounted_rewards_batch = th.FloatTensor([discounted_rewards[idx] for idx in idxs])

                opt.zero_grad()
                loss_a = agent(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
                loss_c = agent.critic_loss(state_val_batch, discounted_rewards_batch)
                loss = loss_a + loss_c
                loss.backward()
                opt.step()

            if it % params.test_interval == 0:
                # test 10 times for more accurate results
                test_sum = 0
                for test_i in range(params.test_trials):
                    test_reward = tester.test(env, agent, self.env_name)
                    test_sum += test_reward
                test_average = test_sum / params.test_trials
                test_results.append(test_average)
                print('Training reward for episode %d: %.2f' % (it, test_average))

            if it % params.save_episode == 0:
                if it == 0:
                    th.save(agent.state_dict(), 'params.ckpt')
                else:
                    if test_average >= max(test_results[:-1]):
                        th.save(agent.state_dict(), 'params.ckpt')
                #pass

        return training_results, test_results

class DQNtrainer:
    def train(self, env, agent, nb_episodes, batch_size):
        print("DQNtrainer")

class A2Ctrainer:
    def __init__(self, env_name):
        self.env_name = env_name

    def train(self, env, state_dim, action_dim, action_offset ,gamma, actor_hidden_dim, critic_hidden_dim,
            value_dim, alpha, beta, num_training_episodes ,num_batch_episodes, t_max, tau,
            test_interval, num_test_episodes):
        
        actor = A2CActor(state_dim, action_dim, actor_hidden_dim)
        critic = A2CCritic(state_dim, critic_hidden_dim, value_dim)

        optimizer_actor = th.optim.Adam(actor.parameters(), lr=alpha)
        optimizer_critic = th.optim.Adam(critic.parameters(), lr=beta)

        episode_rewards = []
        test_rewards = []

        episode = 0

        for te in range(num_training_episodes):
            batch_buffer = []
            batch_rtrns = []
            for e in range(num_batch_episodes):
                state = env.reset()
                prev_state = None
                total_reward = 0
                done = False
                t = 0
                buffer = []

                while not done and (t < t_max):

                    if 'PongNoFrameskip-v4' in self.env_name:
                        state_tensor = env.pre_process(state, prev_state)
                    else:
                        state_tensor = to_tensor(state)
                    logits = actor(state_tensor)
                    action, dist = actor.action_sampler(logits)
                    converted_action = action.item() + action_offset
                    next_state, reward, done, info = env.step(converted_action)
                    if 'PongNoFrameskip-v4' in self.env_name:
                        next_state_tensor = env.pre_process(next_state, state)
                    else:
                        next_state_tensor = to_tensor(next_state)

                    buffer.append((state_tensor, action, reward, next_state_tensor))
                    prev_state = state
                    state = next_state
                    total_reward += reward
                    t+=1

                episode_rewards.append(total_reward)
                episode += 1

                rtrns = []
                if done:
                    R = 0
                else:
                    R = critic(to_tensor(state)).detach().item()

                for _, _, reward, _ in reversed(buffer):
                    R = reward + gamma * R
                    rtrns.append(R)
                rtrns.reverse()

                batch_buffer.extend(buffer)
                batch_rtrns.extend(rtrns)

                if episode % test_interval == 0:
                    actor_state = actor.state_dict()

                    test_reward = a2c_test(env, actor, num_test_episodes, t_max, action_offset)
                    test_rewards.append(test_reward)

                    actor.load_state_dict(actor_state)

            batch_training = BatchTraining()
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_rtrns = batch_training.collate_batch(batch_buffer, batch_rtrns)

            # Update Critic
            optimizer_critic.zero_grad()
            V = critic(batch_states)
            # Critic Loss
            critic_loss = (batch_rtrns - V).pow(2).mean()
            critic_loss.backward()
            optimizer_critic.step()

            # Update Actor
            optimizer_actor.zero_grad()
            logits = actor(batch_states)
            action_dist = Categorical(logits=logits)
            log_probs = action_dist.log_prob(batch_actions)
            entropies = action_dist.entropy()
            V = critic(batch_states).detach()
            advantage = (batch_rtrns - V).detach()

            # Actor Loss
            actor_loss = -(log_probs * advantage).mean() - tau * entropies.mean()
            actor_loss.backward()
            optimizer_actor.step()

        print("Algorithm done")
        return episode_rewards, test_rewards
