import torch as th
from util.parameters import ParametersPPO

class Figure:
    def test(self, env, agent, env_name):
        obs, prev_obs = env.reset(), None
        obs = obs[0]
        reward_sum = 0
        reward_history = []
        params = ParametersPPO()
        for t in range(params.t):
            d_obs = env.pre_process(obs, prev_obs)

            with th.no_grad():
                action, action_prob = agent(d_obs)

            prev_obs = obs
            obs, reward, done, truncated, _ = env.step(agent.convert_action(action, env_name))

            reward_sum += reward
            reward_history.append(reward)

            if done:
                break

        return reward_sum

    def figure(self):  # draw the figures
        print('figure')
