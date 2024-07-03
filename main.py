import numpy as np
from agent.ppo import PPO
from train.train import PPOtrainer
from env.pong import PongEnv
from env.cartpole import CartPoleEnv
from util.parameters import Parameters
from util.logger import Figure
from util.benchmarker import Utils


def run_experiment():
    num_trials = 5

    env_name = 'CartPole-v1'
    env = CartPoleEnv(env_name)
    env.reset()

    params = Parameters()
    nb_episodes = params.nb_episodes
    batch_size = params.batch_size

    all_train_returns = []
    all_test_returns = []

    for trial in range(num_trials):
        print(f"Trial: {trial+1}")
        agent_ppo = PPO()
        trainer_ppo = PPOtrainer(env_name)

        train_rewards, test_rewards = trainer_ppo.train(env, agent_ppo, nb_episodes, batch_size)
        all_train_returns.append(train_rewards)
        all_test_returns.append(test_rewards)

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

