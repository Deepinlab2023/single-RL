import argparse

# Import envs and runners here
from runner.ppo_runner import PPOrunner
from runner.a2c_runner import A2Crunner
from env.cartpole import CartPoleEnv
from env.pong import PongEnv


def main():
    parser  = argparse.ArgumentParser(description = "Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, required=True, help='The environment to run. Choose from "pong", or "cartpole".')
    parser.add_argument('--algo', type=str, required=True, help='The algorithm to use. Choose from "ppo" or "a2c".')
    args = parser.parse_args()

    if args.env == 'cartpole':
        env_name = 'CartPole-v1' 
        env = CartPoleEnv(env_name)
    elif args.env == 'pong':
        env_name = 'PongNoFrameskip-v4'
        env = PongEnv(env_name)
    else:
        raise ValueError("Environment name incorrect or found")
    
    if args.algo == 'ppo':
        runner = PPOrunner(env)
    elif args.algo == 'a2c':
        runner = A2Crunner(env)
    else:
        raise ValueError("Algorithm name incorrect or not found")
    
    runner.run_experiment()

if __name__ == "__main__":
    main()

