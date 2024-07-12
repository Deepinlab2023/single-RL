# single-RL
## agent: 
ppo.py: the code of the framework of PPO

dqn.py: 

a2c.py: The code of the framework of A2C

## env:
pong.py: the environment of Pong, need to be installed manually; how to preprocess the data

PONG MAY NOT WORK FOR PPO

cartpole.py: the environment of CartPole. For A2C, requires Gym version 0.25.2. For PPO, requires Gym environment 0.26.2.

To install gym version:

pip install gym=='0.26.2' for PPO

pip install gym=='0.25.2' for A2C

## doc:
how to install the gym and Atari package as the environment

## util:
logger.py: test the model and draw the figures

train.py: the training process; each algorithm has its own class

## main.py:
Use this command in terminal to run code:

terminal: python main.py --env MY_ENV_HERE --algo MY_ALGO_HERE
