import gym
env = gym.make('SpaceInvaders-v0',render_mode='human')
env.reset()
for t in range(1000):
    env.render()
    action = env.action_space.sample()
    o,r,d,_,i = env.step(action)
    if d:
        env.reset()