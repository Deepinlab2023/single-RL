import torch as th
from helpers.a2c_helper import to_tensor

def a2c_test(env, actor, num_test_episodes, t_max, action_offset):
    test_rewards = []

    for episode in range(num_test_episodes):
        state = env.reset()
        prev_state = None
        total_reward = 0
        done = False
        t = 0

        while not done and (t < t_max):
            with th.no_grad():
                if 'PongNoFrameskip-v4' in env.env_name:
                    state_tensor = env.pre_process_no_prev(state)
                else:
                    state_tensor = to_tensor(state)

                logits = actor(state_tensor)
                action, _ = actor.action_sampler(logits)
                converted_action = action.item() + action_offset

            next_state, reward, done, _ = env.step(converted_action)
            prev_state = state
            state = next_state
            total_reward += reward
            t += 1

        test_rewards.append(total_reward)
    print("Test Complete")
    return sum(test_rewards) / num_test_episodes