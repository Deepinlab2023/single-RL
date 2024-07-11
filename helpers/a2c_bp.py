import torch as th

class BatchTraining:
    def __init__(self):
        pass

    def collate_batch(self, buffer, rtrns):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_rtrns = []

        # Extract data from buffer
        for (data, rtrn) in zip(buffer, rtrns):
            state, action, reward, next_state = data
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state)
            batch_rtrns.append(rtrn)

        batch_states = th.stack(batch_states)
        batch_actions = th.stack(batch_actions)
        batch_rewards = th.tensor(batch_rewards, dtype=th.float32)
        batch_next_states = th.stack(batch_next_states)
        batch_rtrns = th.tensor(batch_rtrns, dtype=th.float32)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_rtrns