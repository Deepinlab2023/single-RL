import torch as th
import torch.nn.functional as F

class A2CActor(th.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
      super(A2CActor, self).__init__()

      # Define Layers
      self.fc1 = th.nn.Linear(state_size, hidden_size)
      self.fc2 = th.nn.Linear(hidden_size, hidden_size)
      self.fc3 = th.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # Build networ that maps state -> logits
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits

    def action_sampler(self, logits):
        softmax_probabilities = F.softmax(logits, dim=-1)
        action_dist = th.distributions.categorical.Categorical(softmax_probabilities)
        action = action_dist.sample()
        return action, action_dist