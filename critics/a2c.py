import torch as th
import torch.nn.functional as F

class A2CCritic(th.nn.Module):
    def __init__(self, state_size, hidden_size, value_size):
        super(A2CCritic, self).__init__()

        # Define Layers
        self.fc1 = th.nn.Linear(state_size, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, hidden_size)
        self.fc3 = th.nn.Linear(hidden_size, value_size)

    def forward(self, state):
        # Build network that maps state -> value
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value