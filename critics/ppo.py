from torch import nn

class Critic(nn.Module):
    def __init__(self, layers1_num, layers2_num):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(layers1_num, layers2_num),
            nn.Tanh(),
            nn.Linear(layers2_num, layers2_num),
            nn.Tanh(),
            nn.Linear(layers2_num, 1)
        )
        
    def forward(self, state):
        state_value = self.critic(state)
        return state_value

    def critic_loss(self, state_val, discounted_rewards):
        loss = nn.MSELoss(reduction='mean')
        loss = loss(state_val, discounted_rewards)
        return loss