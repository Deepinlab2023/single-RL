import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

class Actor(nn.Module):
    def __init__(self, layers1_num, layers2_num, out_num):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(layers1_num, layers2_num), nn.ReLU(),
            nn.Linear(layers2_num, out_num)
        )
        
    def forward(self, d_obs, deterministic=False):
        logits = self.layers(d_obs)
        if deterministic:
            action = int(torch.argmax(logits[0]).detach().cpu().numpy())
            action_prob = 1.0
        else:
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().cpu().numpy()[0])
            action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob

    def convert_action(self, action, env_name):
        if env_name == 'Pong-v0':
            return action + 2
        else:
            return action  # No need to adjust for other environments

    def ppo_loss(self, d_obs, action, action_prob, advantage, eps_clip):
        vs = np.array([[1., 0.], [0., 1.]])  # TODO: Adjust according to your use case
        ts = torch.FloatTensor(vs[action.cpu().numpy()])
        
        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - eps_clip, 1 + eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)
        
        return loss
