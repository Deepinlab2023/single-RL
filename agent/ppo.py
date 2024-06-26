import random
import gym
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch import nn
from util.parameters import Parameters

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        params = Parameters()
        self.gamma = params.gamma
        self.eps_clip = params.eps_clip

        self.layers1_num = params.layers1_num
        self.layers2_num = params.layers2_num
        self.out_num = params.out_num
        self.layers = nn.Sequential(
            nn.Linear(self.layers1_num, self.layers2_num), nn.ReLU(),
            nn.Linear(self.layers2_num, self.out_num),
        )

    def convert_action(self, action):
        return action + 2

    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob
        '''
        # policy gradient (REINFORCE)
        logits = self.layers(d_obs)
        loss = F.cross_entropy(logits, action, reduction='none') * advantage
        return loss.mean()
        '''

        # PPO
        vs = np.array([[1., 0.], [0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss
