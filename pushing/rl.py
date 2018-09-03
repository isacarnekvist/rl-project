import os
import sys
import yaml

import torch
import torch.nn.functional as F
from gym.envs.mujoco.simple_pusher import SimplePusherEnv

from stitching.policy import PolicyABC


class ParameterNoise:
    """
    Applies gaussian noise to the parameters of the model.
    
    Usage:
    noise = ParameterNoise(model).cuda()  # or without cuda()
    noise.apply(0.1)  # Standard deviation of noise passed as argument
    ...
    noise.reset()
    """
    def __init__(self, model, sigma):
        self.is_cuda = False
        self.sigma = sigma
        self._delta = 0.1
        self.noise = dict()
        self.model = model
        self.noise_applied = False
    
    def apply(self):
        if self.noise_applied:
            self.reset()
        self.noise_applied = True
        for i, param in enumerate(self.model.parameters()):
            if self.is_cuda:
                self.noise[i] = (torch.randn(param.shape) * self._delta).cuda()
            else:
                self.noise[i] = torch.randn(param.shape) * self._delta
            param.data.add_(self.noise[i])
    
    def reset(self):
        if self.noise_applied:
            for i, param in enumerate(self.model.parameters()):
                param.data.add_(-self.noise[i])
        self.noise_applied = False

    def feedback(self, d):
        alpha = 1.01
        if d < self.sigma:
            self._delta *= alpha
        else:
            self._delta /= alpha

    def cuda(self):
        self.is_cuda = True
        return self


class WelfordNormalization(torch.nn.Module):

    def __init__(self, num_features):
        super(WelfordNormalization, self).__init__()
        self._count = 0 
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('variance', torch.zeros(num_features))

    def _update(self, x): 
        self._count += 1
        self.mean.data += (x.mean(dim=0) - self.mean) / self._count
        self.variance.data += (((x - self.mean) ** 2).mean(dim=0) - self.variance) / self._count

    def forward(self, x): 
        if self.training:
            self._update(x)
        return (x - self.mean) / (self.variance ** 0.5 + 1e-9)


class Critic(torch.nn.Module):
    
    def __init__(self, x_size, u_size):
        super(Critic, self).__init__()
        hidden_size = 400
        self.input_norm_x = WelfordNormalization(x_size)
        self.input_norm_u = WelfordNormalization(u_size)
        self.fcx = torch.nn.Linear(x_size, hidden_size)
        self.lnx = torch.nn.LayerNorm(hidden_size)

        self.fc1 = torch.nn.Linear(hidden_size + u_size, hidden_size)
        self.ln1 = torch.nn.LayerNorm(hidden_size)

        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

        self.fcy = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x, u):
        x_normed = self.input_norm_x(x)
        y0 = F.relu(self.lnx(self.fcx(x_normed)))
        
        x1 = torch.cat([y0, self.input_norm_u(u)], dim=1)
        y1 = F.relu(self.ln1(self.fc1(x1)))
        
        y2 = F.relu(self.ln2(self.fc2(y1)))
        
        return self.fcy(y2)


class Actor(torch.nn.Module):
    
    def __init__(self, x_size, u_size):
        super(Actor, self).__init__()
        hidden_size = 400
        self.input_norm_x = WelfordNormalization(x_size)
        self.fcx = torch.nn.Linear(x_size, hidden_size)
        self.lnx = torch.nn.LayerNorm(hidden_size)

        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ln1 = torch.nn.LayerNorm(hidden_size)

        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

        self.fcy = torch.nn.Linear(hidden_size, u_size)
        
    def forward(self, x):
        x_normed = self.input_norm_x(x)
        y0 = F.relu(self.lnx(self.fcx(x_normed)))
        y1 = F.relu(self.ln1(self.fc1(y0)))
        y2 = F.relu(self.ln2(self.fc2(y1)))
        return F.tanh(self.fcy(y2))


####################################
# Stitching subclasses and helpers #
####################################


class PushingPolicy(PolicyABC):

    def __init__(self, policy_path, env):
        self._high = env.action_space.high
        self._actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).cuda()
        self._actor.load_state_dict(torch.load(policy_path))
        self._actor.eval()

    def act(self, state):
        state_ = torch.cuda.FloatTensor(state).reshape(1, -1)
        action = self._actor(state_)
        return action.cpu().detach().numpy()[0] * self._high


def load_from_path(policy_dir):
    """Returns a parameterized environment"""
    params_path = os.path.join(policy_dir, 'info.txt')
    with open(params_path) as f:
        params = yaml.load(f)
    env = SimplePusherEnv(**params)
    policy_path = os.path.join(policy_dir, 'actor.pt')
    policy = PushingPolicy(policy_path, env)
    return policy, env


if __name__ == '__main__':
    policy, env = load_from_path('trained_agents/pusher_000/')
    obs = env.reset()
    print(policy.act(obs))
