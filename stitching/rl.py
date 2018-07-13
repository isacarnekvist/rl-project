import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class WelfordNormalization(torch.nn.Module):

    def __init__(self, num_features):
        super(WelfordNormalization, self).__init__()
        self.count = 0
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('M2', torch.zeros(num_features))

    def update_single(self, x):
        self.count += 1
        delta1 = x - self.mean
        self.mean += delta1 / self.count
        delta2 = x - self.mean
        self.M2 += delta1 * delta2

    def update_batch(self, x):
        if self.count > 1e6:
            return
        for n in range(x.size(0)):
            self.update_single(x[n, :])


    @property
    def variance(self):
        return self.M2 / (self.count - 1)

    def forward(self, x):
        return (x - self.mean) / (self.variance ** 0.5 + 1e-9)


class PolicyCollection(object):
    """
    TODO: Parallelize? Use baselines' environment array?

    Usage:
    >>> collection = PolicyCollection()
    >>> # [collection.append(policy, env) for ...]
    >>> # collection.reset()
    >>> # i, x, u, r, x_, u_ = collection.step()
    """

    def __init__(self):
        self.envs = []
        self.policies = []
        self.observations = None

    def append(self, policy, env):
        self.envs.append(env)
        self.policies.append(policy)

    def reset(self):
        self.observations = [env.reset() for env in self.envs]

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    def step(self, epsilon=0.5):
        """
        Returns : (i, x, u, r, x_, u_)
            x  : initial state
            u  : epsilon greedy policy in initial state
            r  : reward
            x_ : successor state
            u_ : policy action in successor state
        """
        if self.observations is None:
            raise ValueError('Call reset() before step()')
        if len(self.observations) != len(self.envs):
            raise ValueError('Number of last observation does not match number of envs')
        for i, (obs, policy, env) in enumerate(zip(self.observations,
                                                   self.policies,
                                                   self.envs)):
            random_action = env.action_space.sample()
            optimal_action = np.array(policy.act(obs), dtype=np.float32)

            # Epsilon greedy
            if np.random.random() < epsilon:
                action = random_action
            else:
                action = optimal_action

            obs_, r, done, _ = env.step(action)
            optimal_action_ = np.array(policy.act(obs_), dtype=np.float32)
            if done:
                self.observations[i] = self.envs[i].reset()
            else:
                self.observations[i] = obs_
            yield np.array([i]), obs, action, optimal_action, np.array([r]), obs_, optimal_action_

    def __len__(self):
        return len(self.envs)


class StateActionFunction(torch.nn.Module):

    def __init__(self, x_size, u_size, z_size):
        super(StateActionFunction, self).__init__()
        self.fc1 = torch.nn.Linear(x_size + z_size + u_size, 400)
        self.n_layers = 10
        for layer_id in range(2, self.n_layers):
            self.__setattr__(f'fc{layer_id}', torch.nn.Linear(400, 400))
        self.fcy = torch.nn.Linear(400, 1)

    def forward(self, x, u, z):
        xuz = torch.cat([x, u, z], dim=1)
        y = F.relu(self.fc1(xuz))
        for layer_id in range(2, self.n_layers):
            fc = self.__getattr__(f'fc{layer_id}')
            y = F.relu(fc(y)) + y
        return self.fcy(y)


class ValueFunction(torch.nn.Module):

    def __init__(self, x_size, z_size):
        super(ValueFunction, self).__init__()
        self.fc1 = torch.nn.Linear(x_size + z_size, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)

    def forward(self, x, z):
        xz = torch.cat([x, z], dim=1)
        a1 = F.relu(self.fc1(xz))
        a2 = F.relu(self.fc2(a1))
        y = self.fc3(a2)
        return y


class Policy(torch.nn.Module):

    def __init__(self, x_size, u_size, z_size, u_max=2.0):
        super(Policy, self).__init__()
        self.u_max = u_max
        self.fc1 = torch.nn.Linear(x_size + z_size, 400)
        self.n_layers = 10
        for layer_id in range(2, self.n_layers):
            self.__setattr__(f'fc{layer_id}', torch.nn.Linear(400, 400))
        self.fcy = torch.nn.Linear(400, u_size)

    def forward(self, x, z):
        xz = torch.cat([x, z], dim=1)
        y = F.relu(self.fc1(xz))
        for layer_id in range(2, self.n_layers):
            fc = self.__getattr__(f'fc{layer_id}')
            y = F.relu(fc(y)) + y
        y = self.u_max * F.tanh(self.fcy(y))
        return y


class LatentMapping(torch.nn.Module):
    """
    Arguments:
        - z_size : int
            size of latent space
        - n_latent : int
            number of rows in latent table
    """
    def __init__(self, z_size, n_latent):
        super(LatentMapping, self).__init__()
        self.z_size = z_size
        self.mean = torch.nn.Parameter(torch.randn(n_latent, self.z_size))
        self.std_logits = torch.nn.Parameter(0.01 * torch.randn(1, self.z_size))

    def forward(self, i, deterministic=False):
        """
        Input:
            i : n x 1 tensor of integers representing the index of the environment
        Returns: (z, kl)
            z  : n x 1 tensor of z-samples
            kl : n x 1 tensor of kl divergences
        """
        µ = self.mean[i.squeeze(), :]
        σ = torch.exp(self.std_logits).repeat(i.size(0), 1)

        # Sample
        eps = torch.randn(i.size(0), self.z_size)
        if deterministic:
            eps = eps * 0
        if i.is_cuda:
            eps = eps.cuda()
        z = µ + eps * σ

        # KL-divergence
        kl = (1 / 2 * (σ ** 2 + µ ** 2 - torch.log(σ ** 2) - 1)).sum(dim=1, keepdim=True)

        return z, kl


class LatentWrapped(torch.nn.Module):
    """
    Example usage:
    >>> n_latent = 10
    >>> x_size, u_size, z_size = 2, 3, 4
    >>> q = QFunction(x_size, u_size, z_size)
    >>> w = LatentWrapped(q, n_latent, z_size)
    """
    def __init__(self, q, latent_mapping):
        super(LatentWrapped, self).__init__()
        self.latent_mapping = latent_mapping
        self.q = q

    def forward(self, i, x, u, deterministic=False):
        """
        Input:
            i : 2-d tensor of integers representing the index of the environment
            x : 2-d tensor of environment observations
            u : 2-d tensor of actions
        Returns: (q, kl)
            q  : n x 1 tensor of q estimates
            kl : n x 1 tensor of kl divergences
        """
        z, kl = self.latent_mapping(i, deterministic=deterministic)

        return self.q(x, u, z), kl


class ReplayBuffer(Dataset):
    """
    Usage:
    >>> buffer = ReplayBuffer(2)  # Creates a replay buffer with capacity 2
    >>> buffer.extend([1, 2])
    >>> buffer[0]
    1
    >>> len(buffer)
    2
    >>> buffer.append(3)
    >>> len(buffer)
    2
    """
    def __init__(self, capacity):
        self._contents = dict()
        self._len = 0
        self._capacity = capacity
        if capacity < 1:
            raise ValueError('capacity have to be > 0')
        if type(capacity) is not int:
            raise ValueError('')

    def append(self, x):
        if self._len == self._capacity:
            i = self._drop_random()
        else:
            i = self._len
            self._len += 1
        self._contents[i] = x

    def extend(self, iterable):
        for x in iterable:
            self.append(x)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        return self._contents[i]

    def _drop_random(self):
        i = np.random.randint(0, len(self))
        del self._contents[i]
        return i


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class PopTart(torch.nn.Module):
    """
    Special case for one-dimensional regression
    """

    def __init__(self, alpha=0.999):
        super(PopTart, self).__init__()
        self.count = 0
        self.α = alpha
        self.β = alpha
        self.register_buffer('m1', torch.zeros(1))
        self.register_buffer('m2', torch.ones(1))
        self.w = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.zeros(1))

    @property
    def µ(self):
        return self.m1

    @property
    def σ(self):
        return torch.abs(self.m2 - self.m1 ** 2) ** 0.5 + 1e-6

    def _update(self, targets):
        n = targets.size(0)
        a = self.α ** n
        b = self.β ** n
        self.m1 = a * self.m1 + (1 - a) * targets.mean(dim=0)
        self.m2 = b * self.m2 + (1 - b) * (targets ** 2).mean(dim=0)

    def mse_loss(self, predictions, targets):
        if self.training:
            self.w.data = self.σ * self.w
            self.b.data = self.σ * self.b + self.µ

            self._update(targets)

            self.w.data = self.w / self.σ
            self.b.data = (self.b - self.µ) / self.σ

        targets_normed = (targets - self.µ) / self.σ
        predictions_normed = self.w * predictions + self.b
        return F.mse_loss(predictions_normed, targets_normed)

    def forward(self, predictions):
        """
        Returns the un-normalized (true) prediction values
        """
        return self.σ * (self.w * predictions + self.b) + self.µ


if __name__ == '__main__':
    poptart = PopTart(alpha=0.99)
    for _ in range(1000):
        x = torch.randn(10, 1) * 5 + 10
        y = x
        loss = poptart.mse_loss(x, y)
        print(poptart.m1)
