import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


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


def normalize_action(env, action):
    action_range = env.action_space.high - env.action_space.low
    return 2 * (action - env.action_space.low) / action_range - 1


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
        All actions return are normalized to range [-1, 1]

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
            random_action = env.action_space.sample().astype(np.float32)
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
            yield (
                np.array([i]),
                obs.astype(np.float32),
                normalize_action(env, action),
                normalize_action(env, optimal_action),
                np.array([r]),
                obs_,
                normalize_action(env, optimal_action_)
            )

    def __len__(self):
        return len(self.envs)


class StateActionFunction(torch.nn.Module):

    def __init__(self, x_size, u_size, z_size, n_layers=10, use_layer_norm=False):
        super(StateActionFunction, self).__init__()
        self.input_norm_x = WelfordNormalization(x_size)
        self.input_norm_u = WelfordNormalization(u_size)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm_1 = torch.nn.LayerNorm(400)
        self.fc1 = torch.nn.Linear(x_size + z_size + u_size, 400)
        self.n_layers = n_layers
        for layer_id in range(2, self.n_layers):
            if self.use_layer_norm:
                self.__setattr__(f'layer_norm{layer_id}', torch.nn.LayerNorm(400))
            self.__setattr__(f'fc{layer_id}', torch.nn.Linear(400, 400))
        self.fcy = torch.nn.Linear(400, 1)

    def forward(self, x, u, z=None):
        x = self.input_norm_x(x)
        u = self.input_norm_u(u)
        if z is None:
            xuz = torch.cat([x, u], dim=1)
        else:
            xuz = torch.cat([x, u, z], dim=1)
        y = F.relu(self.fc1(xuz))
        for layer_id in range(2, self.n_layers):
            fc = self.__getattr__(f'fc{layer_id}')
            if self.use_layer_norm:
                layer_norm = self.__getattr__(f'layer_norm{layer_id}')
                y = F.relu(layer_norm(fc(y))) + y
            else:
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

    def __init__(self, x_size, u_size, z_size, n_layers=10, use_layer_norm=False):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(x_size + z_size, 400)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm_1 = torch.nn.LayerNorm(400)
        self.n_layers = n_layers
        self.input_norm_x = WelfordNormalization(x_size)
        for layer_id in range(2, self.n_layers):
            if self.use_layer_norm:
                self.__setattr__(f'layer_norm{layer_id}', torch.nn.LayerNorm(400))
            self.__setattr__(f'fc{layer_id}', torch.nn.Linear(400, 400))
        self.fcy = torch.nn.Linear(400, u_size)
        w_init = (torch.rand(self.fcy.weight.size()) * 6 - 3) * 1e-3
        self.fcy.weight.data.copy_(w_init)
        self.fcy.bias.data *= 0

    def forward(self, x, z=None):
        x = self.input_norm_x(x)
        if z is None:
            xz = x
        else:
            xz = torch.cat([x, z], dim=1)
        if self.use_layer_norm:
            y = F.relu(self.layer_norm_1(self.fc1(xz)))
        else:
            y = F.relu(self.fc1(xz))
        for layer_id in range(2, self.n_layers):
            fc = self.__getattr__(f'fc{layer_id}')
            if self.use_layer_norm:
                layer_norm = self.__getattr__(f'layer_norm{layer_id}')
                y = F.relu(layer_norm(fc(y))) + y
            else:
                y = F.relu(fc(y)) + y

        return F.tanh(self.fcy(y))


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
    policy = Policy(4, 3, 2, use_layer_norm=True)
    u = policy(torch.randn(10, 4), torch.randn(10, 2))
    print(u)
