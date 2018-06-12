import numpy as np


class PolicyCollection(object):

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

    def step(self, epsilon=0.1):
        if self.observations is None:
            raise ValueError('Call reset() before step()')
        if len(self.observations) != len(self.envs):
            raise ValueError('Number of last observation does not match number of envs')
        for i, (obs, policy, env) in enumerate(zip(self.observations,
                                                   self.policies,
                                                   self.envs)):
            # Epsilon greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.array(policy.act(obs), dtype=np.float32)

            obs_, r, done, _ = env.step(policy.act(obs))
            action_ = np.array(policy.act(obs_), dtype=np.float32)
            if done:
                raise NotImplementedError()
            self.observations[i] = obs_
            yield i, obs, action, np.array([r]), obs, action_


if __name__ == '__main__':
    PolicyCollection()
