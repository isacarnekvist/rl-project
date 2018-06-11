import numpy as np


class ValueFunction(object):

    def __init__(self, state_dims, lower_limits, upper_limits, resolution, value_init=0.0):
        self.resolution = resolution
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self._v = np.ones((resolution, ) * state_dims) * value_init

    def _item_round(self, item):
        item = np.clip(item, self.lower_limits, self.upper_limits)
        unrounded = (item - self.lower_limits) / (self.upper_limits - self.lower_limits) * (self.resolution - 1)
        rounded = np.round(unrounded, 0)
        return rounded.astype(np.int)

    def __getitem__(self, item):
        indices = self._item_round(item)
        return self._v[indices[0], indices[1]]  # Don't know how to do this general yet

    def __setitem__(self, item, v):
        indices = self._item_round(item)
        self._v[indices[0], indices[1]] = v  # Don't know how to do this general yet

    def iter_states(self):
        # Don't know how to do this general yet
        for x in np.linspace(self.lower_limits[0], self.upper_limits[0], self.resolution):
            for y in np.linspace(self.lower_limits[1], self.upper_limits[1], self.resolution):
                yield [x, y]

    def sample_state(self):
        return np.random.uniform(self.lower_limits, self.upper_limits)


class TiledValueFunction(object):

    def __init__(self, state_dims, lower_limits, upper_limits, resolutions, value_init=0.0):
        self._vs = []
        self.n = len(resolutions)
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        for resolution in resolutions:
            self._vs.append(ValueFunction(state_dims, lower_limits, upper_limits, resolution))

    def __getitem__(self, item):
        result = 0.0
        for v in self._vs:
            result += v[item] / self.n
        return result

    def __setitem__(self, item, x):
        for v in self._vs:
            v[item] = x

    def sample_state(self):
        return np.random.uniform(self.lower_limits, self.upper_limits)


class QFunction(object):

    def __init__(self, value_function, forward, reward, lower_limits, upper_limits, resolution=33, gamma=0.99):
        self.gamma = gamma
        self.resolution = resolution
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self._v = value_function
        self._r = reward
        self._f = forward

    def get(self, state, action):
        reward = self._r(state, action)
        state_ = self._f(state, action)
        return reward + self.gamma * self._v[state_]

    def iter_actions(self):
        # for now assume 1-d action space
        for a in np.linspace(self.lower_limits[0], self.upper_limits[0], self.resolution):
            yield [a]

    def max_action(self, state):
        a_max = None
        q_max = -np.inf
        for action in self.iter_actions():
            q = self.get(state, action)
            if q > q_max:
                q_max = q
                a_max = action
        return a_max, q_max
