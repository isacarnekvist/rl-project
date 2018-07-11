import os
import pickle
import datetime
from argparse import ArgumentParser

import gym
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from rl import TiledValueFunction, QFunction

if __name__ == '__main__':
    parser = ArgumentParser(description='Train an optimal pendulum swing-up agent')
    parser.add_argument('store_dir', type=str, help='directory to store agent and plots in')
    args = parser.parse_args()

    env = gym.make('Pendulum-v0')
    params = {
        'mass_coeff': np.random.rand(),
        'action_coeff': np.random.rand(),
    }
    #params = {
    #    'mass_coeff': 1.0,
    #    'action_coeff': 1.0,
    #}
    env.env.init_params(**params)
    resolutions = [71, 93]
    q_resolution = 101
    V = TiledValueFunction(env.observation_space.shape[0],
                      env.observation_space.low,
                      env.observation_space.high,
                      resolutions)
    Q = QFunction(V,
                  env.env.forward,
                  env.env.reward,
                  env.action_space.low,
                  env.action_space.high,
                  resolution=q_resolution)

    agent_id = 'pendulum_{:02}'.format(len(os.listdir(args.store_dir)))
    agent_dir = os.path.join(args.store_dir, agent_id)
    if not os.path.exists(agent_dir):
        os.mkdir(agent_dir)
    agent_path = os.path.join(agent_dir, 'policy.pkl')

    deadline = datetime.datetime.now() + datetime.timedelta(hours=4)
    with open(os.path.join(agent_dir, 'info.txt'), 'w') as f:
        f.write('  mass coefficient: {mass_coeff:.2f}\n'.format(**params) +
                'action coefficient: {action_coeff:.2f}\n'.format(**params))

    i = -1
    while datetime.datetime.now() < deadline:
        i += 1
        V[0, 0] = 0
        for j in range(256):
            state = V.sample_state()
            a_max, q_max = Q.max_action(state)
            V[state] += 0.5 * (q_max - V[state])
            state_inv = [-state[0], -state[1]]
            V[state_inv] = V[state]
        state = env.reset()
        for _ in range(256):
            a_max, q_max = Q.max_action(state)
            V[state] += 0.5 * (q_max - V[state])
            state_inv = [-state[0], -state[1]]
            V[state_inv] = V[state]
            state, _, _, _ = env.step(a_max)
        if i % 15 == 0:
            print('Time remaining:', deadline - datetime.datetime.now())

            # SAVE
            output = {
                'params': params,
                'value_functions': []
            }
            for v in V._vs:
                output['value_functions'].append({
                    'resolution': v.resolution,
                    'lower_limits': v.lower_limits,
                    'upper_limits': v.upper_limits,
                    'values': v._v
                })
            with open(agent_path, 'wb') as f:
                pickle.dump(output, f)
