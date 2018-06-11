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
    resolutions = [79, 93, 101]
    q_resolution = 33
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
    agent_path = os.path.join(agent_dir, 'agent.pkl')

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
            V[state] += 0.1 * (q_max - V[state])
            state_inv = [-state[0], -state[1]]
            V[state_inv] = V[state]
        state = env.reset()
        for _ in range(256):
            a_max, q_max = Q.max_action(state)
            V[state] += 0.1 * (q_max - V[state])
            state_inv = [-state[0], -state[1]]
            V[state_inv] = V[state]
            state, _, _, _ = env.step(a_max)
        if i % 32 == 0:
            print('Time remaining:', deadline - datetime.datetime.now())
            #state = env.reset()
            #plt.figure(figsize=(4, 4))
            #plt.imshow(V._vs[-1]._v, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
            #plt.ylabel('$\\dot{\\theta}$')
            #plt.xlabel('$\\theta$')
            #plt.colorbar()
            #plt.savefig(os.path.join(agent_dir, 'value_function.pdf'))
            #plt.close()
            #actions = np.zeros((resolutions[-1], resolutions[-1]))
            #for col, th in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], resolutions[-1])):
            #    for row, th_dot in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], resolutions[-1])):
            #        a_max, q_max = Q.max_action([th, th_dot])
            #        actions[row, col] = a_max[0]
            #plt.imshow(actions, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
            #plt.ylabel('$\\dot{\\theta}$')
            #plt.xlabel('$\\theta$')
            #plt.colorbar()
            #plt.savefig(os.path.join(agent_dir, 'policy.pdf'))
            #plt.close()
            with open(agent_path, 'wb') as f:
                pickle.dump((V, params), f)
