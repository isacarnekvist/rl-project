import os
import pickle
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl import TiledValueFunction, QFunction


if __name__ == '__main__':
    parser = ArgumentParser(description='Demo a pendulum swing-up agent')
    parser.add_argument('agent_dir', type=str, help='directory to load agent from')
    parser.add_argument('--plot', action='store_true', help='Plot value function and policy')
    args = parser.parse_args()
    agent_dir = os.path.join(args.agent_dir, 'agent.pkl')

    # LOAD
    with open(agent_dir, 'rb') as f:
        V, params = pickle.load(f)
    print(params)
    env = gym.make('Pendulum-v0')
    env.env.init_params(**params)
    resolutions = [v.resolution for v in V._vs]
    #V = TiledValueFunction(env.observation_space.shape[0],
    #                       env.observation_space.low,
    #                       env.observation_space.high,
    #                       resolutions)
    #V._vs = v
    q_resolution = 33
    Q = QFunction(V,
                  env.env.forward,
                  env.env.reward,
                  env.action_space.low,
                  env.action_space.high,
                  resolution=q_resolution)

    # PLAY
    while True:
        state = env.reset()
        n_steps_finished = 0
        ths = []
        thdots = []
        for _ in range(300):
            a_max, q_max = Q.max_action(state)
            state = env.step(a_max)[0]
            ths.append(state[0])
            thdots.append(state[1])
            if np.linalg.norm(state) < 0.1:
                n_steps_finished += 1
                if n_steps_finished > 32:
                    break
            print(state)
            #env.render()
        env.close()
        break

    if args.plot:
        plt.figure(figsize=(4, 4))
        plt.imshow(V._vs[-1]._v, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'value_function.pdf'))
        plt.show()

        res = 64
        actions = np.zeros((res, res))
        for col, th in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], res)):
            for row, th_dot in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], res)):
                a_max, q_max = Q.max_action([th, th_dot])
                actions[row, col] = a_max[0]
        plt.imshow(actions, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        plt.plot(ths, thdots, 'r.')
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'policy.pdf'))
        plt.show()
