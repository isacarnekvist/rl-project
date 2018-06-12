import os
import pickle
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl import load_from_path


if __name__ == '__main__':
    parser = ArgumentParser(description='Demo a pendulum swing-up agent')
    parser.add_argument('agent_dir', type=str, help='directory to load agent from')
    parser.add_argument('--plot', action='store_true', help='Plot value function and policy')
    args = parser.parse_args()
    agent_dir = os.path.join(args.agent_dir, 'policy.pkl')

    # LOAD
    policy, env = load_from_path(agent_dir)

    # PLAY
    while True:
        state = env.reset()
        n_steps_finished = 0
        ths = []
        thdots = []
        for _ in range(300):
            a_max = policy.act(state)
            state = env.step(a_max)[0]
            ths.append(state[0])
            thdots.append(state[1])
            if np.linalg.norm(state) < 0.1:
                n_steps_finished += 1
                if n_steps_finished > 32:
                    break
            if not args.plot:
                env.render()
        env.close()
        break

    if args.plot:
        V = policy._Q._v
        plt.figure(figsize=(6, 6))
        plt.imshow(V._vs[-1]._v, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'value_function.pdf'))
        plt.show()

        res = 31
        actions = np.zeros((res, res))
        for col, th in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], res)):
            if col > res / 2:
                break
            for row, th_dot in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], res)):
                a_max = policy.act([th, th_dot])
                actions[row, col] = a_max[0]
                actions[res - row - 1, res - col - 1] = -a_max[0]
        plt.imshow(actions, extent=[V.lower_limits[0], V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        plt.plot(ths, thdots, 'r.')
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'policy.pdf'))
        plt.show()
