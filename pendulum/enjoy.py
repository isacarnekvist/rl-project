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
    parser.add_argument('--plot_value', action='store_true', help='Plot value function')
    parser.add_argument('--plot_policy', action='store_true', help='Plot policy')
    parser.add_argument('--no_render', action='store_true', help='Skip rendering')
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon greedy value')
    args = parser.parse_args()
    agent_dir = os.path.join(args.agent_dir, 'policy.pkl')

    # LOAD
    policy, env = load_from_path(agent_dir)
    print(f'mass: {env.env.m:.2f} action cost: {env.env.action_cost:.5f}')

    # PLAY
    while True:
        state = env.reset()
        ths = []
        thdots = []
        for _ in range(300):
            if np.random.random() < args.epsilon:
                action = env.action_space.sample()
            else:
                action = policy.act(state)
            state, _, done, _ = env.step(action)
            ths.append(state[0])
            thdots.append(state[1])
            if not (args.plot_value or args.plot_policy or args.no_render):
                env.render()
            if done:
                break
        env.close()
        break

    if args.plot_value:
        V = policy._Q._v
        plt.figure(figsize=(18, 6))
        v_img = np.tile(V._vs[-1]._v.T, [1, 3])
        plt.imshow(v_img, extent=[3 * V.lower_limits[0], 3 * V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'value_function.pdf'))
        plt.close()

    if args.plot_policy:
        V = policy._Q._v
        res = 65
        actions = np.zeros((res, res))
        for col, th in enumerate(np.linspace(V.lower_limits[0], V.upper_limits[0], res)):
            if col > res / 2:
                break
            for row, th_dot in enumerate(np.linspace(V.lower_limits[1], V.upper_limits[1], res)):
                a_max = policy.act([th, th_dot])
                actions[row, col] = a_max[0]
                actions[res - row - 1, res - col - 1] = -a_max[0]
        plt.figure(figsize=(18, 6))
        policy_img = np.tile(actions, [1, 3])
        plt.imshow(policy_img, extent=[3 * V.lower_limits[0], 3 * V.upper_limits[0], V.lower_limits[1], V.upper_limits[1]], aspect='auto', origin='lower')
        ths = np.array(ths)
        thdots = np.array(thdots)
        plt.plot(ths, thdots, 'k.', markersize=10.0)
        plt.plot(ths + 2 * np.pi, thdots, 'k.', markersize=10.0)
        plt.plot(ths - 2 * np.pi, thdots, 'k.', markersize=10.0)
        plt.plot(ths, thdots, 'w.', markersize=6.0)
        plt.plot(ths + 2 * np.pi, thdots, 'w.', markersize=6.0)
        plt.plot(ths - 2 * np.pi, thdots, 'w.', markersize=6.0)
        plt.ylabel('$\\dot{\\theta}$')
        plt.xlabel('$\\theta$')
        plt.colorbar()
        plt.savefig(os.path.join(args.agent_dir, 'policy.pdf'))
        plt.close()
