import os
from argparse import ArgumentParser

import torch

from pushing.rl import Actor
from gym.envs.mujoco import SimplePusherEnv


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('actor_dir', type=str)
    parser.add_argument('--n_sim_steps', type=int, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.actor_dir, 'info.txt'), 'r') as f:
        rotation_x = float(f.read().split()[1])

    n_sim_steps = 1
    env = SimplePusherEnv(rotation_x=rotation_x, n_sim_steps=args.n_sim_steps)
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    actor.eval()

    if torch.cuda.is_available:
        actor.cuda()

    actor.load_state_dict(torch.load(os.path.join(args.actor_dir, 'actor.pt')))

    while True:
        obs = env.reset()
        done = False
        while not done:
            x = torch.cuda.FloatTensor(obs.reshape(1, -1))
            u = actor(x).detach().cpu().numpy()[0]
            obs, r, done, _ = env.step(0.1 * u)
            env.render()
