import os
import yaml
from argparse import ArgumentParser

import torch

from stitching.rl import Policy, LatentMapping
from gym.envs.mujoco.simple_pusher import SimplePusherEnv


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('env_dir', type=str)
    parser.add_argument('--n_sim_steps', type=int, default=1)
    args = parser.parse_args()

    params_path = os.path.join(args.env_dir, 'info.txt')
    with open(params_path) as f:
        params = yaml.load(f)
    env = SimplePusherEnv(**params, n_sim_steps=args.n_sim_steps)

    env_id = int(args.env_dir[-3:])

    latent_size = 8
    latent_mapping = LatentMapping(latent_size, 32)
    latent_mapping.load_state_dict(torch.load('stitching/models/pushing/latent_mapping.pt'))

    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], latent_size)
    policy.load_state_dict(torch.load('stitching/models/pushing/policy.pt'))
    policy.eval()

    z = latent_mapping.mean[env_id].reshape(1, -1)

    while True:
        obs = env.reset()
        done = False
        while not done:
            x = torch.FloatTensor(obs).reshape(1, -1)
            x[0, [2, 3]] = 0.0
            u = policy(x, z).detach().numpy().flatten()
            obs, reward, done, info = env.step(u * env.action_space.high)
            env.render()
