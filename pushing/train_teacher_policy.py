import os
import copy
from argparse import ArgumentParser

import gym
import torch
import numpy as np
from tensorboardX import SummaryWriter

from pushing.rl import (
    ParameterNoise,
    Actor,
    Critic
)
from stitching.rl import (
    ReplayBuffer,
    PopTart,
    soft_update,
)

def main():
    parser = ArgumentParser()
    parser.add_argument('store_dir', type=str)
    parser.add_argument('id', type=str)
    args = parser.parse_args()

    buffer = ReplayBuffer(int(100000))
    dataloader = torch.utils.data.DataLoader(buffer, batch_size=128, shuffle=True)
    env = gym.make('SimplePusher-v0')
    gamma = 0.99

    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    critic_target = copy.deepcopy(critic)
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    actor_target = copy.deepcopy(actor)

    norm_q = PopTart()
    parameter_noise = ParameterNoise(actor, 0.2)

    def parameters():
        for model in [critic, norm_q]:
            for parameter in model.parameters():
                yield parameter

    critic_opt = torch.optim.Adam(parameters())
    actor_opt = torch.optim.Adam(actor.parameters(), weight_decay=1e-2)

    logger_name = f'pushing/runs/rot-x{env.env.rotation_x:.3f}-{args.id}'

    with open(os.path.join(args.store_dir, 'info.txt'), 'w') as f:
        f.write(f'rotation_x: {env.env.rotation_x}\n')
    logger = SummaryWriter(logger_name)

    if torch.cuda.is_available:
        actor.cuda()
        actor_target.cuda()
        critic.cuda()
        critic_target.cuda()
        norm_q.cuda()
        parameter_noise.cuda()

    def action_transform(u):
        """
        Transforms from [-1, 1] to [actionspace.low, actionspace.high]
        """
        u_01 = (u + 1) / 2
        action_range = (env.action_space.high - env.action_space.low)
        return u_01 * action_range + env.action_space.low

    def nn_policy(obs):
        x = torch.cuda.FloatTensor(obs.reshape(1, -1))
        u = actor(x).cpu().detach().numpy().flatten()
        return u

    nb_rollout_steps = 100
    nb_epoch_cycles = 20
    nb_epochs = int(2000000 / (nb_rollout_steps * nb_epoch_cycles))
    nb_train_steps = 50


    def rollout(policy, epsilon=0.0, render=False, fill_buffer=True, nb_rollout_steps=nb_rollout_steps):
        obs = env.reset()
        R = 0.0
        for t in range(nb_rollout_steps):
            if np.random.rand() > epsilon:
                u = policy(obs)
            else:
                u = 2 * np.random.rand(2) - 1
            obs_, r, done, info = env.step(action_transform(u))
            if render:
                env.render()
            R += gamma ** t * r
            if fill_buffer:
                buffer.append((obs.astype(np.float32),
                               u.astype(np.float32),
                               np.array([r]).astype(np.float32),
                               obs_.astype(np.float32)))
            obs = obs_
        return R

    while len(buffer) < 50000:
        rollout(nn_policy, epsilon=1.0)

    def noise_estimate(x):
        u_no_noise = actor(x)
        parameter_noise.apply()
        u_noise = actor(x)
        parameter_noise.reset()
        d = ((u_noise - u_no_noise) ** 2).mean().sqrt()
        return d

    actor.train()
    critic.train()

    rolling_return = None
    best_rolling_return = -np.inf

    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):

            # Rollouts
            parameter_noise.apply()
            rollout(nn_policy, epsilon=0.01, nb_rollout_steps=nb_rollout_steps)
            parameter_noise.reset()

            # Train
            for batch_ind, batch in enumerate(dataloader):
                if batch_ind >= nb_train_steps:
                    break

                x, u, r, x_ = map(lambda x: x.cuda(), batch)

                # Update parameter noise
                d = noise_estimate(x)
                parameter_noise.feedback(d)

                # update critic
                q_target = r + gamma * norm_q(critic_target(x_, actor_target(x_)))
                q_loss = norm_q.mse_loss(critic(x, u), q_target.detach())
                critic_opt.zero_grad()
                q_loss.backward(retain_graph=True)
                critic_opt.step()

                # update actor
                actor_u = actor(x)
                actor_loss = -critic(x, actor_u).mean() + 0.1 * (actor_u ** 2).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(critic_target, critic, 0.001)
                soft_update(actor_target, actor, 0.001)

            step = (epoch * nb_epoch_cycles + cycle) * nb_rollout_steps

            logger.add_scalar('q_loss', q_loss, step)
            logger.add_scalar('actor_loss', actor_loss, step)

            # Evaluate
            R_eval = rollout(nn_policy,
                             epsilon=0.0,
                             nb_rollout_steps=nb_rollout_steps,
                             render=False,
                             fill_buffer=False)
            if rolling_return is None:
                rolling_return = R_eval
            else:
                rolling_return = 0.8 * rolling_return + 0.2 * R_eval
            if rolling_return > best_rolling_return:
                best_rolling_return = rolling_return
                torch.save(actor.state_dict(), os.path.join(args.store_dir, 'actor.pt'))
                torch.save(critic.state_dict(), os.path.join(args.store_dir, 'critic.pt'))
                torch.save(norm_q.state_dict(), os.path.join(args.store_dir, 'norm_q.pt'))
            logger.add_scalar('rolling_return', rolling_return, step)
            logger.add_scalar('R_eval', R_eval, step)


if __name__ == '__main__':
    main()
