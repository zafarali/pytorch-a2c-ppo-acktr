import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import explorer_a2c, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.explorer_policies import MixPolicy
from a2c_ppo_acktr.explorer_storage import RolloutStorageWithExploration
from a2c_ppo_acktr.explorer_manager import DecayExplorer, GaussianExplorer
from evaluation import evaluate

import os
from mlresearchkit.io import utils as mlio

EXPLORER_LAG = 15

def main():
    args = get_args()
    mlio.create_folder(args.log_dir)
    summary_path = os.path.join(args.log_dir, 'summary.csv')
    mlio.argparse_saver(os.path.join(args.log_dir, 'args.txt'), args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    mlio.touch(summary_path)
    mlio.put(summary_path,
             ('frames,mean_deterministic_reward,std_deterministic_reward,'
              'entropy,exploration_coeff,mean_correction,actor_loss,value_loss'))

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = MixPolicy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    
    if args.algo == 'a2c':
        agent = explorer_a2c.A2C_explorer(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    else:
        raise NotImplementedError('Mix only works for A2C right now.')

    rollouts = RolloutStorageWithExploration(
            args.num_steps, args.num_processes,
            envs.observation_space.shape, envs.action_space,
            actor_critic.recurrent_hidden_state_size)
    exploration_manager = GaussianExplorer()
    #exploration_manager = DecayExplorer()
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    explorer_episode_rewards= deque(maxlen=EXPLORER_LAG)
    explorer_exp_params = deque(maxlen=EXPLORER_LAG)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # Draw initial exploration parameters:
    exp_ps = exploration_manager.draw_exploration_coefficients(args.num_processes)
    exp_ps = torch.from_numpy(exp_ps).to(device)
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        actor_critic.dist.set_exploration_parameters(exp_ps)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i_info, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

                    # Save rewards and exploration parameters to use in updates.
                    explorer_episode_rewards.append(info['episode']['r'])
                    explorer_exp_params.append(exp_ps[i_info].item())

                    # Obtain a new exploration coefficient for this environment.
                    new_exp_ps = exploration_manager.draw_exploration_coefficients(1)[0]
                    exp_ps[i_info].copy_(torch.from_numpy(new_exp_ps).to(device))

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks,
                            bad_masks, actor_critic.dist.exploration_parameters)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, mean_correction = agent.update(rollouts)

        rollouts.after_update()
        if len(explorer_exp_params) >= EXPLORER_LAG:
            exploration_manager.update_exploration_distribution(
                np.array(explorer_exp_params).reshape(-1),
                np.array(explorer_episode_rewards).reshape(-1)
            )
            # Clear for new batch of data (iid?)
            explorer_exp_params.clear()
            explorer_episode_rewards.clear()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                ("Updates {}, num timesteps {}, FPS {} \n Last {}"
                 "training episodes: mean/median reward {:.1f}/{:.1f},"
                 "min/max reward {:.1f}/{:.1f}, entropy:{},  exploration coeff: {}\n")
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy,
                        exploration_manager.mu))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if len(envs.observation_space.shape) == 1:
                ob_rms = utils.get_vec_normalize(envs).ob_rms
            else:
                ob_rms = None
            actor_critic.dist.set_exploration_parameters(torch.zeros(args.num_processes, 1).to(device))
            mean_rew, std_rew = evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
            mlio.stash(summary_path,
                    ('{},{},{},{},{},{},{},{}').format(
                           total_num_steps, mean_rew, std_rew,
                           dist_entropy, exploration_manager.mu,
                           mean_correction, action_loss, value_loss))

if __name__ == "__main__":
    main()
