#!/usr/bin/env python
import os
import sys
import logging
import gym
import torch
import time
import datetime
from contextlib import contextmanager
import functools
from operator import itemgetter
import numpy as np
from treeqn.utils.bench.monitor import load_global_results
from treeqn.utils.logger import Logger, HumanOutputFormat
from treeqn.utils.bl_common import explained_variance
from treeqn.utils.seed import set_global_seeds
from treeqn.utils import bench
from treeqn.utils.bl_common.vec_env.subproc_vec_env import SubprocVecEnv
from treeqn.utils.bl_common.atari_wrappers import wrap_deepmind
from treeqn.utils.treeqn_utils import get_timestamped_dir, append_scalar, append_list
from treeqn.models.models import DQNPolicy, TreeQNPolicy
from treeqn.envs.push import Push
import myTorch
from treeqn.envs.blocksworldMatrixEnv import BlocksWorldMatrixEnv
from treeqn.nstep_learn import Learner, Runner

from sacred import Experiment
from sacred.arg_parser import parse_args
from sacred.commands import save_config, print_config
from sacred.observers import FileStorageObserver

logger = logging.getLogger('treeqn_log')


# Sacred setup to support config hierarchies
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


parsed_args = dict([x.split("=") for x in parse_args(sys.argv)["UPDATE"]])
if "config" in parsed_args:
    path = parsed_args["config"]
else:
    path = "./conf/default.yaml"


def fetch_parents(current_path, parents=[]):
    tmp_ex = Experiment('treeqn')
    tmp_ex.add_config(current_path)
    with suppress_stdout():
        tmp_ex.run("print_config")
    if tmp_ex.current_run is not None and "parent_config" in tmp_ex.current_run.config:
        return fetch_parents(tmp_ex.current_run.config["parent_config"], [current_path] + parents)
    else:
        return [current_path] + parents


configs = fetch_parents(path)
ex = Experiment('treeqn')
for path in configs:
    ex.add_config(path)

ex.logger = logger

ex.observers.append(FileStorageObserver.create('./results'))


@ex.config
def my_config(save_folder, env_id, architecture, label, name):
    pytorch_version = torch.__version__
    # Timestamp experiment directory
    save_folder = get_timestamped_dir(save_folder)

    # Environment switches
    # obs_dtype as str does the job and plays nice with sacred
    obs_dtype, input_mode = 'uint8', "atari"
    if "push" in env_id:
        obs_dtype, input_mode = 'float32', "push"
    if "blocksworld" in env_id:
        obs_dtype, input_mode = 'float32', "blocksworld"

    if architecture == "dqn":
        predict_rewards = False

    # Setup Logger
    Logger.DEFAULT = Logger.CURRENT = Logger(dir=save_folder, output_formats=[HumanOutputFormat(sys.stdout)])

    # Create monitor dir
    monitor_dir = os.path.join(save_folder, "%s-%s-%s" % (env_id, label, name))
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)


@ex.capture
def create_env(env_id, monitor_dir, num_cpu, frameskip, seed, _run):
    def make_env(rank, dir):
        def _thunk():
            if "push" in env_id:
                mode = env_id.split("-")[-1]
                env = Push(mode=mode)
            elif "blocksworld" in env_id:
                is_colorless = False
                game_level = 2
                env = BlocksWorldMatrixEnv(
                        "treeqn/envs/games/colorless_{}".format(is_colorless), 
                        game_level=game_level, 
                        is_colorless=is_colorless)
            else:
                env = gym.make(env_id + "NoFrameskip-v4")
            env.seed(seed + rank)
            filename = dir and os.path.join(dir, "%d.monitor.json" % rank)
            env = bench.Monitor(env, filename, cpu=rank)
            gym.logger.setLevel(logging.WARN)
            if "push" not in env_id and "blocksworld" not in env_id:
                env = wrap_deepmind(env, skip=frameskip)
            return env

        return _thunk

    vecenv = SubprocVecEnv([make_env(i, monitor_dir) for i in range(num_cpu)])
    return vecenv


@ex.capture
def create_model(env,
                 architecture,
                 gamma,
                 input_mode,
                 embedding_dim,
                 td_lambda,
                 use_actor_critic,
                 transition_fun_name,
                 transition_nonlin,
                 value_aggregation,
                 normalise_state,
                 residual_transition,
                 tree_depth,
                 predict_rewards,
                 nstack,
                 extra_layers,
                 nsteps):

    shared_policy_args = {
                          "embedding_dim": embedding_dim,
                          "use_actor_critic": use_actor_critic,
                          "input_mode": input_mode,
                          "gamma": gamma,
                          "predict_rewards": predict_rewards,
                          "value_aggregation": value_aggregation,
                          "td_lambda": td_lambda,
                          "normalise_state": normalise_state,
                          }

    if architecture == "dqn":
        policy = functools.partial(DQNPolicy,
                                   extra_layers=extra_layers,
                                   **shared_policy_args)
    elif architecture == "treeqn":
        policy = functools.partial(TreeQNPolicy,
                                   **shared_policy_args,
                                   transition_fun_name=transition_fun_name,
                                   transition_nonlin=transition_nonlin,
                                   residual_transition=residual_transition,
                                   tree_depth=tree_depth)
    else:
        raise ValueError("Architecture should be dqn or treeqn")

    model = policy(env.observation_space, env.action_space, env.num_envs, nsteps, nstack)
    return model


@ex.capture
def train(nstack,
          frameskip,
          use_actor_critic,
          rew_loss_coef,
          st_loss_coef,
          subtree_loss_coef,
          tree_depth,
          gamma,
          million_frames,
          eps_million_frames,
          nsteps,
          target_update_interval,
          alpha,
          epsilon,
          lr,
          lrschedule,
          max_grad_norm,
          vf_coef,
          ent_coef,
          log_interval,
          debug_log,
          number_checkpoints,
          log_rolling_window,
          save_folder,
          obs_dtype,
          monitor_dir,
          _run=None):

    # initialise environment and model
    env = create_env()
    model = create_model(env)
    nbatch = env.num_envs * nsteps

    # divide due to frameskip, then do a little extras so episodes end
    max_timesteps = int(1e6 * million_frames / frameskip * 1.1)
    number_updates = max_timesteps // nbatch + 1

    # initialise learner (computes loss and does training step)
    learner = Learner(model=model,
                      ent_coef=ent_coef,
                      vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm,
                      lr=lr,
                      alpha=alpha,
                      epsilon=epsilon,
                      number_updates=number_updates,
                      lrschedule=lrschedule,
                      use_actor_critic=use_actor_critic,
                      rew_loss_coef=rew_loss_coef,
                      st_loss_coef=st_loss_coef,
                      subtree_loss_coef=subtree_loss_coef,
                      nsteps=nsteps,
                      nenvs=env.num_envs,
                      tree_depth=tree_depth)

    # initialise runner (carries out interactions with environment)
    runner = Runner(env, learner, nsteps=nsteps, nstack=nstack, gamma=gamma, obs_dtype=obs_dtype,
                    eps_million_frames=eps_million_frames)

    tstart = time.time()

    rolling_rewards = []
    rolling_lengths = []

    if number_checkpoints > 0:
        checkpoint_interval = number_updates // number_checkpoints
    else:
        checkpoint_interval = 0

    # main training loop
    for update in range(1, number_updates):
        mb_data = runner.run()
        obs, next_obs, returns, rewards, masks, actions, values = mb_data
        policy_loss, value_loss, reward_loss, state_loss, subtree_loss, policy_entropy, grad_norm = learner.train(
            *mb_data)
        nseconds = time.time() - tstart
        fps = (update * nbatch) / nseconds

        # target net update
        if not use_actor_critic and update % (target_update_interval // nbatch) == 0:
            learner.target_model.load_state_dict(learner.model.state_dict())

        # logging
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            append_scalar(_run, "nupdates", update)

            total_timesteps = update * nbatch
            append_scalar(_run, "total_timesteps", total_timesteps)

            if debug_log:
                append_scalar(_run, "fps", int(fps))
                append_scalar(_run, "policy_entropy", float(policy_entropy))
                append_scalar(_run, "value_loss", float(value_loss))
                append_scalar(_run, "reward_loss", float(reward_loss))
                append_scalar(_run, "state_loss", float(state_loss))
                append_scalar(_run, "subtree_loss", float(subtree_loss))
                append_scalar(_run, "explained_variance", float(ev))

            rewards, lengths, steps = load_global_results(monitor_dir)

            if debug_log:
                append_list(_run, "rewards_raw", rewards)
                append_list(_run, "lengths_raw", lengths)
                append_list(_run, "steps_raw", steps)

            rolling_rewards.extend(zip(rewards, steps))
            rolling_lengths.extend(zip(lengths, steps))
            rolling_rewards = sorted(rolling_rewards, key=itemgetter(1))[-log_rolling_window:]
            rolling_lengths = sorted(rolling_lengths, key=itemgetter(1))[-log_rolling_window:]

            rewards = [x[0] for x in rolling_rewards]
            lengths = [x[0] for x in rolling_lengths]

            rewards_mean = np.mean(rewards) if len(rewards) > 0 else 0
            lengths_mean = np.mean(lengths) if len(lengths) > 0 else 0
            append_scalar(_run, "rewards_mean", rewards_mean)
            append_scalar(_run, "lengths_mean", lengths_mean)

            seconds_per_million = int(1e6 / fps)
            time_per_million = datetime.timedelta(seconds=seconds_per_million)
            if debug_log:
                append_scalar(_run, "seconds_per_million", seconds_per_million)
                append_scalar(_run, "time_per_million", time_per_million)
                append_scalar(_run, "grad_norm", grad_norm)

            print(" | ".join(["i: %8d", "m: %12s", "r: %10.2f", "l: %8.0f", "vl: %8.5f", "rl: %8.5f",
                              "sl: %8.5f", "sbtl: %8.5f", "p: %8.5f", "e: %8.5f", "ev: %6.4f", "gn: %6.4f"]) %
                  (total_timesteps, time_per_million, rewards_mean, lengths_mean, value_loss,
                   reward_loss, state_loss, subtree_loss, policy_loss, policy_entropy, ev, grad_norm))

        if number_checkpoints > 0 and (update % checkpoint_interval == 0
                                       or update == number_updates - 1):
            # Save model to file to save it in db
            name_model = save_folder + 'model_iteration_' + str(update)
            torch.save(learner.model.state_dict(), name_model)
            _run.add_artifact(name_model)
            model_size = os.path.getsize(name_model) / (1024 * 1024)
            os.remove(name_model)

            print("Model checkpoint saved. Size: {} MB".format(model_size))

    env.close()


@ex.automain
def main(_run, seed, save_folder, config_filename):
    set_global_seeds(seed)
    logger.info("Run id: {}".format(_run._id))

    print_config(ex.current_run)

    # saving config
    save_config(ex.current_run.config, ex.logger,
                config_filename=save_folder + config_filename)
    train()
