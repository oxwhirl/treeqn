#!/usr/bin/env python

import os
import math
import numpy as np
import argparse
import gym
from gym import spaces
from myTorch.environment import EnivironmentBase
from myTorch.environment.BlocksworldMatrix import BlocksWorld, WorldBuilder


class BlocksWorldMatrixEnv(gym.Env):
    def __init__(self, game_base_dir, height=10, width=10, num_steps_cutoff=80, mode="train", game_level=2, start_game_id=0, max_games_per_level=10000, seed=1234, is_one_hot_world=False, max_num_blocks=20, id_normalizing_factor=20, is_colorless=False):
        self._height = height
        self._width = width
        self._is_one_hot_world = is_one_hot_world
        self._actions = {0:'left',1:'right',2:'pick',3:'drop'}
        self._legal_moves = np.array(list(self._actions.keys()))
        self._num_steps_cutoff = num_steps_cutoff
        self._numpy_rng = np.random.RandomState(seed=seed)

        self._mode = mode
        self._is_colorless = is_colorless
        self._game_level = game_level
        self._world_builder = WorldBuilder(game_base_dir, max_games_per_level=max_games_per_level)
        self._game_id = start_game_id
        self._game_id_sequence = np.arange(max_games_per_level)
        self._numpy_rng.shuffle(self._game_id_sequence)

        self._id_normalizing_factor = id_normalizing_factor
        self._max_num_blocks = max_num_blocks
        if self._is_colorless:
            self._object_ids = np.array([1]+[2]*self._max_num_blocks)
        else:
            self._object_ids = np.array([float(i)/self._id_normalizing_factor \
                for i in range(1, self._id_normalizing_factor)])

        self.load_games()

    def reset(self, game_level=None):

        if game_level is not None:
            if game_level != self._game_level:
                self._game_level = game_level
                self.load_games()
                self._game_id = 0
                self._numpy_rng.shuffle(self._game_id_sequence)

        game = self._games[self._game_id_sequence[self._game_id]]

        self._input_world = BlocksWorld(self._height, self._width, self._max_num_blocks, 
                                        is_agent_present=True, is_colorless=self._is_colorless)
        self._target_world = BlocksWorld(self._height, self._width, self._max_num_blocks, 
                                         is_agent_present=False, is_colorless=self._is_colorless)
        self._numpy_rng.shuffle(self._object_ids[1:])
        self._target_world.reset(blocks_info=game["target_towers"], object_ids=self._object_ids)
        self._input_world.reset(blocks_info=game["input_towers"], object_ids=self._object_ids,
                                order_look_up=self._target_world.order.order_look_up,
                                target_height_at_loc=self._target_world.height_at_loc)
        combine_op = np.concatenate if self._is_one_hot_world else np.stack
        obs = combine_op((self._input_world.as_numpy(self._is_one_hot_world), self._target_world.as_numpy(self._is_one_hot_world)), axis=0)
        self._game_id = (self._game_id + 1) % self._num_available_games
        self._num_steps_done = 0
        return obs

    @property
    def have_games_exhausted(self):
        return (self._game_id == self._num_available_games - 1)

    def step(self, action):
        self._num_steps_done += 1
        reward, done = self._input_world.update(self._actions[action])
        combine_op = np.concatenate if self._is_one_hot_world else np.stack
        obs = combine_op((self._input_world.as_numpy(self._is_one_hot_world), self._target_world.as_numpy(self._is_one_hot_world)), axis=0)
        if self._num_steps_done >= self._num_steps_cutoff:
            done = True
        return obs, reward, done, {}

    def create_games(self, num_levels):
        for level in range(2, num_levels+2):
            self._world_builder.create_games(game_level=level)

    def load_games(self):
        self._games = self._world_builder.load_games(self._game_level, self._mode)
        self._num_available_games = len(self._games)
        if self._mode != "train":
            self._num_available_games = int((0.002*self._num_available_games))

    @property
    def action_dim(self):
        return len(self._actions) 

    @property
    def action_space(self):
        return spaces.Discrete(len(self._actions))
    
    @property
    def observation_space(self):
        return spaces.Box(0, 1, (2, self._width, self._height), np.float32)
    
    @property
    def obs_dim(self):
        return (2 if not self._is_one_hot_world else self._max_num_blocks + 1, self._width, self._height)

    @property
    def input_world(self):
        return self._input_world

    @property
    def target_world(self):
        return self._target_world

    @property
    def max_episode_len(self):
        return self._num_steps_cutoff
    
    def render(self, mode='rgb_array'):
        pass

    def seed(self, seed):
        self._numpy_rng = np.random.RandomState(seed=seed)
        self._numpy_rng.shuffle(self._game_id_sequence)

    def get_random_state(self):
        pass

    def set_random_state(self, state):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Environment")
    parser.add_argument('--num_games', type=int, default=0, help="config name")
    parser.add_argument('--colorless', type=bool, default=False, help="is colorless world")
    args = parser.parse_args()

    env = BlocksWorldMatrixEnv(game_base_dir="games/colorless_{}/".format(args.colorless), game_level=4, is_colorless=args.colorless)
    if args.num_games > 0:
        env.create_games(args.num_games)
        import pdb; pdb.set_trace()
    env.reset()
    print("Target World Down:")
    print((np.flipud(np.transpose(env.target_world.as_numpy()))))
    print("Input World Down:")
    print((np.flipud(np.transpose(env.input_world.as_numpy()))))
    print(("Agent loc : {}".format(env.input_world.agent.loc)))

    action_dict = {"l":0, "r":1, "p":2, "d":3}
    while True:
        action = input("Action: Choose among: l,r,p,d \n")
        if action in action_dict:
            obs, _, reward, done = env.step(action_dict[action])
            print("Target World Down:")
            print((np.flipud(np.transpose(obs[1]))))
            print("Input World Down:")
            print((np.flipud(np.transpose(obs[0]))))
            print(("Reward : {}, done : {}".format(reward,done)))
            print(("Agent loc : {}".format(env.input_world.agent.loc)))

            if done:
                print("GAME OVER !!")
                obs, _ = env.reset()
                print("Target World Down:")
                print((np.flipud(np.transpose(obs[1]))))
                print("Input World Down:")
                print((np.flipud(np.transpose(obs[0]))))
                print(("Agent loc : {}".format(env.input_world.agent.loc)))
