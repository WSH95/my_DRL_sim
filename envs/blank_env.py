from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, Any
import gym
from gym.spaces import Box
import numpy as np
from typing import Optional, Sequence


class BlankVecEnv(VecEnv):
    metadata = {}

    def __init__(self):
        obs_space = Box(-np.full(1, np.inf), np.full(1, np.inf), dtype=np.float32)
        act_space = Box(-np.full(1, np.inf), np.full(1, np.inf), dtype=np.float32)

        super(BlankVecEnv, self).__init__(1, obs_space, act_space)

    def reset(self):
        return np.array([-0.5])

    def step_async(self, actions: np.ndarray) -> None:
        pass

    def step_wait(self):
        return np.array([-0.5]), 0, False, []

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs):
        pass

    def seed(self, seed: Optional[int] = None):
        return [1]

    def get_images(self) -> Sequence[np.ndarray]:
        pass


class BlankGymEnv(gym.Env):
    action_space = Box(-np.full(1, np.inf), np.full(1, np.inf), dtype=np.float32)
    observation_space = Box(-np.full(1, np.inf), np.full(1, np.inf), dtype=np.float32)

    def step(self, action):
        return np.array([-0.5]), 0, False, {}

    def reset(self):
        return np.array([-0.5])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return [1]
