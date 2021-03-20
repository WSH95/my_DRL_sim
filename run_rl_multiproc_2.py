from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from envs.make_env_for_multiprocessing import env_change_input
from robots.legged_robots.robot_config import MotorControlMode
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from multiprocessing import Process, freeze_support, set_start_method
from tasks.test_task import TestTask
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams

# Single joint
from examples.test2_single_joint import SingleJointParams
from robots.legged_robots.leggedRobot import LeggedRobot
from tasks.test2_single_joint_task import TestSingleJointTask

import os
import time


def make_env(env_param, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_param: (dict) the environment params
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        e = env_change_input(**env_param)
        e.seed(seed + rank)
        return e
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    freeze_support()

    ROBOT = 'SingleJoint'
    assert ROBOT in ['SingleJoint', 'WholeBody']
    TIME_STEP = 1. / 1000.
    TEST_OR_TRAIN = "train"
    # TEST_OR_TRAIN = "test"
    NUM_CPUS = 7
    COUNT = 1

    test_or_train = TEST_OR_TRAIN
    assert test_or_train in ["train", "test"]

    env_params = None
    policy_kwargs = None

    if ROBOT is 'SingleJoint':
        env_params = {'time_step': 0.001, 'robot_class': LeggedRobot, 'robot_config_class': SingleJointParams,
                      'task_class': TestSingleJointTask, 'on_rack': True, 'enable_self_collision': False,
                      'motor_control_mode': MotorControlMode.HYBRID_COMPUTED_POS_VEL, 'train_or_test': test_or_train}
        policy_kwargs = {"net_arch": [{"pi": [32, 16], "vf": [32, 16]}]}
    elif ROBOT is 'WholeBody':
        env_params = {'time_step': TIME_STEP, 'robot_class': QuadrupedRobot, 'robot_config_class': MiniCheetahParams,
                      'task_class': TestTask, 'on_rack': False, 'enable_self_collision': True,
                      'motor_control_mode': MotorControlMode.HYBRID_COMPUTED_POS_VEL, 'train_or_test': test_or_train}
        policy_kwargs = {"net_arch": [{"pi": [256, 128], "vf": [256, 128]}]}

    curr_time = time.strftime("%d-%m-%Y_%H-%M-%S")
    policy_save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/policies')

    policy_save_path: str = ''
    env_stats_path: str = ''

    if TEST_OR_TRAIN == "train":
        policy_save_filename = 'ppo_model_' + str(COUNT) + '_S_PV_4096_700w_' + curr_time + '.zip'
        policy_save_path = os.path.join(policy_save_dir, policy_save_filename)

        env_stats_filename = 'ppo_env_' + str(COUNT) + '_S_PV_4096_700w_' + curr_time + '.pkl'
        env_stats_path = os.path.join(policy_save_dir, env_stats_filename)

        # env = make_vec_env(env_change_input, n_envs=NUM_CPUS, seed=0, env_kwargs=env_params, vec_env_cls=None)
        env = SubprocVecEnv([make_env(env_params, i) for i in range(NUM_CPUS)])
        env = VecNormalize(env, norm_reward=True)
        if not (os.path.exists(policy_save_dir)):
            os.makedirs(policy_save_dir)
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
        # model.set_parameters(os.path.join(policy_save_dir, 'ppo_17_S_PV_4096_4000w_20-03-2021_07-20-24.zip'))
        t1 = time.time()
        model.learn(total_timesteps=7000000)
        print(f"The training spent {time.time() - t1} s.")
        model.save(policy_save_path)
        env.save(env_stats_path)
    else:
        # env = env_change_input(time_step=env_params['time_step'],
        #                        robot_class=env_params['robot_class'],
        #                        on_rack=env_params['on_rack'],
        #                        enable_self_collision=env_params['enable_self_collision'],
        #                        motor_control_mode=env_params['motor_control_mode'],
        #                        train_or_test=env_params['train_or_test'])
        # env = env_change_input(**env_params)

        env = SubprocVecEnv([lambda: env_change_input(**env_params)])
        env_stats_load_path = os.path.join(policy_save_dir, 'ppo_env_1_S_PV_0.6_10w_20-03-2021_22-07-27.pkl')
        env = VecNormalize.load(env_stats_load_path, env)
        env.training = False
        env.norm_reward = False

        model_load_path = os.path.join(policy_save_dir, 'ppo_1_S_PV_0.6_10w_20-03-2021_22-07-27.zip')
        model = PPO.load(model_load_path, env=env)
        obs = env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                obs = env.reset()
