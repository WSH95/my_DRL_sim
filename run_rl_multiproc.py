from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.make_env_for_multiprocessing import env_change_input
from robots.legged_robots.robot_config import MotorControlMode

import os
import time

TIME_STEP = 1. / 1000.
TEST_OR_TRAIN = "train"
NUM_CPUS = 24
COUNT = 5


def main():
    test_or_train = TEST_OR_TRAIN
    assert test_or_train in ["train", "test"]

    env_params = {'time_step': TIME_STEP, 'robot_class': 'quadruped', 'on_rack': False, 'enable_self_collision': True,
                  'motor_control_mode': MotorControlMode.HYBRID_COMPUTED_POS_TROT, 'train_or_test': test_or_train}

    policy_save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/policies')
    policy_save_filename = 'ppo_' + str(COUNT) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    policy_save_path = os.path.join(policy_save_dir, policy_save_filename)

    if TEST_OR_TRAIN == "train":
        env = make_vec_env(env_change_input, n_envs=NUM_CPUS, seed=0, env_kwargs=env_params)
        if not (os.path.exists(policy_save_dir)):
            os.makedirs(policy_save_dir)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100000000)
        model.save(policy_save_path)
    else:
        env = env_change_input(time_step=env_params['time_step'],
                               robot_class=env_params['robot_class'],
                               on_rack=env_params['on_rack'],
                               enable_self_collision=env_params['enable_self_collision'],
                               motor_control_mode=env_params['motor_control_mode'],
                               train_or_test=env_params['train_or_test'])
        model_load_path = os.path.join(policy_save_dir, 'ppo_3_17-03-2021_15-39-42')
        model = PPO.load(model_load_path)
        obs = env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()
