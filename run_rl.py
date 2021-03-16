from stable_baselines3 import PPO
# from stable_baselines3.common.base_class import BaseAlgorithm

from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams
from robots.legged_robots.robot_config import MotorControlMode
from tasks.test_task import TestTask

TIME_STEP = 1. / 1000.
TEST_OR_TRAIN = "train"
POLICY_SAVE_PATH = "/home/wsh/Documents/my_DRL_sim/data/policies/ppo_1.zip"


def main():
    test_or_train = TEST_OR_TRAIN
    assert test_or_train in ["train", "test"]
    gym_config = SimulationParameters(time_step=TIME_STEP)
    robot_class = QuadrupedRobot
    robot_params = MiniCheetahParams(on_rack=False, enable_self_collision=True,
                                     motor_control_mode=MotorControlMode.HYBRID_COMPUTED_POS)
    task = TestTask(train_or_test=TEST_OR_TRAIN)

    env = LocomotionGymEnv(gym_config, robot_class, robot_params, task)

    if TEST_OR_TRAIN == "train":
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save(POLICY_SAVE_PATH)
    else:
        model = PPO.load(POLICY_SAVE_PATH)
        obs = env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()
