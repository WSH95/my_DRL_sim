from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed
from envs.locomotion_gym_env import LocomotionGymEnv
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams
from tasks.test_task import TestTask


def env_change_input(time_step,
                     robot_class,
                     on_rack,
                     enable_self_collision,
                     motor_control_mode,
                     train_or_test):

    if train_or_test == 'train':
        gym_config = SimulationParameters(time_step=time_step)
    else:
        gym_config = SimulationParameters(time_step=time_step,
                                          enable_rendering=True,
                                          enable_rendering_gui=True,
                                          egl_rendering=False)

    # robotClass = QuadrupedRobot if (robot_class == 'quadruped') else None
    robotClass = robot_class
    robot_params = MiniCheetahParams(on_rack=on_rack,
                                     enable_self_collision=enable_self_collision,
                                     motor_control_mode=motor_control_mode)
    task = TestTask(train_or_test=train_or_test)

    env = LocomotionGymEnv(gym_config, robotClass, robot_params, task)

    return env
