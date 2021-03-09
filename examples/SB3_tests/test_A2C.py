import gym
from stable_baselines3 import A2C, PPO

from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams
from robots.legged_robots.robot_config import MotorControlMode
from tasks.test_task import TestTask

# env = gym.make('CartPole-v1')
time_step = 1. / 1000.
gym_config = SimulationParameters(time_step=time_step)
robot_class = QuadrupedRobot
onRack = False
robot_params = MiniCheetahParams(on_rack=onRack, enable_self_collision=True, motor_control_mode=MotorControlMode.POSITION)
task = TestTask()

env = LocomotionGymEnv(gym_config, robot_class, robot_params, task)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
