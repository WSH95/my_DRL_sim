import math
from robots.legged_robots.robot_config import *
from robots.legged_robots.leggedRobot import LeggedRobot
import numpy as np

PI = math.pi


class QuadrupedRobot(LeggedRobot):
    def __init__(self,
                 pybullet_client,
                 robot_params: RobotSimParams,
                 time_step: float = 1.0 / 240.0):

        super(QuadrupedRobot, self).__init__(
            pybullet_client=pybullet_client,
            robot_params=robot_params,
            time_step=time_step)

