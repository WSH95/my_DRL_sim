from robots.legged_robots.robot_config import *
from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.leggedRobot import LeggedRobot
from tasks.leggedRobot_test_task import TestLeg
from envs.make_env_for_multiprocessing import env_change_input
from tasks.test2_single_joint_task import TestSingleJointTask
import os


class SingleJointParams(RobotSimParams):
    def __init__(self,
                 on_rack: bool = True,
                 enable_self_collision: bool = True,
                 # time_step: float = 1.0 / 240.0,
                 motor_control_mode: MotorControlMode = MotorControlMode.POSITION):
        self.urdf_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          '../urdf_files/single_joint_urdf/single_joint.urdf')
        self.joint_angle_MinMax = [[-np.pi * 2], [np.pi * 2]]
        self.joint_velocity_MinMax = [[-23.5], [23.5]]
        self.joint_torque_MinMax = [[-15], [15]]
        self.num_motors = 1
        self.dofs_per_leg = 1
        self.on_rack = on_rack
        self.enable_self_collision = enable_self_collision
        self.init_position = [0, 0, 0.5]
        self.init_orientation = [0, 0, 0, 1]
        self.init_motor_angles = [0]
        self.motor_offset = [0]
        # self.time_step = time_step
        self.motor_direction = [1]
        self.motor_control_mode = motor_control_mode
        self.motor_kp = [0]
        self.motor_kd = [0]

        self.sim_names = RobotSimNames()
        self.sim_names.motor_names = ["joint0"]
        self.sim_names.link_names = ["rotary_link"]
        self.sim_names.link_leg_names = ["rotary_link"]
        self.sim_names.link_single_leg_names = [["rotary_link"]]
        self.sim_names.link_disable_collision_names = [["rotary_link"]]
        self.sim_names.link_foot_names = []

        self.reset_at_current_pose = False
        self.control_latency = 0.00
        self.pd_latency = 0.00


def main():
    # gym_config = SimulationParameters(time_step=0.001,
    #                                   enable_rendering=True,
    #                                   enable_rendering_gui=True,
    #                                   egl_rendering=False)
    # robot_class = LeggedRobot
    # robot_params = SingleJointParams(motor_control_mode=MotorControlMode.HYBRID_COMPUTED_POS)
    # env = LocomotionGymEnv(gym_config, robot_class, robot_params)
    env_params = {'time_step': 0.001, 'robot_class': LeggedRobot, 'robot_config_class': SingleJointParams,
                  'task_class': TestSingleJointTask, 'on_rack': True, 'enable_self_collision': False,
                  'motor_control_mode': MotorControlMode.HYBRID_COMPUTED_POS_VEL, 'train_or_test': 'test'}
    env = env_change_input(**env_params)
    action_gen = TestLeg(env.getClient())

    while 1:
        a = action_gen.singleJointAction()
        env.step(a)


if __name__ == '__main__':
    main()
