from robots.legged_robots.robot_config import *
from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.leggedRobot import LeggedRobot
from tasks.leggedRobot_test_task import TestLeg


class SingleJointParams(RobotSimParams):
    def __init__(self,
                 time_step: float = 1.0 / 240.0,
                 motor_control_mode: MotorControlMode = MotorControlMode.POSITION):
        self.urdf_filepath = "/home/wsh/Documents/my_DRL_sim/urdf_files/single_joint_urdf/single_joint.urdf"
        self.joint_angle_MinMax = [[-np.inf], [np.inf]]
        self.joint_velocity_MinMax = [[-np.inf], [np.inf]]
        self.joint_torque_MinMax = [[-np.inf], [np.inf]]
        self.num_motors = 1
        self.dofs_per_leg = 1
        self.on_rack = True
        self.enable_self_collision = False
        self.init_position = [0, 0, 0]
        self.init_orientation = [0, 0, 0, 1]
        self.init_motor_angles = [0]
        self.motor_offset = [0]
        self.time_step = time_step
        self.motor_direction = [-1]
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
        self.control_latency = 0.0
        self.pd_latency = 0.0


def main():
    gym_config = SimulationParameters()
    robot_class = LeggedRobot
    robot_params = SingleJointParams(motor_control_mode=MotorControlMode.HYBRID)
    env = LocomotionGymEnv(gym_config, robot_class, robot_params)
    task = TestLeg(env.getClient())

    while 1:
        a = task.singleJointAction()
        env.step(a)


if __name__ == '__main__':
    main()
