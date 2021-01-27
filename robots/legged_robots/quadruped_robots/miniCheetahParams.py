import numpy as np
from robots.legged_robots.robot_config import *

MINI_CHEETAH_URDF_FILE_PATH = "/home/wsh/Documents/my_DRL_sim/urdf_files/mit_mini_cheetah/mit_mini_cheetah_toes.urdf"
MINI_CHEETAH_NUM_MOTORS = 12
MINI_CHEETAH_DOFS_PER_LEG = 3
MINI_CHEETAH_NUM_LEGS = 4

MOTOR_NAMES = [
    "torso_to_abduct_fl_j", "abduct_fl_to_thigh_fl_j", "thigh_fl_to_knee_fl_j",
    "torso_to_abduct_fr_j", "abduct_fr_to_thigh_fr_j", "thigh_fr_to_knee_fr_j",
    "torso_to_abduct_hl_j", "abduct_hl_to_thigh_hl_j", "thigh_hl_to_knee_hl_j",
    "torso_to_abduct_hr_j", "abduct_hr_to_thigh_hr_j", "thigh_hr_to_knee_hr_j"]
LINK_NAMES = [
    "abduct_fl", "thigh_fl", "shank_fl", "toe_fl",
    "abduct_fr", "thigh_fr", "shank_fr", "toe_fr",
    "abduct_hl", "thigh_hl", "shank_hl", "toe_hl",
    "abduct_hr", "thigh_hr", "shank_hr", "toe_hr"]
LINK_LEG_NAMES = [
    "thigh_fl", "shank_fl",
    "thigh_fr", "shank_fr",
    "thigh_hl", "shank_hl",
    "thigh_hr", "shank_hr"]
LINK_FOOT_NAMES = ["toe_fl", "toe_fr", "toe_hl", "toe_hr"]

MINI_CHEETAH_INIT_RACK_POSITION = [0, 0, 1]
MINI_CHEETAH_INIT_POSITION = [0, 0, 0.281]
MINI_CHEETAH_INIT_ORIENTATION = [0, 0, 0, 1]

MINI_CHEETAH_DEFAULT_ABDUCTION_ANGLE = 0
MINI_CHEETAH_DEFAULT_HIP_ANGLE = 1.0041  # 0.67
MINI_CHEETAH_DEFAULT_KNEE_ANGLE = -2.0621  # -1.25
MINI_CHEETAH_INIT_MOTOR_ANGLES = np.array(
    [MINI_CHEETAH_DEFAULT_ABDUCTION_ANGLE, MINI_CHEETAH_DEFAULT_HIP_ANGLE, MINI_CHEETAH_DEFAULT_KNEE_ANGLE] * 4)

MINI_CHEETAH_ABDUCTION_OFFSET = 0.0
MINI_CHEETAH_HIP_JOINT_OFFSET = 0.0
MINI_CHEETAH_KNEE_JOINT_OFFSET = 0.0
MINI_CHEETAH_MOTOR_OFFSET = np.array(
    [MINI_CHEETAH_ABDUCTION_OFFSET, MINI_CHEETAH_HIP_JOINT_OFFSET, MINI_CHEETAH_KNEE_JOINT_OFFSET] * 4)

MINI_CHEETAH_MOTOR_DIRECTION = np.array([-1] * MINI_CHEETAH_NUM_MOTORS)

MINI_CHEETAH_ABDUCTION_P_GAIN = 220.0
MINI_CHEETAH_ABDUCTION_D_GAIN = 0.3
MINI_CHEETAH_HIP_P_GAIN = 220.0
MINI_CHEETAH_HIP_D_GAIN = 2.0
MINI_CHEETAH_KNEE_P_GAIN = 220.0
MINI_CHEETAH_KNEE_D_GAIN = 2.0

UPPER_BOUND = 2 * np.pi
LOWER_BOUND = -2 * np.pi


class MiniCheetahParams(RobotSimParams):
    JOINT_ANGLE_LIMIT = [
        ScalarField(name="motor_angle_0",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_1",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_2",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_3",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_4",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_5",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_6",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_7",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_8",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_9",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_10",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND),
        ScalarField(name="motor_angle_11",
                    upper_bound=UPPER_BOUND,
                    lower_bound=LOWER_BOUND)
    ]

    def __init__(self,
                 on_rack: bool = True,
                 enable_self_collision: bool = True,
                 # time_step: float = 1.0 / 240.0,
                 motor_control_mode: MotorControlMode = MotorControlMode.POSITION):
        self.urdf_filepath = MINI_CHEETAH_URDF_FILE_PATH
        self.num_motors = MINI_CHEETAH_NUM_MOTORS
        self.dofs_per_leg = MINI_CHEETAH_DOFS_PER_LEG
        self.on_rack = on_rack
        self.enable_self_collision = enable_self_collision
        self.init_position = MINI_CHEETAH_INIT_RACK_POSITION if on_rack else MINI_CHEETAH_INIT_POSITION
        self.init_orientation = MINI_CHEETAH_INIT_ORIENTATION
        self.init_motor_angles = MINI_CHEETAH_INIT_MOTOR_ANGLES
        self.motor_offset = MINI_CHEETAH_MOTOR_OFFSET
        # self.time_step = time_step
        self.motor_direction = MINI_CHEETAH_MOTOR_DIRECTION
        self.motor_control_mode = motor_control_mode
        self.motor_kp = [MINI_CHEETAH_ABDUCTION_P_GAIN, MINI_CHEETAH_HIP_P_GAIN, MINI_CHEETAH_KNEE_P_GAIN] * 4
        self.motor_kd = [MINI_CHEETAH_ABDUCTION_D_GAIN, MINI_CHEETAH_HIP_D_GAIN, MINI_CHEETAH_KNEE_D_GAIN] * 4

        self.sim_names = RobotSimNames()
        self.sim_names.motor_names = MOTOR_NAMES
        self.sim_names.link_names = LINK_NAMES
        self.sim_names.link_leg_names = LINK_LEG_NAMES
        self.sim_names.link_foot_names = LINK_FOOT_NAMES
