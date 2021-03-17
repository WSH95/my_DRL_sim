import os
from robots.legged_robots.robot_config import *

# MINI_CHEETAH_URDF_FILE_PATH = "/home/wsh/Documents/my_DRL_sim/urdf_files/mit_mini_cheetah/mit_mini_cheetah_toes.urdf"
MINI_CHEETAH_URDF_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           '../../../urdf_files/mit_mini_cheetah/mit_mini_cheetah_toes.urdf')
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
LINK_SINGLE_LEG_NAMES = [LINK_NAMES[:4], LINK_NAMES[4:8], LINK_NAMES[8:12], LINK_NAMES[12:]]
LINK_DISABLE_COLLISION_NAMES = [["base"] + names for names in LINK_SINGLE_LEG_NAMES] + [["base"] + [name] for name in LINK_NAMES]

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
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_1",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_2",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_3",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_4",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_5",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_6",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_7",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_8",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_9",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_10",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi),
        ScalarField(name="motor_angle_11",
                    upper_bound=2 * np.pi,
                    lower_bound=-2 * np.pi)
    ]

    JOINT_VELOCITY_LIMIT = [
        ScalarField(name="motor_velocity_0",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_1",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_2",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_3",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_4",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_5",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_6",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_7",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_8",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_9",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_10",
                    upper_bound=23.5,
                    lower_bound=-23.5),
        ScalarField(name="motor_velocity_11",
                    upper_bound=23.5,
                    lower_bound=-23.5)
    ]

    JOINT_TORQUE_LIMIT = [
        ScalarField(name="motor_torque_0",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_1",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_2",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_3",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_4",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_5",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_6",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_7",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_8",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_9",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_10",
                    upper_bound=35,
                    lower_bound=-35),
        ScalarField(name="motor_torque_11",
                    upper_bound=35,
                    lower_bound=-35)
    ]

    def __init__(self,
                 on_rack: bool = True,
                 enable_self_collision: bool = True,
                 # time_step: float = 1.0 / 240.0,
                 motor_control_mode: MotorControlMode = MotorControlMode.POSITION):
        self.joint_angle_MinMax = [[elem.lower_bound for elem in self.JOINT_ANGLE_LIMIT],
                                   [elem.upper_bound for elem in self.JOINT_ANGLE_LIMIT]]
        self.joint_velocity_MinMax = [[elem.lower_bound for elem in self.JOINT_VELOCITY_LIMIT],
                                      [elem.upper_bound for elem in self.JOINT_VELOCITY_LIMIT]]
        self.joint_torque_MinMax = [[elem.lower_bound for elem in self.JOINT_TORQUE_LIMIT],
                                    [elem.upper_bound for elem in self.JOINT_TORQUE_LIMIT]]

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
        self.sim_names.link_single_leg_names = LINK_SINGLE_LEG_NAMES
        self.sim_names.link_disable_collision_names = LINK_DISABLE_COLLISION_NAMES

        self.reset_at_current_pose = False
        self.control_latency = 0.0
        self.pd_latency = 0.0
