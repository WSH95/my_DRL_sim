from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams
from robots.legged_robots.robot_config import MotorControlMode
from tasks.leggedRobot_test_task import TestLeg
from multiprocessing import Process
from utilities.debug_curve import DebugCurve
from utilities import global_values
import sys
from tasks.contact_fall_for_legged_robot import is_contact_fall


def curve():
    curve_drawing_obj = DebugCurve(
        global_values.q_curve,
        pause_time=0.0005,
        num_curves=2,
        x_length=1000,
        xlabel='time',
        ylabel='base height',
        title='base_height_curve',
        xlim=[-10, 1000],
        ylim=[-10, 600],
        grid=True)

    curve_drawing_obj.loop_func()


def main():
    time_step = 1. / 1000.
    gym_config = SimulationParameters(time_step=time_step)
    robot_class = QuadrupedRobot
    onRack = False

    robot_params = MiniCheetahParams(on_rack=onRack, enable_self_collision=True,
                                     motor_control_mode=MotorControlMode.HYBRID)
    env = LocomotionGymEnv(gym_config, robot_class, robot_params, task=None)
    pbClient = env.getClient()

    task = TestLeg(pbClient, on_rack=onRack)

    n = 0
    i = 0

    curve_drawing_proc = Process(target=curve, args=())
    curve_drawing_proc.start()

    while 1:
        H_desired, a = task.bodyHeight2Cmd()
        env.step(a)
        contact_fall = is_contact_fall(env)

        if n % 20 == 0:
            H_obs_cur = env.baseHeight

            global_values.q_curve.put((i, [H_desired * 1000, H_obs_cur * 1000]))

            i += 1

        n += 1

        curve_proc_terminate = global_values.global_userDebugParams.readValue("terminate_curve_process", 0)

        if curve_proc_terminate == 1 or contact_fall:
            # sys.exit(0)
            curve_drawing_proc.terminate()
        if curve_proc_terminate > 1:
            sys.exit(0)


if __name__ == '__main__':

    main()
