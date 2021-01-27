from envs.locomotion_gym_env import LocomotionGymEnv
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.quadruped_robots.quadrupedRobot import QuadrupedRobot
from robots.legged_robots.quadruped_robots.miniCheetahParams import MiniCheetahParams
from robots.legged_robots.robot_config import MotorControlMode
from tasks.leggedRobot_test_task import TestLeg
from utilities.debug_curve import DebugCurve
from mpi4py import MPI
from utilities import global_values
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.rank
print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmy rank is : ", rank)


def curve():
    curve_drawing_obj = DebugCurve(
        comm=comm,
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
    env = LocomotionGymEnv(gym_config, robot_class, robot_params)
    pbClient = env.getClient()

    task = TestLeg(pbClient, on_rack=onRack)
    # curveProcID = pbClient.addUserDebugParameter("start/terminate curve process", 1, 0, 0)
    # global_values.global_userDebugParams.AddSlider("start/terminate curve process", 1, 0, 0)

    # a = pbClient.getConnectionInfo()
    # if a['connectionMethod'] == pbClient.GUI:
    #     print(f"connectionMethod: {a['connectionMethod']}")

    n = 0
    i = 0

    # curve_drawing_proc = Process(target=curve, args=())
    # curve_drawing_proc.start()
    # time_last = 0

    while 1:
        time_start = time.time()
        H_desired, a = task.bodyHeight2Cmd()
        env.step(a)

        if n % 20 == 0:
            H_obs_cur = env.baseHeight

            # global_values.q_curve.put((i, [H_desired * 1000, H_obs_cur * 1000]))
            curve_data = (i, [H_desired * 1000, H_obs_cur * 1000])
            comm.send(curve_data, dest=1, tag=1)
            i += 1

        n += 1

        time_spent = time.time() - time_start
        if i % 100 == 0:
            print(f"time spent is: {time_spent * 1000} ms.")
        # print(f"curve_proc_terminate: {curve_proc_terminate}")
        time_sleep = time_step - time_spent
        if time_sleep > 0:
            time.sleep(time_sleep)


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

if rank == 0:
    main()

if rank == 1:
    # data = comm.recv(source=0)
    curve()
