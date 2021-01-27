import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)  # 连接到仿真服务器
p.setGravity(0, 0, -9.8)  # 设置重力值
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置pybullet_data的文件路径

# 加载地面
floor = p.loadURDF("plane_implicit.urdf")

startPos = [0, 0, 0]

button = p.addUserDebugParameter("value", -1, 1, 0)

# 加载urdf文件
robot = p.loadURDF("/home/wsh/Documents/my_DRL_sim/urdf_files/single_joint_urdf/single_joint.urdf",
                   startPos,
                   useFixedBase=1,
                   flags=p.URDF_USE_INERTIA_FROM_FILE)

p.resetDebugVisualizerCamera(0.3, 0, -60, [0, 0, 0])
p.changeVisualShape(robot, 0, rgbaColor=[174.0/255.0, 187.0/255.0, 143.0/255.0, 0.7])

time_step = 0.001  # 1.0 / 240.0
p.setTimeStep(time_step)
p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, 0, force=10)
aa = p.getNumConstraints()


def update_obs():
    joint_state = p.getJointState(robot, 0)
    pos = joint_state[0]
    vel = joint_state[1]

    return pos, vel


def position_ctl(val):
    pos = val * 2 * np.pi
    p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, pos, force=10)


def velocity_ctl(val):
    vel = val * 2 * np.pi
    p.setJointMotorControl2(robot, 0, p.VELOCITY_CONTROL, targetVelocity=vel, force=10)


def torque_ctl(val):
    torque = val * 0.01
    p.setJointMotorControl2(robot, 0, p.TORQUE_CONTROL, force=torque)


def pd_vel_ctl(val):
    pos = val * 2 * np.pi
    p.setJointMotorControl2(robot, 0, p.PD_CONTROL, pos, targetVelocity=0, positionGain=0.08, velocityGain=0.000)


def hybrid_ctl(val, curr_pos, curr_vel):
    tor = 0
    pos = val * 2 * np.pi
    vel = 0
    kp = 1.1
    kd = 0.05
    t = tor + kp * (pos - curr_pos) + kd * (vel - curr_vel)
    p.setJointMotorControlArray(robot, np.asarray([0]), p.TORQUE_CONTROL, forces=np.asarray([t]))


# p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, force=0)
i = 0

while True:
    i += 1
    if i < 0:
        position_ctl(i / 100.0)
    else:
        # p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, force=0)
        current_val = p.readUserDebugParameter(button)
        current_joint_pos, current_joint_vel = update_obs()
        hybrid_ctl(current_val, current_joint_pos, current_joint_vel)

    p.stepSimulation()
    time.sleep(time_step)
