import numpy as np
from robots.motors.bldc_motor import HybridCommands
from utilities import global_values

BODY_HEIGHT_MIN = 0.080
BODY_HEIGHT_MAX = 0.415

THIGH_LEN = 0.2115
SHANK_LEN = 0.20477


class TestLeg:
    def __init__(self, pybullet_client, on_rack: bool = False):
        self._pybullet_client = pybullet_client
        self._on_rack = on_rack
        self._baseDof = 0 if self._on_rack else 6
        self._bodyH_min = BODY_HEIGHT_MIN
        self._bodyH_max = BODY_HEIGHT_MAX
        self._bodyH_lenth = BODY_HEIGHT_MAX - BODY_HEIGHT_MIN
        # self._bodyHeightId = self._pybullet_client.addUserDebugParameter("setBodyHeight", -1, 1, 0.6)
        # self._kp_ID = self._pybullet_client.addUserDebugParameter("kp", 0, 1, 0.353)
        # self._kd_ID = self._pybullet_client.addUserDebugParameter("kd", 0, 1, 0.758)

        self.action = HybridCommands(torque_estimate=None,
                                     position_desired=None,
                                     velocity_desired=None,
                                     kp=0.0,
                                     kd=0.0)

    def bodyHeight2Cmd(self):
        # e = self._pybullet_client.readUserDebugParameter(self._bodyHeightId)
        e = global_values.global_userDebugParams.readValue("setBodyHeight", 0.6)
        H = self._bodyH_min + self._bodyH_lenth * (1 - np.abs(e))
        cos1 = (np.power(H, 2) + np.power(THIGH_LEN, 2) - np.power(SHANK_LEN, 2)) / (2 * H * THIGH_LEN)
        cos2 = (np.power(THIGH_LEN, 2) + np.power(SHANK_LEN, 2) - np.power(H, 2)) / (2 * SHANK_LEN * THIGH_LEN)
        theta1 = np.arccos(cos1) * (1 if e > 0 else -1)
        theta2 = (np.pi - np.arccos(cos2)) * (-1 if e > 0 else 1)

        self.action.torque_estimate = None  # [0] * 12
        self.action.velocity_desired = [0] * ((self._baseDof if self.action.torque_estimate is None else 0) + 12)
        if self._on_rack or self.action.torque_estimate is not None:
            self.action.position_desired = [0, 1 * theta1, 1 * theta2] * 4
        else:
            self.action.position_desired = [None, None, H + 0.025, 0, 0, 0, 1] + [0, 1 * theta1, 1 * theta2] * 4
        # kp = self._pybullet_client.readUserDebugParameter(self._kp_ID) * 2000
        # kd = self._pybullet_client.readUserDebugParameter(self._kd_ID) * 50
        kp = global_values.global_userDebugParams.readValue("kp", 0.353) * 2000
        kd = global_values.global_userDebugParams.readValue("kd", 0.758) * 50
        self.action.kp = kp
        self.action.kd = kd
        return H, np.array([0, 1 * theta1, 1 * theta2] * 4 + [0] * 12) / (2 * np.pi)

    def singleJointAction(self):
        # e = self._pybullet_client.readUserDebugParameter(self._bodyHeightId)
        e = global_values.global_userDebugParams.readValue("setBodyHeight")
        tor = None
        # pos = np.array([e * 2 * np.pi])
        pos = np.array([e])
        vel = np.array([0])
        kp = 0.8
        kd = 0.05
        self.action.torque_estimate = tor
        self.action.position_desired = pos
        self.action.velocity_desired = vel
        self.action.kp = kp
        self.action.kd = kd

        return np.concatenate((self.action.position_desired, self.action.velocity_desired))
        # return self.action.position_desired
