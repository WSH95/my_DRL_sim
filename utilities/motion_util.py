from pybullet_utils import transformations
import numpy as np
import math


def NormalizeAngle2Pi(theta):
    if -np.pi <= theta <= np.pi:
        return theta
    else:
        norm_theta = np.fmod(theta, 2 * np.pi)
        if norm_theta < -np.pi:
            norm_theta += 2 * np.pi
        elif norm_theta > np.pi:
            norm_theta += -2 * np.pi
        return norm_theta


def TransOfQuaternionsRef2World(quat_from, quat_to):
    assert len(quat_from) == 4
    assert len(quat_to) == 4
    return transformations.quaternion_multiply(quat_to, transformations.quaternion_conjugate(quat_from))


def Quaternion2AxisAngle(quat):
    assert len(quat) == 4
    if not np.isclose(1.0, np.linalg.norm(quat)):
        raise ValueError(f"Quaternion should have unit length: |q| = {np.linalg.norm(quat)}, q = {quat}")

    axis = quat[:3].copy()
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        axis = np.array([0, 0, 1], dtype=np.float64)
    else:
        axis /= axis_norm

    sin_half_angle = axis_norm
    cos_half_angle = quat[3]
    half_angle = math.atan2(sin_half_angle, cos_half_angle)
    angle = 2 * half_angle

    return axis, angle
