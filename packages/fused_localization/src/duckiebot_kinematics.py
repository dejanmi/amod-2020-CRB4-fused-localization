import math
import numpy as np
from geometry import SE2, SE2_from_xytheta, translation_from_SE2, angle_from_SE2


class DuckiebotKinematics:
    radius: float
    baseline: float
    max_number_ticks_encoder: int

    def __init__(self, radius: float, baseline: float) -> None:
        self.radius = radius
        self.baseline = baseline
        self.max_number_ticks_encoder = 135

    def step(self, number_ticks_left, number_ticks_right, pose1: SE2) -> SE2:
        d_l = 2*np.pi*self.radius*number_ticks_left / self.max_number_ticks_encoder
        d_r = 2*np.pi*self.radius*number_ticks_right / self.max_number_ticks_encoder
        d = (d_l + d_r)/2
        delta_theta = (d_r - d_l) / (self.baseline)
        translation1 = translation_from_SE2(pose1)
        theta1 = angle_from_SE2(pose1)
        theta2 = theta1 + delta_theta
        x2 = translation1[0] + d*math.cos(theta2)
        y2 = translation1[1] + d*math.sin(theta2)
        pose2 = SE2_from_xytheta([x2, y2, theta2])
        return pose2
