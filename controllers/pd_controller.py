import numpy as np
from .controller import Controller

class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[:2]
        q_dot = x[2:4]
        
        e = q_d - q
        e_dot = q_d_dot - q_dot

        v = q_d_ddot + self.kd @ e_dot + self.kp @ e
        u = v[:, np.newaxis]
        
        return u
