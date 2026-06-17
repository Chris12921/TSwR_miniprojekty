import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]])
        B = np.array([[0.0],
                      [self.b],
                      [0.0]])
        W = np.array([[1.0, 0.0, 0.0]])

        L = np.array([[3.0 * p],
                      [3.0 * p**2],
                      [p**3]])

        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        B = np.array([[0.0],
                      [self.b],
                      [0.0]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[0]

        z = self.eso.get_state()
        q_est = z[0]
        q_dot_est = z[1]
        f_est = z[2]

        # kompensacja zakłóceń
        v = q_d_ddot + self.kd * (q_d_dot - q_dot_est) + self.kp * (q_d - q)
        u = (v - f_est) / self.b

        self.eso.update(q, u)
        
        return u 