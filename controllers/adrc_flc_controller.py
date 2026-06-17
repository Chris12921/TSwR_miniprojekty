import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        
        p1 = p[0]
        p2 = p[1]

        self.L = np.array([
            [3*p1, 0],
            [0, 3*p2],
            [3*p1**2, 0],
            [0, 3*p2**2],
            [p1**3, 0],
            [0, p2**3]
        ])
        
        W = np.hstack([np.eye(2), np.zeros((2, 4))])
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))

        z0 = np.concatenate([q0, np.zeros(2)]) 
        self.eso = ESO(A, B, W, self.L, z0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.concatenate([q, q_dot])
        M = self.model.M(x)
        C = self.model.C(x)
        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)
        self.eso.A = A
 
        B = np.zeros((6, 2))
        B[2:4, 0:2] = M_inv
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[:2]

        z = self.eso.get_state()
        q_est = z[0:2]
        q_dot_est = z[2:4]
        f_est = z[4:6]

        e = q_d - q
        e_dot = q_d_dot - q_dot_est
        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        x_est = np.concatenate([q_est, q_dot_est])
        M = self.model.M(x_est)
        C = self.model.C(x_est)
        
        u = M @ (v - f_est) + C @ q_dot_est

        self.update_params(q_est, q_dot_est)
        self.eso.update(q, u)
        
        return u