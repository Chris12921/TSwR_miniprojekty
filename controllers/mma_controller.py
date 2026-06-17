import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        self.models = [ManiuplatorModel(Tp), ManiuplatorModel(Tp), ManiuplatorModel(Tp)]

        self.models[0].m3 = 0.1
        self.models[0].r3 = 0.05
        self.models[0].I_3 = 2.0 / 5.0 * 0.1 * (0.05 ** 2)

        self.models[1].m3 = 0.01
        self.models[1].r3 = 0.01
        self.models[1].I_3 = 2.0 / 5.0 * 0.01 * (0.01 ** 2)

        self.models[2].m3 = 1.0
        self.models[2].r3 = 0.3
        self.models[2].I_3 = 2.0 / 5.0 * 1.0 * (0.3 ** 2)
        
        self.i = 0

        self.Kd = np.diag([20.0, 20.0])
        self.Kp = np.diag([100.0, 100.0])

    def choose_model(self, x):

        if len(x) > 4:
            x_real = x[:4]
            errors = []
            
            for j in range(3):
                x_mi = x[4 + j*4 : 8 + j*4]

                error = np.sum((x_real - x_mi)**2)
                errors.append(error)

            self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        q = x[:2]
        q_dot = x[2:4] 

        e = q_r - q
        e_dot = q_r_dot - q_dot
        v = q_r_ddot + self.Kd @ e_dot + self.Kp @ e

        M = self.models[self.i].M(x[:4])
        C = self.models[self.i].C(x[:4])
        
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u