import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# import i komentarze sa do taska 9
from models.manipulator_model import ManiuplatorModel

class ADRController(Controller):
    def __init__(self, Tp, params):
        # wprowadzamy model
        self.model = ManiuplatorModel(Tp)
        
        self.joint_controllers = []
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        # pobieramy M, odwracamy ja i beirzemy b z przekatnej
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        self.joint_controllers[0].set_b(M_inv[0, 0])
        self.joint_controllers[1].set_b(M_inv[1, 1])

        u = []
        for i, controller in enumerate(self.joint_controllers):
            u.append(controller.calculate_control([x[i], x[i+2]], q_d[i], q_d_dot[i], q_d_ddot[i]))
        u = np.array(u)[:, np.newaxis]
        return u

