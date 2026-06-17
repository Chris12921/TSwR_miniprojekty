import numpy as np

class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 10.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.10
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        
        alpha = self.m1 * d1**2 + self.I_1 + self.m2 * (self.l1**2 + d2**2) + self.I_2 + self.m3 * (self.l1**2 + self.l2**2) + self.I_3
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3
        
        c2 = np.cos(q2)
        
        M11 = alpha + 2 * beta * c2
        M12 = gamma + beta * c2
        M21 = gamma + beta * c2
        M22 = gamma
        
        return np.array([[M11, M12], [M21, M22]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        
        d2 = self.l2 / 2
        
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        
        s2 = np.sin(q2)
        
        C11 = -beta * s2 * q2_dot
        C12 = -beta * s2 * (q1_dot + q2_dot)
        C21 = beta * s2 * q1_dot
        C22 = 0.0
        
        return np.array([[C11, C12], [C21, C22]])