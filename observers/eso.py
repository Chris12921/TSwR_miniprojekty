from copy import copy
import numpy as np

class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        
        y = np.atleast_1d(q)
        u_val = np.atleast_1d(u)
        
        z_dot = self.A @ self.state + self.B @ u_val + self.L @ (y - self.W @ self.state)
        self.state = self.state + z_dot * self.Tp

    def get_state(self):
        return self.state