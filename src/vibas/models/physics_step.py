import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatStep(nn.Module):
    """ T_{t+1} = T_t + dt (alpha ∇²T_t + Q_theta([T, M])) """
    def __init__(self, alpha=1.5e-5, dt=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.dt = dt
        self.q = nn.Conv2d(2, 1, 3, padding=1)  # learnable source term

    @staticmethod
    def laplacian(T):
        k = torch.tensor(
            [[0., 1., 0.],
             [1.,-4., 1.],
             [0., 1., 0.]], dtype=T.dtype, device=T.device
        ).view(1,1,3,3)
        return F.conv2d(T, k, padding=1)

    def forward(self, T, M):
        lap = self.laplacian(T)
        Q = self.q(torch.cat([T, M], dim=1))
        return T + self.dt * (self.alpha * lap + Q)
