import torch
from torch import nn

class GaussianKernel(nn.Module):
    def __init__(self, alpha = 1.0):
        super(GaussianKernel, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor):
        l2_distance_square = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2)
        self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * self.sigma_square))

class MKMMD(nn.Module):
    def __init__(self, kernels, linear: bool = False):
        super(MKMMD, self).__init__()
        self.kernels = kernels
        self.index_mat = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_mat = MKMMD.update_index_mat(batch_size, self.index_mat, self.linear).to(z_s.device)

        kernel_mat = sum([kernel(features) for kernel in self.kernels]) 
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_mat * self.index_mat).sum() + 2.0 / float(batch_size - 1)

        return loss

    @staticmethod
    def update_index_mat(batch_size: int, index_mat: torch.Tensor, linear: bool) -> torch.Tensor:
        if index_mat is None or index_mat.size(0) != batch_size * 2:
            index_mat = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_mat[s1, s2] = 1.0 / float(batch_size)
                    index_mat[t1, t2] = 1.0 / float(batch_size)
                    index_mat[s1, t2] = -1.0 / float(batch_size)
                    index_mat[s2, t1] = -1.0 / float(batch_size)
            else:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            index_mat[i][j] = 1.0 / float(batch_size * (batch_size - 1))
                            index_mat[i + batch_size][j + batch_size] = 1.0 / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for j in range(batch_size):
                        index_mat[i][j + batch_size] = -1.0 / float(batch_size * batch_size)
                        index_mat[i + batch_size][j] = -1.0 / float(batch_size * batch_size)
        return index_mat