import torch
from torch.autograd import Function

class KGradF(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_ : float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.lambda_
        return grad_input, None