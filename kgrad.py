from torch import tensor
from torch.autograd import Function
from torch.nn import Module

class KGradF(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.lambda_
        return grad_input, None