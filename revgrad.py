from torch import tensor
from torch.autograd import Function
from torch.nn import Module

class RevGradF(Function):
    @staticmethod
    def forward(ctx, input_, lambda_):
        ctx.save_for_backward(lambda_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        lambda_, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * lambda_
        return grad_input, None

class RevGrad(Module):
    def __init__(self, lambda_=1.0):
        super(RevGrad, self).__init__()
        self._lambda = tensor(lambda_, requires_grad=False)

    def forward(self, input):
        return RevGradF.apply(input, self._lambda)
