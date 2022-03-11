from torch.autograd import Function
from torch.nn import Module

class RevGradF(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input, None

class RevGrad(Module):
    def __init__(self):
        super(RevGrad, self).__init__()

    def forward(self, input):
        return RevGradF.apply(input)
