from torch import tensor
from torch.autograd import Function
from torch.nn import Module

class KGradF(Function):
    @staticmethod
    def forward(ctx, input_, lambda_):
        ctx.save_for_backward(input_, lambda_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, lambda_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * lambda_
        return grad_input, None

class KGrad(Module):
    def __init__(self, lambda_=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lambda = tensor(lambda_, requires_grad=False)

    def forward(self, input):
        return KGradF.apply(input, self._lambda)
