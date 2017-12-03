import torch
from torch.nn import Module


def layer_norm(input, weight=None, bias=None, eps=1e-5):
    """ 
    taken from https://github.com/pytorch/pytorch/pull/2019/files#diff-792bf0736d92f4c00031da98098d40ad
    """
    if input is not None and input.dim() != 2:
        raise ValueError("Expected 2D tensor as input, got {}D tensor instead.".format(input.dim()))

    mean = input.mean(1, keepdim=True)
    # Prevent NaN gradients when sample std is 0 by using alternative standard deviation calculation
    std = ((input - mean).pow(2).sum(1, keepdim=True).div(input.size(1) - 1) + eps).sqrt()
    output = (input - mean) / std

    # Resize weights and biases to match dims
    if weight is not None:
        if input.size(1) != weight.nelement():
            raise RuntimeError('Expected {} features as input, got {} features instead.'
                               .format(weight.nelement(), input.size(1)))
        output = weight * output
    if bias is not None:
        if input.size(1) != bias.nelement():
            raise RuntimeError('Expected {} features as input, got {} features instead.'
                               .format(bias.nelement(), input.size(1)))
        output = output + bias
    return output


class LayerNorm(Module):
    """ 
    taken from https://github.com/pytorch/pytorch/pull/2019/files#diff-792bf0736d92f4c00031da98098d40ad
    """
    def __init__(self, num_features=None, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = num_features is not None
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return layer_norm(input, weight=self.weight, bias=self.bias,
                            eps=self.eps)

    def __repr__(self):
        if self.affine:
            return ('{name}({num_features}, eps={eps})'
                    .format(name=self.__class__.__name__, **self.__dict__))
        else:
            return ('{name}(eps={eps})'
                    .format(name=self.__class__.__name__, **self.__dict__))
