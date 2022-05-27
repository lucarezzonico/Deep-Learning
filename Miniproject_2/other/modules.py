import torch
from torch.nn.functional import fold, unfold

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# implemented modules and losses will inherit from Module
class Module(object):
    # forward should get for input and returns, a tensor or a tuple of tensors
    def forward(self, *input):
        raise NotImplementedError

    # backward should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect to
    # the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
    # containing the gradient of the loss wrt the module’s input.
    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    # param should return a list of pairs composed of a parameter tensor and a gradient tensor of the same size.
    # This list should be empty for parameterless modules (such as ReLU).
    def param(self):
        return []

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

# Convolution layer
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Xavier initialization
        a = 3**0.5 * (2/((self.in_channels + self.out_channels) * self.kernel_size[0] * self.kernel_size[1]))**0.5
        self.weight = torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-a, a).to(device)
        self.d_weight = torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]).zero_().to(device)
        self.bias = torch.empty(out_channels).uniform_(-a, a).to(device)
        self.d_bias = torch.empty(out_channels).zero_().to(device)

    def forward(self, input):
        self.input = input.to(device)
        self.input_unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        wxb = self.weight.view(self.out_channels, -1) @ self.input_unfolded + self.bias.view(1, -1, 1)
        wxb = wxb.view(self.input.shape[0], self.out_channels,
                       int((self.input.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1) - 1)/self.stride[0] + 1),
                       int((self.input.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1) - 1)/self.stride[1] + 1))
        return wxb

    def backward(self, gradwrtoutput):
        gradwrtoutput_reshaped = gradwrtoutput.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
        input_unfolded_reshaped = self.input_unfolded.permute(2, 0, 1).reshape(gradwrtoutput_reshaped.shape[1], -1)
        self.d_weight.data = (gradwrtoutput_reshaped @ input_unfolded_reshaped).reshape(self.weight.shape)
        self.d_bias.data = gradwrtoutput.sum(axis=(0,2,3))
        gradwrtinput_unfolded = self.weight.reshape(self.out_channels, -1).t() @ gradwrtoutput_reshaped
        gradwrtinput_unfolded = gradwrtinput_unfolded.reshape(self.input_unfolded.permute(1, 2, 0).shape).permute(2, 0, 1)
        gradwrtinput = fold(gradwrtinput_unfolded, (self.input.shape[2], self.input.shape[3]), kernel_size=self.kernel_size,
                                                   stride=self.stride, padding=self.padding, dilation=self.dilation)
        # print('gradwrtinput : ', gradwrtinput.shape)
        return gradwrtinput

    def param(self):
        return [[self.weight, self.d_weight],
                [self.bias, self.d_bias]]

# Upsampling layer, which is usually implemented with transposed convolution, but you can alternatively use a combination of Nearest neighbor upsampling + Convolution for this mini-project
class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor if isinstance(scale_factor, tuple) else (scale_factor, scale_factor)

    def forward(self, input):
        self.input = input
        # print(input.shape)
        upsampled_input = self.input.repeat_interleave(self.scale_factor[1], dim=3)
        upsampled_input = upsampled_input.repeat_interleave(self.scale_factor[0], dim=2)
        # print(upsampled_input.shape)
        return upsampled_input

    def backward(self, gradwrtoutput):
        gradwrtinput = gradwrtoutput.unfold(dimension=3, size=self.scale_factor[1], step=self.scale_factor[1])
        gradwrtinput = gradwrtinput.sum(dim=4)
        gradwrtinput = gradwrtinput.unfold(dimension=2, size=self.scale_factor[0], step=self.scale_factor[0])
        gradwrtinput = gradwrtinput.sum(dim=4)
        gradwrtinput = gradwrtinput.div(self.scale_factor[0] * self.scale_factor[1])
        return gradwrtinput

    def param(self):
        return []

# ReLU
class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input  # save previous input for backward step
        output = self.input  # prepare output
        output[output < 0] = 0  # relu function
        return output

    def backward(self, gradwrtoutput):
        # print(gradwrtoutput[0, 0, :, :])
        return gradwrtoutput * (self.input > 0)

    def param(self):
        return []

# Sigmoid
class Sigmoid(Module):
    def __init__(self):
        pass

    def forward(self, input):
        self.sigma = input.sigmoid()  # save previous sigma for backward step
        return self.sigma

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.sigma * (1 - self.sigma)

    def param(self):
        return []

# container to put together an arbitrary configuration of modules together
class Sequential(Module):
    def __init__(self, *args):
        self.blocks = list(args)

    def forward(self, input):
        for b in self.blocks:
            # the input of the next layer is the output of the previous layer
            input = b.forward(input)
        # print('input', input.size())
        return input

    def backward(self, gradwrtoutput):
        for b in reversed(self.blocks):
            # output of the previous layer is the input of the next layer
            gradwrtoutput = b.backward(gradwrtoutput)
        # print('gradwrtoutput', gradwrtoutput.size())
        return gradwrtoutput

    def param(self):
        # Regroup the parameters of each block of Sequential inside block_parameters_list
        block_parameters_list = []
        for b in self.blocks:
            for p in b.param():
                block_parameters_list.append(p)
        # print(len(block_parameters_list))
        return block_parameters_list

# Mean Squared Error Loss Function
class MSE(Module):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.input = input
        self.target = target
        self.loss = ((input-target)**2).mean()
        return self.loss

    def backward(self):
        self.output = 2 * (self.input-self.target) / (self.input.shape[0] * self.input.shape[1] * self.input.shape[2] * self.input.shape[3])
        return self.output

    def param(self):
        return []

# Stochastic Gradient Descent (SGD) optimizer
class SGD(Module):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for [param, d_param] in self.params:
            param.data.add_(-self.lr * d_param)  # put d_param to 0

    def zero_grad(self):
        for p in self.param():
            p[1].zero_()  # put d_param to 0

    def param(self):
        return self.params


