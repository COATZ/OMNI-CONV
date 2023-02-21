import math
import sys

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple
from torchvision.extension import _assert_has_ops


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """

    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

    # offset = offset.type(torch.float16)
    # print(offset)

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)

class DeformConv2d_sphe(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode="zeros"
    ):
        super(DeformConv2d_sphe, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def return_offset_sphe(self, x, isactiv=False, pad0=False, offset_file=''):
        # https://cs231n.github.io/convolutional-networks/
        h2 = int((x.shape[-2] + 2*self.padding[0] - self.kernel_size[0] - (self.kernel_size[0]-1)*(self.dilation[0]-1))/self.stride[0] + 1)
        w2 = int((x.shape[-1] + 2*self.padding[1] - self.kernel_size[1] - (self.kernel_size[1]-1)*(self.dilation[1]-1))/self.stride[1] + 1)
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        if isactiv:
            if offset_file == '':
                if pad0:
                    print("Padding 0 ACTIVE")
                    offset_file = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
                        self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'_pad0.pt'
                else:
                    offset_file = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
                        self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'.pt'
                offset = torch.load(offset_file).cuda()
                print("Loading offset file: ", offset_file)
        else:
            offset = torch.zeros(1, 2*self.kernel_size[0]*self.kernel_size[1], h2, w2).cuda()
        print("OFFSET Shape ", offset.shape)

        return offset

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
                out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """

        new_input = input
        # print(input.shape, self.padding)
        # new_input = torch.nn.functional.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='circular')
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # mem_0 = torch.cuda.memory_allocated(device)
        # print("Before Table: {} Diff: {}".format(mem_0, mem_0))
        if self.init == 0:
            print("python3 create_offset_tensor.py --w {} --h {} --k {} --s {} --p {} --d {}".format(
                new_input.shape[-1], new_input.shape[-2], self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]))
            self.offset_sphe = self.return_offset_sphe(new_input, isactiv=True, pad0=False, offset_file='').cuda()
            self.offset_sphe.require_gradient = False
            self.offset_sphe = self.offset_sphe.type(torch.float16)
            self.init = 1

            # for ux in range(self.offset_sphe.shape[-1]):
            #     for uy in range(self.offset_sphe.shape[-2]):
            #         for uk in range(self.offset_sphe.shape[-3]):
            #             if (ux + self.offset_sphe[0, uk, uy, ux] < 0) or (ux + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-1]):
            #                 # print("Alerte ux")
            #                 # print(ux, uy, uk)
            #                 # print(ux + self.offset_sphe[0, uk, uy, ux])
            #                 self.offset_sphe[0, uk, uy, ux] = 0
            #             if (uy + self.offset_sphe[0, uk, uy, ux] < 0) or (uy + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-2]):
            #                 # print("Alerte uy")
            #                 # print((uy + self.offset_sphe[0, uk, uy, ux] < 0))
            #                 # print((uy + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-2]))
            #                 # print(ux, uy, uk)
            #                 # print(ux + self.offset_sphe[0, uk, uy, ux])
            #                 self.offset_sphe[0, uk, uy, ux] = 0
            # h2 = int((new_input.shape[-2] + 2*self.padding[0] - self.kernel_size[0] - (self.kernel_size[0]-1)*(self.dilation[0]-1))/self.stride[0] + 1)
            # w2 = int((new_input.shape[-1] + 2*self.padding[1] - self.kernel_size[1] - (self.kernel_size[1]-1)*(self.dilation[1]-1))/self.stride[1] + 1)
            # offset_filename = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
            #     self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'_pad0.pt'
            # print("Saving offset file: ", offset_filename)
            # torch.save(self.offset_sphe.cpu(), offset_filename)

        offset_sphe_cat = torch.cat([self.offset_sphe for _ in range(new_input.shape[0])], dim=0).cuda()
        # mem_1 = torch.cuda.memory_allocated(device)
        # print("After Table: {} Diff: {}".format(mem_1, mem_1-mem_0))

        return deform_conv2d(new_input, offset_sphe_cat, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)
        # return deform_conv2d(new_input, offset_sphe_cat, self.weight, self.bias, stride=self.stride,
        #                       padding=((0, 0)), dilation=self.dilation, mask=mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


class DeformConv2d_sphe2_SameExport(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode="zeros"
    ):
        super(DeformConv2d_sphe2_SameExport, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.pad = None
        self.pad_input_size = (0, 0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def return_offset_sphe(self, x, isactiv=False, pad0=False, offset_file=''):
        # https://cs231n.github.io/convolutional-networks/
        h2 = int((x.shape[-2] + 2*self.padding[0] - self.kernel_size[0] - (self.kernel_size[0]-1)*(self.dilation[0]-1))/self.stride[0] + 1)
        w2 = int((x.shape[-1] + 2*self.padding[1] - self.kernel_size[1] - (self.kernel_size[1]-1)*(self.dilation[1]-1))/self.stride[1] + 1)
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        if isactiv:
            if offset_file == '':
                if pad0:
                    print("Padding 0 ACTIVE")
                    offset_file = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
                        self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'_pad0.pt'
                else:
                    offset_file = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
                        self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'.pt'
                offset = torch.load(offset_file).cuda()
                print("Loading offset file: ", offset_file)
        else:
            offset = torch.zeros(1, 2*self.kernel_size[0]*self.kernel_size[1], h2, w2).cuda()
        print("OFFSET Shape ", offset.shape)

        return offset

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width,
                out_height, out_width]): masks to be applied for each position in the
                convolution kernel.
        """

        input_size = input.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            input = self.pad(input)
        new_input = input
        # new_input = torch.nn.functional.pad(new_input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='circular')

        if self.init == 0:
            print("python3 create_offset_tensor.py --w {} --h {} --k {} --s {} --p {} --d {}".format(
                new_input.shape[-1], new_input.shape[-2], self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0]))
            self.offset_sphe = self.return_offset_sphe(new_input, isactiv=True, pad0=False, offset_file='').cuda()
            self.offset_sphe.require_gradient = False
            self.offset_sphe = self.offset_sphe.type(torch.float16)
            self.init = 1

            # for ux in range(self.offset_sphe.shape[-1]):
            #     for uy in range(self.offset_sphe.shape[-2]):
            #         for uk in range(self.offset_sphe.shape[-3]):
            #             if (ux + self.offset_sphe[0, uk, uy, ux] < 0) or (ux + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-1]):
            #                 # print("Alerte ux")
            #                 # print(ux, uy, uk)
            #                 # print(ux + self.offset_sphe[0, uk, uy, ux])
            #                 self.offset_sphe[0, uk, uy, ux] = 0
            #             if (uy + self.offset_sphe[0, uk, uy, ux] < 0) or (uy + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-2]):
            #                 # print("Alerte uy")
            #                 # print((uy + self.offset_sphe[0, uk, uy, ux] < 0))
            #                 # print((uy + self.offset_sphe[0, uk, uy, ux] > self.offset_sphe.shape[-2]))
            #                 # print(ux, uy, uk)
            #                 # print(ux + self.offset_sphe[0, uk, uy, ux])
            #                 self.offset_sphe[0, uk, uy, ux] = 0
            # h2 = int((new_input.shape[-2] + 2*self.padding[0] - self.kernel_size[0] - (self.kernel_size[0]-1)*(self.dilation[0]-1))/self.stride[0] + 1)
            # w2 = int((new_input.shape[-1] + 2*self.padding[1] - self.kernel_size[1] - (self.kernel_size[1]-1)*(self.dilation[1]-1))/self.stride[1] + 1)
            # offset_filename = './OFFSETS/offset_'+str(w2)+'_'+str(h2)+'_'+str(self.kernel_size[0])+'_'+str(
            #     self.kernel_size[1])+'_'+str(self.stride[0])+'_'+str(self.stride[1])+'_'+str(self.dilation[0])+'_pad0.pt'
            # print("Saving offset file: ", offset_filename)
            # torch.save(self.offset_sphe.cpu(), offset_filename)

        offset_sphe_cat = torch.cat([self.offset_sphe for _ in range(new_input.shape[0])], dim=0).cuda()

        return deform_conv2d(new_input, offset_sphe_cat, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)
        # return deform_conv2d(new_input, offset_sphe_cat, self.weight, self.bias, stride=self.stride,
        #                       padding=((0, 0)), dilation=self.dilation, mask=mask)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)
