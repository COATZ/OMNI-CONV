import math
import torch
from torch.nn.modules.utils import _pair
import getopt
import sys


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = torch.as_tensor(axis, device='cuda', dtype=torch.float32)
    axis = axis / math.sqrt(torch.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    ROT = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], device='cpu', dtype=torch.float32)
    return ROT


def equi_coord(pano_W, pano_H, k_W, k_H, u, v):
    fov_w = k_W * math.radians(360./float(pano_W))
    focal = (float(k_W)/2) / math.tan(fov_w/2)
    # print(focal)
    c_x = 0
    c_y = 0

    u_r, v_r = u, v
    u_r, v_r = u_r-float(pano_W)/2., v_r-float(pano_H)/2.
    phi, theta = u_r/(pano_W) * (math.pi) * 2, -v_r/(pano_H) * (math.pi)

    ROT = rotation_matrix((0, 1, 0), phi)
    ROT = torch.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

    h_range = torch.tensor(range(k_H), device='cpu', dtype=torch.float32)
    w_range = torch.tensor(range(k_W,), device='cpu', dtype=torch.float32)
    w_ones = (torch.ones(k_W, device='cpu', dtype=torch.float32))
    h_ones = (torch.ones(k_H, device='cpu', dtype=torch.float32))
    h_grid = torch.matmul(torch.unsqueeze(h_range, -1), torch.unsqueeze(w_ones, 0))+0.5-float(k_H)/2
    w_grid = torch.matmul(torch.unsqueeze(h_ones, -1), torch.unsqueeze(w_range, 0))+0.5-float(k_W)/2

    K = torch.tensor([[focal, 0, c_x], [0, focal, c_y], [0., 0., 1.]], device='cpu', dtype=torch.float32)
    inv_K = torch.inverse(K)
    rays = torch.stack([w_grid, h_grid, torch.ones(h_grid.shape, device='cpu', dtype=torch.float32)], 0)
    rays = torch.matmul(inv_K, rays.reshape(3, k_H*k_W))
    rays /= torch.norm(rays, dim=0, keepdim=True)
    rays = torch.matmul(ROT, rays)
    rays = rays.reshape(3, k_H, k_W)

    phi = torch.atan2(rays[0, ...], rays[2, ...])
    theta = torch.asin(torch.clamp(rays[1, ...], -1, 1))
    x = (pano_W)/(2.*math.pi)*phi + float(pano_W)/2.
    y = (pano_H)/(math.pi)*theta + float(pano_H)/2.

    roi_y = h_grid+v_r + float(pano_H)/2.
    roi_x = w_grid+u_r + float(pano_W)/2.

    new_roi_y = (y)
    new_roi_x = (x)

    offsets_x = (new_roi_x - roi_x)
    offsets_y = (new_roi_y - roi_y)

    return offsets_x, offsets_y


def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=1, s_height=1, bs=16):
    # n=1
    offset = torch.zeros(2*k_H*k_W, pano_H, pano_W, device='cpu', dtype=torch.float32)

    for v in range(0, pano_H, s_height):
        for u in range(0, pano_W, s_width):
            offsets_x, offsets_y = equi_coord(pano_W, pano_H, k_W, k_H, u, v)
            offsets = torch.cat((torch.unsqueeze(offsets_y, -1), torch.unsqueeze(offsets_x, -1)), dim=-1)
            total_offsets = offsets.flatten()
            offset[:, v, u] = total_offsets

    offset = torch.unsqueeze(offset, 0)
    offset = torch.cat([offset for _ in range(bs)], dim=0)
    offset.requires_grad_(False)
    print(offset.shape)
    # print(offset)
    return offset


def distortion_aware_col(pano_W, pano_H, k_W, k_H, s_width=1, s_height=1, bs=1):
    # n=1
    print(pano_W, pano_H, k_W, k_H, s_width, s_height, bs)
    offset = torch.zeros(2*k_H*k_W, pano_H, 1, device='cpu', dtype=torch.float32)

    for v in range(0, pano_H, s_height):
        for u in range(0, s_width, s_width):
            offsets_x, offsets_y = equi_coord(pano_W, pano_H, k_W, k_H, u, v)
            offsets = torch.cat((torch.unsqueeze(offsets_y, -1), torch.unsqueeze(offsets_x, -1)), dim=-1)
            total_offsets = offsets.flatten()
            offset[:, v, u] = total_offsets

    offset = torch.unsqueeze(offset, 0)
    offset = torch.cat([offset for _ in range(bs)], dim=0)
    offset.requires_grad_(False)
    print(offset.shape)
    # print(offset)
    return offset


arguments_BS = 1
arguments_WIDTH = 1
arguments_HEIGHT = 10
arguments_KERNEL = 3
arguments_KERNEL_X = 3
arguments_KERNEL_Y = 3
arguments_PADDING = 0
arguments_PADDING_X = 1
arguments_PADDING_Y = 1
arguments_STRIDE = 1
arguments_DILATION = 1

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--w' and strArgument != '':
        arguments_WIDTH = strArgument
    if strOption == '--h' and strArgument != '':
        arguments_HEIGHT = strArgument
    if strOption == '--k' and strArgument != '':
        arguments_KERNEL = strArgument
    if strOption == '--kx' and strArgument != '':
        arguments_KERNEL_X = strArgument
    if strOption == '--ky' and strArgument != '':
        arguments_KERNEL_Y = strArgument
    if strOption == '--p' and strArgument != '':
        arguments_PADDING = strArgument
    if strOption == '--px' and strArgument != '':
        arguments_PADDING_X = strArgument
    if strOption == '--py' and strArgument != '':
        arguments_PADDING_Y = strArgument
    if strOption == '--s' and strArgument != '':
        arguments_STRIDE = strArgument
    if strOption == '--d' and strArgument != '':
        arguments_DILATION = strArgument
    if strOption == '--bs' and strArgument != '':
        arguments_BS = strArgument

if __name__ == "__main__":

    torch.manual_seed(0)
    input = torch.zeros(int(arguments_BS), 1, int(arguments_HEIGHT), int(arguments_WIDTH))
    weight = torch.zeros(1, 1, int(arguments_KERNEL), int(arguments_KERNEL))
    # weight = torch.zeros(1,1,int(arguments_KERNEL_Y),int(arguments_KERNEL_X)) # Si k_x != k_y
    stride = int(arguments_STRIDE)
    padding = int(arguments_PADDING)
    dilation = int(arguments_DILATION)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    # pad_h, pad_w = [int(arguments_PADDING_Y),int(arguments_PADDING_X)] # Si p_x != p_y
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    bs, n_in_channels, in_h, in_w = input.shape

    pano_W = int((in_w + 2*pad_w - dil_w*(weights_w-1)-1)//stride_w + 1)
    pano_H = int((in_h + 2*pad_h - dil_h*(weights_h-1)-1)//stride_h + 1)

    print(pano_W, pano_H, weights_w, weights_h, stride_w, stride_h, bs)
    k_W = weights_w
    k_H = weights_h
    offset = distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=stride_w, s_height=stride_h, bs=bs)
    torch.save(offset, './OFFSETS/offset_'+str(pano_W)+'_'+str(pano_H)+'_'+str(k_W)+'_'+str(k_H)+'_'+str(stride_w)+'_'+str(stride_h)+'_'+str(bs)+'.pt')

    # offset_col = distortion_aware_col(pano_W, pano_H, weights_w, weights_h, s_width = stride_w, s_height = stride_h, bs = bs) # Pour obtenir uniquement une colonne
    # torch.save(offset_col,'./OFFSETS/offset_col_'+str(pano_H)+'_'+str(weights_w)+'_'+str(weights_h)+'_'+str(stride_w)+'_'+str(stride_h)+'_'+str(bs)+'.pt')

