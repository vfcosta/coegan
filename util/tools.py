import torch
import os
import numpy as np
from torch.nn import functional as F


def save_checkpoint(discriminator, generator, epoch, data_folder):
    out_dir = '%s/models' % data_folder
    os.makedirs(out_dir)
    torch.save(discriminator.state_dict(), '%s/D_epoch_%d' % (out_dir, epoch))
    torch.save(generator.state_dict(), '%s/G_epoch_%d' % (out_dir, epoch))


def round_array(array, max_sum, invert=False):
    if invert and len(array) > 1:
        # invert sizes as low fitness values is better than high values
        array = (max_sum - np.array(array)) / (len(array) - 1)

    array = np.array(array)
    array = np.clip(array, 0, max_sum)
    rounded = np.floor(array)
    diff = int(max_sum) - int(np.sum(rounded))
    if diff > 0:
        for i in range(diff):
            max_index = (array - rounded).argmax()
            if len(array) == 2:
                max_index = array.argmin()
            rounded[max_index] += 1
    return rounded


def permutations(generators, discriminators, random=False):
    ng, nd = len(generators), len(discriminators)

    if ng != nd or random:
        pairs = np.array(np.meshgrid(range(ng), range(nd))).T.reshape(-1, 2)
        np.random.shuffle(pairs)
        return pairs

    # FIXME: should chache pairs for same nd, ng or transform in a generator
    pairs = []
    j = 0
    for start_j in range(nd):
        for i in range(ng):
            pairs.append((i, (j + start_j) % nd))
            j = (j + 1) % nd
    return pairs


def is_cuda_available():
    return torch.cuda.is_available()


def cuda(variable):
    return variable.cuda() if is_cuda_available() else variable


# based on https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
def resize_activations(v, so):
    """
    Resize activation tensor 'v' of shape 'si' to match shape 'so'.
    :param v:
    :param so:
    :return:
    """
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so)# and si[0] == so[0]

    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]
    if si[0] > so[0]:
        v = v[:so[0], :]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = (si[2] // so[2], si[3] // so[3])
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

    # Extend spatial axes. Below is a wrong implementation
    # shape = [1, 1]
    # for i in range(2, len(si)):
    #     if si[i] < so[i]:
    #         assert so[i] % si[i] == 0
    #         shape += [so[i] // si[i]]
    #     else:
    #         shape += [1]
    # v = v.repeat(*shape)
    if si[2] < so[2]:
        assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]  # currently only support this case
        v = F.interpolate(v, scale_factor=so[2]//si[2], mode='nearest', align_corners=True)

    # Increase feature maps.
    if si[1] < so[1]:
        z = torch.zeros([v.shape[0], so[1] - si[1]] + so[2:])
        v = torch.cat([v, z], 1)
    if si[0] < so[0]:
        z = torch.zeros([so[0] - si[0], v.shape[1]] + so[2:])
        v = torch.cat([v, z], 0)
    return v
