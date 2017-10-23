import torch
import copy


# based on https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py
class MinibatchStdDev(torch.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super(MinibatchStdDev, self).__init__()
        # lower case the passed parameter
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], \
                'Invalid averaging mode %s' % self.averaging

    def adjusted_std(self, x, **kwargs):
        # calculate the std_dev in such a way that it doesn't result in 0 otherwise 0 norm operation's gradient is nan
        return torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = torch.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)
        # concatenate the constant feature map to the input
        return torch.cat([x, vals], 1)
