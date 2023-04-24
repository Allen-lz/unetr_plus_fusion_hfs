from torch.nn import functional as F
import torch

import torch.nn as nn


class hfs():
    def __init__(self, n=3, s=2, use_square_kernel=True):
        self.n = n
        self.s = s
        self.use_square_kernel = use_square_kernel

    def run(self, x):
        """
            :param x: input_tensor
            :param n: maximum number of kernel
            :param s: the scale of the largest kernel is 1 / s of the size of the input_tensor
            :param type: square kernel or proportional kernel
            :return: x * mask
        """
        b, c, h, w = x.shape
        b, c, h, w = int(b), int(c), int(h), int(w)
        # -------------------------------------------------------------------------
        avg = x.mean(1)
        thr = avg.view(b, -1).sum(-1) / (h * w)
        mask = (avg > thr.unsqueeze(-1).unsqueeze(-1)).float()
        # -------------------------------------------------------------------------
        xs = list()
        if self.use_square_kernel:
            k_stride = max(1, min(h, w) // self.n - 1)
            k_size = min(h, w) // self.s
            for i in range(self.n):
                if k_size <= 1:
                    break
                xs.append(F.avg_pool2d(x, kernel_size=k_size, stride=2))
                k_size = k_size - k_stride
        else:
            k_stride = (max(1, h // self.n - 1), max(1, w // self.n))

            k_size = [h // self.s, w // self.s]
            for i in range(self.n):
                if k_size[0] <= 1 or k_size[1] <= 1:
                    break
                # print(k_size)
                xs.append(F.avg_pool2d(x, kernel_size=tuple(k_size), stride=2))
                k_size = [k_size[0] - k_stride[0], k_size[1] - k_stride[1]]

        for i, xi in enumerate(xs):
            avgi = xi.mean(1)
            _, hi, wi = avgi.size()
            thri = avgi.view(b, -1).sum(-1) / (hi * wi)
            maski = (avg > thri.unsqueeze(-1).unsqueeze(-1)).float()
            xs[i] = maski

        for maski in xs:
            mask += maski
        mask = mask + 1
        mask = torch.log10(mask) + 1
        # mask = F.sigmoid(mask)
        # mask = F.normalize(mask.view(b, -1), p=1, dim=1)
        # mask = mask.view(b, h, w)
        mask = mask.unsqueeze(1)
        return mask


class hfs1d(nn.Module):
    def __init__(self):
        super(hfs1d, self).__init__()
        self.st = nn.Softmax(dim=-1)
        self.pool1 = nn.AvgPool1d(kernel_size=7)
        self.pool2 = nn.AvgPool1d(kernel_size=5)
        self.pool3 = nn.AvgPool1d(kernel_size=3)

    def forward(self, x):
        x1 = self.pool1(x.unsqueeze(1))
        x2 = self.pool2(x.unsqueeze(1))
        x3 = self.pool3(x.unsqueeze(1))
        m1 = torch.mean(x1.squeeze(1), dim=1, keepdim=True)
        m2 = torch.mean(x2.squeeze(1), dim=1, keepdim=True)
        m3 = torch.mean(x3.squeeze(1), dim=1, keepdim=True)

        mask1 = (x > m1) + 0.0
        mask2 = (x > m2) + 0.0
        mask3 = (x > m3) + 0.0

        mask = mask1 + mask2 + mask3 + 1.0

        # hs_attention = torch.log10(mask) + 1.0
        device = mask.device
        a = torch.Tensor([10]).to(device)
        hs_attention = torch.log(mask) / torch.log(a) + 1.0

        x = hs_attention * x
        return x