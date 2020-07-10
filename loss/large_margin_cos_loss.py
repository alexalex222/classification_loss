import torch
from torch import nn


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class LargeMarginCosLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
    """
    def __init__(self, feature_size, class_num, s=7.0, m=0.20):
        super(LargeMarginCosLoss, self).__init__()
        self.feature_size = feature_size
        self.class_num = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(class_num, feature_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = cosine_sim(x, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output, cosine
