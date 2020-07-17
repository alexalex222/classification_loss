import torch
from torch import nn


def euclidean_metric(x, m):
    # x: batch_size x D
    # y: num_proto x D
    n = x.shape[0]
    m = w.shape[0]
    x = x.unsqueeze(1).expand(n, m, -1)
    w = w.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class dce_loss(nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):
   
        super(dce_loss, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.feat_dim,self.n_classes).cuda(),requires_grad=True)
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
   
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers

        return self.centers, -dist


def center_regularization(features, centers, labels):
        distance=(features-torch.t(centers)[labels])

        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)

        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance