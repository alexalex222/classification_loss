import argparse
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch


def get_command_line_parser():
    parser = argparse.ArgumentParser("Loss Example")
    parser.add_argument('--data-dir', type=str, default='D:\\Temp\\torch_dataset', help='Path to data set')
    parser.add_argument('--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    # optimization
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
    parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--stepsize', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
    # model
    parser.add_argument('--model', type=str, default='cnn')
    # misc
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
    parser.add_argument('--plot-normalized', action='store_true', help="whether to plot normalized feature")
    return parser


def plot_features(features, weights, centers, labels, num_classes, epoch, prefix, args):
    """

    :param features:
    :param weights:
    :param centers:
    :param labels:
    :param num_classes:
    :param epoch:
    :param prefix:
    :param args:
    :return:
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
            alpha=0.1
        )

        if weights is not None:
            plt.scatter(
                weights[label_idx, 0],
                weights[label_idx, 1],
                c=colors[label_idx],
                s=20,
                marker='^'
            )
        if args.weight_cent > 0 and centers is not None:
            plt.scatter(
                centers[label_idx, 0],
                centers[label_idx, 1],
                c='black',
                s=80,
                marker='x'
            )

    #if args.weight_cent > 0:
    #    plt.xlim(-7, 7)
    #    plt.ylim(-7, 7)
    #else:
    #    plt.xlim(-90, 90)
    #    plt.ylim(-90, 90)

    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


class AverageMeter:
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def l2_norm(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized
