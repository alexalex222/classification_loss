import os
import time
import datetime
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import AverageMeter, plot_features, get_command_line_parser
from prepare_data import get_mnist_data
from models import ConvNet
from loss.center_loss import CenterLoss


class BaseLine(nn.Module):
    def __init__(self, feature_extractor, num_base_class, embed_size):
        super(BaseLine, self).__init__()
        self.feature_extractor = feature_extractor
        self.embed_size = embed_size
        self.num_base_class = num_base_class
        self.feature_embedding = nn.Linear(feature_extractor.output_dim, self.embed_size)
        # self.scale_factor = nn.Parameter(torch.tensor([10], dtype=torch.float32))
        self.scale_factor = 1.0
        init_protos = torch.randn(num_base_class, self.embed_size)
        proto_norm = init_protos.norm(p=2, dim=1, keepdim=True)
        init_protos =  init_protos.div(proto_norm.expand_as(init_protos))
        self.protos = nn.Parameter(init_protos)
        # nn.init.kaiming_normal_(self.protos)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = F.relu(h)
        features = self.feature_embedding(h)
        dist_mat = self.distance(features)
        return dist_mat, features

    def extract(self, x):
        x = self.feature_embedding(self.feature_extractor(x))
        return x

    def distance(self, features):
        batch_size = features.size(0)
        dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_base_class) + \
            torch.pow(self.protos, 2).sum(dim=1, keepdim=True).expand(self.num_base_class, batch_size).t()
        dist_mat.addmm_(features, self.protos.t(), beta=1, alpha=-2)
        return dist_mat

    def center_loss(self, features, y):
        """
        Args:
            dist_mat: distance matrix with shape (batch_size, num_class).
            y: ground truth labels with shape (batch_size).
        """
        batch_size = features.shape[0]
        batch_centers = self.protos[y]

        dist = torch.pow(features - batch_centers, 2)
        loss = (dist.clamp(min=1e-12, max=1e+12).sum() / batch_size) / self.embed_size

        return loss

    def compute_total_loss(self, x, y, loss_weight=1.0):
        distmat, features = self.forward(x)
        loss_reg = self.center_loss(distmat, y)
        ce_loss = F.cross_entropy(-self.scale_factor * distmat, y)
        total_loss = ce_loss + loss_weight * loss_reg
        return total_loss, ce_loss, loss_reg


def train(model,
          optimizer_model, 
          trainloader, use_gpu, num_classes, epoch, args):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        distmat, features = model.forward(data)
        loss_cent = model.center_loss(features, labels)
        loss_xent = F.cross_entropy(-model.scale_factor * distmat, labels)
        if epoch < 10:
            weight_loss = 0.0
        else:
            weight_loss = 0.01
        weight_loss = 0.001
        loss = loss_xent + weight_loss * loss_cent

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                          cent_losses.val, cent_losses.avg))

    if args.plot:
        # weights = model.classifier.weight.data.cpu().numpy()
        weights = None
        centers = model.protos.data.cpu().numpy()
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, weights, centers, all_labels, num_classes, epoch, prefix='train', args=args)


def evaluate(model, testloader, use_gpu, num_classes, epoch, args):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            dist_mat, features = model.forward(data)
            outputs = -dist_mat
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        weights = None
        centers = model.protos.data.cpu().numpy()
        plot_features(all_features, weights, centers, all_labels, num_classes, epoch, prefix='test', args=args)

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

def main():
    parser = get_command_line_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    trainloader, testloader = get_mnist_data(train_batch_size=args.batch_size, workers=args.workers)

    print("Creating model: {}".format(args.model))
    feature_extractor = ConvNet(depth=6, input_channel=1)
    model = BaseLine(feature_extractor=feature_extractor, num_base_class=10, embed_size=2)

    if use_gpu:
        model = model.cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=10, feat_dim=model.embed_size)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model)

    if args.stepsize > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, 
              optimizer_model,
              trainloader, use_gpu, 10, epoch, args)

        if args.stepsize > 0:
            scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            acc, err = evaluate(model, testloader, use_gpu, 10, epoch, args=args)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()


# %%

# %%
