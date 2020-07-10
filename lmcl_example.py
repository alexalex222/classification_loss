import os
import time
import datetime
import torch
from torch import nn
import numpy as np

from utils import AverageMeter, plot_features, get_command_line_parser
from prepare_data import get_mnist_data
from models import ConvNet
from loss.large_margin_cos_loss import LargeMarginCosLoss


class BaseLine(nn.Module):
    def __init__(self, feature_extractor, num_base_class, embed_size):
        super(BaseLine, self).__init__()
        self.feature_extractor = feature_extractor
        self.embed_size = embed_size
        self.feature_embedding = nn.Linear(feature_extractor.output_dim, self.embed_size)

    def forward(self, x):
        feature = self.feature_embedding(self.feature_extractor(x))
        return feature


def train(model,
          classifier,
          criterion,
          optimizer,
          trainloader,
          use_gpu,
          num_classes,
          epoch,
          args):
    model.train()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features = model(data)
        outputs, _ = classifier(features, labels)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), labels.size(0))

        # features = torch.norm(features, p=2, dim=1)

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})"
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    if args.plot:
        weights = None
        centers = classifier.weight.data.cpu().numpy()
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, weights, centers, all_labels, num_classes, epoch, prefix='train', args=args)


def evaluate(model, classifier, criterion, testloader, use_gpu, num_classes, epoch, args):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features = model(data)
            _, cosine = classifier(features, labels)
            predictions = cosine.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            # features = torch.norm(features, p=2, dim=1)

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
        centers = classifier.weight.data.cpu().numpy()
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
    feature_extractor = ConvNet(depth=4, input_channel=1)
    model = BaseLine(feature_extractor=feature_extractor, num_base_class=10, embed_size=2)
    classifier = LargeMarginCosLoss(feature_size=2, class_num=10)

    if use_gpu:
        model = model.cuda()
        classifier = classifier.cuda()

    criterion = nn.CrossEntropyLoss()
    # optimizer_model = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}], lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                 lr=args.lr_model, weight_decay=5e-04)

    if args.stepsize > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, classifier, criterion,
              optimizer,
              trainloader, use_gpu, 10, epoch, args)

        if args.stepsize > 0:
            scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            acc, err = evaluate(model, classifier, criterion, testloader, use_gpu, 10, epoch, args=args)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()


