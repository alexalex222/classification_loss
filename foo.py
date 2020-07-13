import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np

from utils import AverageMeter, plot_features, get_command_line_parser
from prepare_data import get_mnist_data
from models import ConvNet
from loss.center_loss import CenterLoss

class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        basenet = torchvision.models.resnet18(pretrained=pretrained)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x


class BaseLine(nn.Module):
    def __init__(self, feature_extractor, num_base_class, embed_size):
        super(BaseLine, self).__init__()
        self.feature_extractor = feature_extractor
        self.embed_size = embed_size
        self.feature_embedding = nn.Linear(feature_extractor.output_dim, self.embed_size)
        self.classifier = nn.Linear(self.embed_size, num_base_class, bias=True)

    def forward(self, x):
        x = self.feature_embedding(self.feature_extractor(x))
        scores = self.classifier(x)
        return scores

    def extract(self, x):
        x = self.feature_embedding(self.feature_extractor(x))
        return x


def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
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
        outputs = model(data)
        features = model.extract(data)
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss = loss_xent + 0.1 * loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        # for param in criterion_cent.parameters():
        #    param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

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
        weights = model.classifier.weight.data.cpu().numpy()
        centers = criterion_cent.centers.data.cpu().numpy()
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, weights, centers, all_labels, num_classes, epoch, prefix='train', args=args)


def evaluate(model, criterion_cent, testloader, use_gpu, num_classes, epoch, args):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            features = model.extract(data)
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
        weights = model.classifier.weight.data.cpu().numpy()
        centers = criterion_cent.centers.data.cpu().numpy()
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

    image_size = 224
    # data augmentation transformation
    aug_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # standard data transformation
    standard_transform = transforms.Compose(
        [
            transforms.Resize((int(image_size * 1.143), int(image_size * 1.143))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(root='C:\\Research\\image_data\\CUB\\CUB_200_2011\\images', transform=aug_transform)

    trainloader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=4, pin_memory=True)

    print("Creating model: {}".format(args.model))
    feature_extractor = ResNet18()
    model = BaseLine(feature_extractor=feature_extractor, num_base_class=200, embed_size=256)

    if use_gpu:
        model = model.cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=200, feat_dim=model.embed_size)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, weight_decay=5e-04)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

    if args.stepsize > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              trainloader, use_gpu, 10, epoch, args)

        if args.stepsize > 0:
            scheduler.step()

        # if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
        #     print("==> Test")
        #    acc, err = evaluate(model, criterion_cent, testloader, use_gpu, 10, epoch, args=args)
        #    print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


if __name__ == '__main__':
    main()


