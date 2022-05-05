from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_mnist_data(data_dir='D:\\Temp\\torch_dataset', train_batch_size=64, test_batch_size=1000, workers=4):
    train_loader = DataLoader(
        datasets.MNIST(data_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader
