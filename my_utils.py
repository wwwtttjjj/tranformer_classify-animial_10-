import torchvision
import torch

def get_my_data(train_root, test_root, transform, batch_size):
    data_iter = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(data_iter, batch_size = batch_size, shuffle = True)

    test_iter = torchvision.datasets.ImageFolder(root=test_root, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_iter, batch_size = batch_size, shuffle = False)

    return train_dataloader, test_dataloader
