from unittest import TestResult
import torch
import torchvision
import wandb
import os
from configs import lr, epochs, batch_size, device, transform, loss_f
from model import Vit
import my_utils
import trainModel



if __name__ == "__main__":
    train_root = "../data/train/raw-img/"
    test_root = "../data/test/raw-img/"
    train_dataloader, test_dataloader = my_utils.get_my_data(train_root=train_root, test_root=test_root,transform=transform, batch_size=batch_size)


    wandb.init(entity='wtj', project='transformer object classifier')
    wandb.config = {
        'lr':lr,
        'epochs':epochs, 
        'batch_size':batch_size
    }

    if os.path.exists('transformer.pt'):
        modelTransFormer = torch.load('./transformer.pt', map_location=torch.device(device))
        print("model is load finish")
    else:
        modelTransFormer = Vit(
            image_size=256,
            patch_size=32,
            num_classes=10,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1)
        modelTransFormer.to(device)

    optimizer = torch.optim.SGD(modelTransFormer.parameters(),
            lr=lr, momentum=0.8, weight_decay=5e-4)
        # 学习率下降策略

    trainModel.train_model(modelTransFormer,epochs, optimizer, train_dataloader, test_dataloader, loss_f, batch_size)
    
