import torch
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

lr = 2e-4
epochs = 100
batch_size = 32
loss_f = torch.nn.CrossEntropyLoss()
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(256),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406],std = [0.229, 0.224, 0.225])
    ])
