from configs import device
import torch
import wandb

def train_model(model, epochs, optimizer, train_dataloader, test_dataloader, loss_f, batch_size):
    print('start to train')
    for epoch in range(epochs):
        loss_arr = []
        for x_cpu, y_cpu in train_dataloader:
            optimizer.zero_grad()
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            y_p = model(x)
            train_loss = loss_f(y_p, y).sum()
            train_loss.backward()
            loss_arr.append(train_loss.detach().cpu())
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mean_loss = torch.tensor(loss_arr).mean()
            correct = 0
            total = 0
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                y_p = model(x)
                y_p = y_p.cpu()
                y_p = torch.argmax(y_p, dim = 1)
                label = y.cpu()
                correct += (y_p == label).sum()
                total += batch_size
            test_acc = correct / total
            print(test_acc, train_mean_loss)
            wandb.log({'train_loss':train_mean_loss,'test_acc:':test_acc})    
        model.train()
    torch.save(model, './transformer.pt')
