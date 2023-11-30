import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

def train(cfg, model, trainloader):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg.lr, 
        momentum=0.9, 
        nesterov=True,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1000, 
        gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.epochs):
        model.train()
        nb_batches = len(trainloader)
        losses = 0.0
        train_acc = 0.0

        for z, y in trainloader:
            z = z.float().to(device)
            y = y.long().to(device)

            out = model(z)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            losses += loss.item()
            optimizer.step()
            train_acc += (F.softmax(out, dim=1).argmax(1) == y).cpu().numpy().mean()
        scheduler.step()
       
        info = {
            "epoch": epoch+1,
            "train_loss": np.round(losses/nb_batches, 4),
            "train_acc": np.round(train_acc/nb_batches, 4)
        }
        print(info)

        if cfg.deploy:
            wandb.log(info)

    return model

class TestDataset(Dataset):
    def __init__(self, testdata, dataset, contextlength=200):
        self.testdata = torch.from_numpy(testdata)
        self.traindata = dataset.data
        self.contextlength = contextlength

    def __len__(self):
        return len(self.testdata)

    def __getitem__(self, idx):
        z = torch.cat((self.traindata[-self.contextlength:], self.testdata[idx:idx+1]))
        y = z[-1, -1].clone()
        z[-1, -1] = 0.5
        return z, y

def evaluate(cfg, model, dataset, sample):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    preds = []
    truths = []
    model.eval()

    for rep in range(cfg.num_reps):
        np.random.seed(10*cfg.seed + 100*rep)
        testdata = np.array([sample(s) for s in range(cfg.t, cfg.T)])
        testdataset = TestDataset(testdata, dataset)
        testloader = DataLoader(testdataset, batch_size=100, shuffle=False)

        pred_rep = []
        truth_rep = []
        for z, y in testloader:
            z = z.float().to(device)
            y = y.long().to(device)
            out = model(z)
            pred_rep.extend(out.detach().cpu().argmax(1).numpy())
            truth_rep.extend(y.detach().cpu().numpy())
        preds.append(pred_rep)
        truths.append(truth_rep)
    preds = np.array(preds)
    truths = np.array(truths)
    return preds, truths
