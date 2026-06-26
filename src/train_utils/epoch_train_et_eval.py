from tqdm import tqdm
import torch
import os
import numpy as np 
import matplotlib.pyplot as plt

def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss, total = 0.0, 0,

    pbar = tqdm(train_loader, unit="batch", desc="Training")
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / total


def train_steps(model, optimizer, criterion, train_loader, data_iter, device,interval_steps,step,epoch,scheduler=None):
    model.train()
    total_loss, total = 0.0, 0
    with tqdm(total=interval_steps, desc=f"Training Steps {step} to {step + interval_steps - 1}") as pbar:
        for _ in range(interval_steps):
            try:
                data, target = next(data_iter)
            except StopIteration:
                epoch += 1
                # print(f"Starting epoch {epoch}")
                data_iter = iter(train_loader)
                data, target = next(data_iter)

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            step += 1
            pbar.update(1)
    return total_loss / total, data_iter, step, epoch


def eval_epoch(model, criterion, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_loss, total = 0.0, 0,

        pbar = tqdm(test_loader, unit="batch", desc="Eval", leave=True)
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        return total_loss / total


def training_loss_plotting(x,x_name,y_test,y_train,out_path):
    if x is None:
        x = np.arange(len(y_test))
    plt.plot(x,y_test,label = 'Test Loss')
    plt.plot(x,y_train,label = 'Train Loss')
    plt.xlabel(x_name)

    plt.ylabel('Loss')
    plt.title(os.path.basename(out_path))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss.png'))
    plt.close()


def training_puzzle_plotting(x,x_name,puzzle_acc,out_path):
    if x is None:
        x = np.arange(len(puzzle_acc))
    plt.plot(x,puzzle_acc,label = 'puzzle_acc')
    plt.xlabel(x_name)

    plt.ylabel('Puzzle_acc')
    plt.title(os.path.basename(out_path))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(out_path, 'training_puzzle_acc.png'))
    plt.close()