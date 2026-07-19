# -*- coding: utf-8 -*-
import os
import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.model_ELU import EEGNet
#from models.model_ReLU import EEGNet
#from models.model_LeakyReLU import EEGNet
#from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Dataset
# ==========================================================
class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


# ==========================================================
# Plot functions
# ==========================================================
def plot_train_acc(train_acc_list, epochs):
    plt.figure()
    plt.plot(range(1, len(train_acc_list)+1), train_acc_list)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/ELU_EP150_train_accuracy.png", dpi=300)
    plt.close()


def plot_train_loss(train_loss_list, epochs):
    plt.figure()
    plt.plot(range(1, len(train_loss_list)+1), train_loss_list, color='orange')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/ELU_EP150_train_loss.png", dpi=300)
    plt.close()


def plot_test_acc(test_acc_list, epochs):
    plt.figure()
    plt.plot(range(1, len(test_acc_list)+1), test_acc_list, color='green')
    plt.title("Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/ELU_EP150_test_accuracy.png", dpi=300)
    plt.close()

def plot_test_loss(test_loss_list, epochs):
    plt.figure()
    plt.plot(range(1, len(test_loss_list)+1), test_loss_list, color='red')
    plt.title("Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/ELU_EP150_test_loss.png", dpi=300)
    plt.close()

# ==========================================================
# Training / Testing
# ==========================================================
def train(model, loader, criterion, optimizer, args, test_loader, device):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    test_loss_list = []
    best_epoch = 0
    best_train_acc = 0.0
    best_train_loss = 0.0
    best_test_loss = 0.0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        avg_acc = 0.0
        avg_loss = 0.0
        total_samples = len(loader.dataset)

        with torch.set_grad_enabled(True):
            for i, data in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}"), 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).sum().item()

        avg_loss /= total_samples
        avg_acc = (avg_acc / total_samples) * 100
        avg_acc_list.append(avg_acc)
        avg_loss_list.append(avg_loss)

        print(f"\nEpoch [{epoch}/{args.num_epochs}]")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Train Accuracy: {avg_acc:.2f}%")

        test_acc, test_loss = test(model, test_loader, device, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # 若 test_acc 更佳，更新最佳紀錄
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = deepcopy(model.state_dict())
            best_epoch = epoch
            best_train_acc = avg_acc
            best_train_loss = avg_loss
            best_test_loss = test_loss

    os.makedirs("./weights", exist_ok=True)
    torch.save(best_wts, "./weights/ELU_EP150_best.pt")

    print("\n✅ Training Done.")
    print("=" * 50)
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"Train Accuracy : {best_train_acc:.2f}%")
    print(f"Train Loss     : {best_train_loss:.4f}")
    print(f"Test Accuracy  : {best_acc:.2f}%")
    print(f"Test Loss      : {best_test_loss:.4f}")
    print("=" * 50)

    return avg_acc_list, avg_loss_list, test_acc_list, test_loss_list


def test(model, loader, device, criterion=None):
    model.eval()
    total = len(loader.dataset)
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct += pred.eq(labels).sum().item()

            # 若提供 criterion 就計算 loss
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

    avg_acc = (correct / total) * 100
    avg_loss = (total_loss / total) if criterion is not None else None
    return avg_acc, avg_loss


# ==========================================================
# Main
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Load dataset ===
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # === Model ===
    model = EEGNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000001)

    # === Train ===
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = train(
        model, train_loader, criterion, optimizer, args, test_loader, device
    )

    # === Plot results ===
    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)
    plot_test_loss(test_loss_list, args.num_epochs)

