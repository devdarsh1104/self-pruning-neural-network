import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores / 0.08)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            PrunableLinear(3072, 1024),
            nn.ReLU(),
            PrunableLinear(1024, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 10),
        )

    def forward(self, x):
        return self.net(x)

    def prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

def sparsity_loss(model):
    total = 0.0
    count = 0

    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores / 0.08)
        total += (gates + 0.01 * gates**2).sum()
        count += gates.numel()

    return total / count

@torch.no_grad()
def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores / 0.08)
        pruned += (gates < threshold).sum().item()
        total += gates.numel()

    return pruned / total

def debug_gates(model):
    all_gates = []

    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores / 0.08).detach().cpu()
        all_gates.append(gates.view(-1))

    all_gates = torch.cat(all_gates)

    print("\n[Gate Stats]")
    print("Min:", all_gates.min().item())
    print("Max:", all_gates.max().item())
    print("Mean:", all_gates.mean().item())

def train(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        ce = F.cross_entropy(out, y)
        sp = sparsity_loss(model)

        loss = ce + lam * sp
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

    return correct / total

def get_data():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    return trainloader, testloader

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainloader, testloader = get_data()

    lambdas = [0.05, 0.1, 0.2]

    for lam in lambdas:
        print(f"\n===== Lambda = {lam} =====")

        model = PrunableMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(15):
            loss, acc = train(model, trainloader, optimizer, lam, device)
            print(f"Epoch {epoch+1} | Loss {loss:.4f} | Acc {acc*100:.2f}%")

        test_acc = evaluate(model, testloader, device)
        sparsity = compute_sparsity(model)

        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Sparsity: {sparsity*100:.2f}%")

        debug_gates(model)

if __name__ == "__main__":
    main()
