# src/train_patch_classifier.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimplePatchCNN


DATA_DIR = "data/patch_dataset"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 12
MODEL_PATH = "models/patch_classifier.pth"


def load_data():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f">> Loaded dataset with {len(dataset)} samples")
    print(f">> Classes: {dataset.classes}")

    return loader, len(dataset.classes)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">> Training on:", device)

    loader, num_classes = load_data()

    model = SimplePatchCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("\n>> Model saved to", MODEL_PATH)


if __name__ == "__main__":
    train()
