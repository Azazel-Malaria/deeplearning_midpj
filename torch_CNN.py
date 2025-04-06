import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import gzip
import numpy as np
from struct import unpack
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            Flatten(),
            nn.Linear(in_features=64*7*7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        with gzip.open(images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
        
        with gzip.open(labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

train_dataset = MNISTDataset(train_images_path, train_labels_path, transform=transform)
test_dataset = MNISTDataset(test_images_path, test_labels_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.06)
scheduler = MultiStepLR(optimizer, milestones=[800, 2400, 4000], gamma=0.5)
criterion = nn.CrossEntropyLoss()

def plot_training_curves(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def visualize_predictions(model, device, test_loader, num_samples=12):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

train_losses = []
train_accs = []
test_losses = []
test_accs = []
all_preds = []
all_labels = []

epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    epoch_train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss = epoch_train_loss / len(train_loader)
    train_acc = 100. * correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch}:')
    print(f'Train Loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_acc:.2f}%)')
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_acc:.2f}%)')

plot_training_curves(train_losses, train_accs, test_losses, test_accs)
plot_confusion_matrix(all_labels, all_preds, classes=[str(i) for i in range(10)])
visualize_predictions(model, device, test_loader)

torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved to mnist_cnn.pth")