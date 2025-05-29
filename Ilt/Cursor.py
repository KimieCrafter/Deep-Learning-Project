import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_BANDS = 125
PATCH_SIZE = 128
NUM_CLASSES = 100  # We want classes 0-99
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperspectralDataset(Dataset):
    def __init__(self, df, base_path, patch_size=PATCH_SIZE, augment=False, num_bands=NUM_BANDS):
        self.df = df
        self.base_path = base_path
        self.patch_size = patch_size
        self.augment = augment
        self.num_bands = num_bands
        
        # Print dataset statistics
        print(f"Dataset size: {len(df)}")
        print(f"Label range: {df['label'].min()} to {df['label'].max()}")
        print(f"Number of unique labels: {df['label'].nunique()}")
        
        # Enhanced augmentations with more aggressive transformations
        self.transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.7),
            K.RandomCrop((patch_size, patch_size), padding=4, p=0.7),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.3),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3)
        )
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.base_path}/{row['id']}"

        try:
            img = np.load(img_path)
            
            # Handle different image dimensions
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], self.num_bands, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] > self.num_bands:
                    img = img[:, :, :self.num_bands]
                elif img.shape[2] < self.num_bands:
                    pad_width = ((0, 0), (0, 0), (0, self.num_bands - img.shape[2]))
                    img = np.pad(img, pad_width, mode='constant')

            # Normalize and convert to tensor
            img = img.astype(np.float32) / 65535.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            # Apply augmentations
            if self.augment:
                img = self.transform(img.unsqueeze(0)).squeeze(0)

            # Resize if necessary
            if img.shape[1] != self.patch_size or img.shape[2] != self.patch_size:
                img = F.interpolate(img.unsqueeze(0), size=(self.patch_size, self.patch_size), 
                                  mode='bilinear', align_corners=True).squeeze(0)

            # Labels are already 0-based from the DataFrame
            label = torch.tensor(row['label'], dtype=torch.long)
            return img, label

        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Instead of returning zeros, skip this sample
            return self.__getitem__((idx + 1) % len(self))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1, 1)

class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Enhanced feature extraction with more gradual reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def evaluate_model(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return total_loss / len(loader.dataset), accuracy, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        valid_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                continue
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if torch.isnan(outputs).any():
                continue
                
            loss = criterion(outputs, labels)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                valid_samples += inputs.size(0)
        
        if valid_samples > 0:
            train_loss /= valid_samples
            val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion)
            
            if scheduler is not None:
                scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Sample predictions: {val_preds[:5]}, True labels: {val_labels[:5]}")
            
            # Print class-wise accuracy
            unique_labels = np.unique(val_labels)
            for label in unique_labels:
                mask = val_labels == label
                if np.sum(mask) > 0:
                    class_acc = np.mean(val_preds[mask] == val_labels[mask])
                    print(f"Class {label} accuracy: {class_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'Spectrum_CNN_best.pth')
        else:
            print(f"Epoch {epoch+1}: No valid training samples")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model

def main():
    # Load and prepare data
    train_df = pd.read_csv('C:/IIUM/AI Note IIUM/Deep_Learning/Project/Data/train.csv')
    base_path = 'C:/IIUM/AI Note IIUM/Deep_Learning/Project/Data/ot'
    
    # Print label statistics before adjustment
    print("Label range before adjustment:", train_df['label'].min(), "to", train_df['label'].max())
    print("Number of unique labels before adjustment:", train_df['label'].nunique())
    
    # Adjust labels to be 0-based and ensure they're in range 0-99
    train_df['label'] = train_df['label'] - 1  # Make 0-based
    train_df = train_df[train_df['label'] < NUM_CLASSES]  # Remove any labels >= 100
    
    # Print label statistics after adjustment
    print("Label range after adjustment:", train_df['label'].min(), "to", train_df['label'].max())
    print("Number of unique labels after adjustment:", train_df['label'].nunique())
    
    # Calculate class weights for imbalanced data
    label_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    
    # Ensure we have weights for all classes
    class_weights = torch.zeros(NUM_CLASSES, device=DEVICE)
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (NUM_CLASSES * count)
    
    print("Class weights shape:", class_weights.shape)
    print("Number of classes with weights:", (class_weights > 0).sum().item())
    
    # Remove problematic samples
    train_df = train_df[train_df['id'] != 'sample2451']
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    
    train_dataset = HyperspectralDataset(train_df, base_path, augment=True)
    val_dataset = HyperspectralDataset(val_df, base_path, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model and training components
    model = HyperspectralCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use weighted loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    model = train_model(model, train_loader, val_loader, EPOCHS, criterion, optimizer, scheduler)
    
    # Load best model
    checkpoint = torch.load('Spectrum_CNN_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == '__main__':
    model = main()
