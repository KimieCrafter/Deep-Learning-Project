import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchvision import transforms
from typing import Tuple, List, Dict, Any

# Constants
BATCH_SIZE = 16  # Smaller batch size for better memory management
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_BANDS = 100  # Keep all bands as they might be important
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
NUM_FOLDS = 5
ACCUMULATION_STEPS = 1
PATCH_SIZE = 32  # Keep smaller patch size for faster processing

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)  # Output single channel attention map
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Create attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        
        x_att = F.relu(self.bn1(self.conv1(concat)))
        x_att = F.relu(self.bn2(self.conv2(x_att)))
        attention = self.sigmoid(self.conv3(x_att))
        
        # Apply attention to input
        return x * attention  # Broadcasting will handle the channel dimension

class EnhancedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x).view(b, c))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x).view(b, c))))
        
        out = avg_out + max_out
        out = self.bn(out.view(b, c))
        return self.sigmoid(out).view(b, c, 1, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.ca = EnhancedChannelAttention(out_channels)
        self.sa = EnhancedSpatialAttention(out_channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply channel attention
        out = self.ca(out) * out
        
        # Apply spatial attention
        out = self.sa(out) * out
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels=NUM_BANDS):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Improved regressor with skip connections
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x * 100  # Scale to 0-100 range

class HyperspectralDataset(Dataset):
    def __init__(self, df, base_path, patch_size=PATCH_SIZE, augment=False, is_test=False):
        self.df = df
        self.base_path = base_path
        self.patch_size = patch_size
        self.augment = augment
        self.is_test = is_test
        
        # Pre-compute file paths
        self.file_paths = [f"{self.base_path}/{npy_file}" for npy_file in self.df['id']]
        
        # Define augmentations using PyTorch transforms - simplified for speed
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.transform = transforms.Compose([])  # Empty transform
        
        print(f"Dataset initialized with {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # Load data
            data = np.load(self.file_paths[idx])
            
            if len(data.shape) == 3:
                data = data.transpose(2, 0, 1)  # Change to (channels, height, width)
            
            if data.shape[0] != NUM_BANDS:
                if data.shape[0] > NUM_BANDS:
                    data = data[:NUM_BANDS]
                else:
                    padding = np.zeros((NUM_BANDS - data.shape[0], data.shape[1], data.shape[2]))
                    data = np.concatenate([data, padding], axis=0)
            
            # Convert to torch tensor
            data = torch.from_numpy(data).float()
            
            # Resize if needed - using nearest neighbor for speed
            if data.shape[1] != self.patch_size or data.shape[2] != self.patch_size:
                data = F.interpolate(data.unsqueeze(0), 
                                   size=(self.patch_size, self.patch_size), 
                                   mode='nearest').squeeze(0)
            
            # Apply augmentations if enabled and not test data
            if self.augment and not self.is_test:
                # Apply spatial augmentations to each channel
                augmented_data = []
                for i in range(data.shape[0]):
                    channel_data = data[i:i+1]  # Keep channel dimension
                    augmented_channel = self.transform(channel_data)
                    augmented_data.append(augmented_channel)
                data = torch.cat(augmented_data, dim=0)
            
            # Normalize data
            data = (data - data.mean()) / (data.std() + 1e-8)
            
            if self.is_test:
                return data
            else:
                disease_percentage = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
                return data, disease_percentage
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            raise e

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, fold):
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_mae = float('inf')
    patience = 15
    patience_counter = 0
    scaler = GradScaler('cuda')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        valid_samples = 0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                continue
            
            with autocast('cuda'):
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)  # Remove last dimension to match labels
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * ACCUMULATION_STEPS
            train_mae += torch.abs(outputs - labels).sum().item()
            valid_samples += inputs.size(0)
        
        if valid_samples > 0:
            train_loss /= valid_samples
            train_mae /= valid_samples
            
            # Validation phase
            val_loss, val_mae = evaluate_model(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_maes.append(train_mae)
            val_maes.append(val_mae)
            
            scheduler.step(val_loss)
            
            print(f"Fold {fold}, Epoch {epoch+1}:")
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, f'best_model_fold_{fold}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model, train_losses, val_losses, train_maes, val_maes

def evaluate_model(model, loader, criterion, device=DEVICE) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)  # Remove last dimension to match labels
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_mae += torch.abs(outputs - labels).sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    
    return avg_loss, avg_mae

def main():
    print("Starting main function...")
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('C:/IIUM/AI Note IIUM/Deep_Learning/Project/Data/train.csv')
    test_df = pd.read_csv('C:/IIUM/AI Note IIUM/Deep_Learning/Project/Data/test.csv')
    base_path = 'C:/IIUM/AI Note IIUM/Deep_Learning/Project/Data/ot'
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results: List[Dict[str, Any]] = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        print(f"\nTraining Fold {fold + 1}/{NUM_FOLDS}")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Split data
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = HyperspectralDataset(train_fold, base_path, augment=True, is_test=False)
        val_dataset = HyperspectralDataset(val_fold, base_path, augment=False, is_test=False)
        
        # Create data loaders with optimized parameters
        print("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=0,  # Keep at 0 for debugging
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,  # Keep at 0 for debugging
            pin_memory=True
        )
        
        print("Initializing model...")
        # Initialize model and training components
        model = HyperspectralCNN().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print("Starting training...")
        # Train model
        model, train_losses, val_losses, train_maes, val_maes = train_model(
            model, train_loader, val_loader, EPOCHS, criterion, optimizer, scheduler, fold
        )
        
        # Store results
        fold_results.append({
            'fold': fold,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_maes': train_maes,
            'val_maes': val_maes
        })
        
        # Plot training curves for this fold
        plot_training_curves(train_losses, val_losses, train_maes, val_maes, fold)
    
    # Make predictions on test set using ensemble of models
    test_dataset = HyperspectralDataset(test_df, base_path, augment=False, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    all_preds = []
    for fold in range(NUM_FOLDS):
        # Load best model for this fold
        checkpoint = torch.load(f'best_model_fold_{fold}.pth')
        model = HyperspectralCNN().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for inputs in tqdm(test_loader, desc=f"Making predictions (Fold {fold+1})"):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                fold_preds.extend(outputs.cpu().numpy())
        
        all_preds.append(fold_preds)
    
    # Average predictions from all folds
    final_preds = np.mean(all_preds, axis=0)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'id': test_df['id'],
        'predicted_percentage': final_preds
    })
    results_df.to_csv('test_predictions.csv', index=False)
    
    # Plot ensemble results
    plot_ensemble_results(fold_results)

def plot_training_curves(train_losses, val_losses, train_maes, val_maes, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss (Fold {fold+1})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_maes, label='Train MAE', marker='o')
    ax2.plot(val_maes, label='Val MAE', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (%)')
    ax2.set_title(f'Training and Validation MAE (Fold {fold+1})')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_curves_fold_{fold+1}.png')
    plt.close()

def plot_ensemble_results(fold_results):
    # Plot average performance across folds
    avg_train_losses = np.mean([r['train_losses'] for r in fold_results], axis=0)
    avg_val_losses = np.mean([r['val_losses'] for r in fold_results], axis=0)
    avg_train_maes = np.mean([r['train_maes'] for r in fold_results], axis=0)
    avg_val_maes = np.mean([r['val_maes'] for r in fold_results], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(avg_train_losses, label='Avg Train Loss', marker='o')
    ax1.plot(avg_val_losses, label='Avg Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Average Training and Validation Loss Across Folds')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(avg_train_maes, label='Avg Train MAE', marker='o')
    ax2.plot(avg_val_maes, label='Avg Val MAE', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (%)')
    ax2.set_title('Average Training and Validation MAE Across Folds')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ensemble_results.png')
    plt.close()

if __name__ == "__main__":
    main() 