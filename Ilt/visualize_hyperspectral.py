import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_sample(sample_path):
    """Load a single hyperspectral sample."""
    return np.load(sample_path)

def plot_spectral_signature(data, title="Spectral Signature"):
    """Plot the spectral signature of a hyperspectral sample."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.mean(axis=(0, 1)))
    plt.title(title)
    plt.xlabel("Band Number")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.show()

def plot_rgb_composite(data, title="RGB Composite"):
    """Plot RGB composite of the hyperspectral data."""
    # Assuming the data has many bands, we'll select bands that roughly correspond to RGB
    # You might need to adjust these indices based on your data
    r_band = data.shape[-1] // 3
    g_band = 2 * data.shape[-1] // 3
    b_band = data.shape[-1] - 1
    
    rgb = np.stack([
        data[:, :, r_band],
        data[:, :, g_band],
        data[:, :, b_band]
    ], axis=-1)
    
    # Normalize to 0-1 range
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_band_histogram(data, band_idx=None):
    """Plot histogram of a specific band or mean of all bands."""
    plt.figure(figsize=(10, 6))
    if band_idx is None:
        data_to_plot = data.mean(axis=(0, 1))
        title = "Histogram of Mean Band Values"
    else:
        data_to_plot = data[:, :, band_idx].flatten()
        title = f"Histogram of Band {band_idx}"
    
    sns.histplot(data_to_plot, bins=50)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

def visualize_sample(sample_path, label=None):
    """Visualize a single hyperspectral sample in multiple ways."""
    data = load_sample(sample_path)
    
    # Print basic information
    print(f"Data shape: {data.shape}")
    print(f"Value range: [{data.min():.3f}, {data.max():.3f}]")
    if label is not None:
        print(f"Label: {label}")
    
    # Create visualizations
    plot_spectral_signature(data, f"Spectral Signature{' - Label: ' + str(label) if label else ''}")
    plot_rgb_composite(data, f"RGB Composite{' - Label: ' + str(label) if label else ''}")
    plot_band_histogram(data)

# Example usage
if __name__ == "__main__":
    # Load the training data
    data_dir = Path("Deep_Learning/Project/Data/ot")
    train_csv = pd.read_csv("Deep_Learning/Project/Data/train.csv")
    
    # Visualize a few samples
    for idx, row in train_csv.head(3).iterrows():
        sample_path = data_dir / row['id']
        print(f"\nVisualizing sample: {row['id']}")
        visualize_sample(sample_path, row['label']) 