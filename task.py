import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # using a sample dataset

# Load the dataset
digits = load_digits() 
images = digits.images      # images have shape (n_samples, height, width)
labels = digits.target      # corresponding labels

# Check and print the dataset shape
print("Dataset shape (images):", images.shape)
print("Dataset shape (labels):", labels.shape)

# Display a few images with their labels
plt.figure(figsize=(10, 4))
for i in range(6):  # display first 6 images
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')  # display image in grayscale
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Verify grayscale format by checking a single image
sample_img = images[0]
print("Sample image shape:", sample_img.shape)
print("Unique pixel values in sample image:", np.unique(sample_img))
