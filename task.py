import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file from the specified path
data = pd.read_csv('C:/Users/raghu/OneDrive/Documents/VS_CODE/PYTHON/data.csv')

# Print the overall shape of the dataset and show the first few rows
print("Dataset shape:", data.shape)
print(data.head())

# Separate labels and pixel data:
# If there is a 'label' column, use that; otherwise, assume the first column contains the label.
if 'label' in data.columns:
    labels = data['label']
    images = data.drop(columns=['label'])
else:
    labels = data.iloc[:, 0]
    images = data.iloc[:, 1:]

# Determine the number of pixels per image and the image dimension (assumes square images)
num_pixels = images.shape[1]
img_dim = int(np.sqrt(num_pixels))
if img_dim * img_dim != num_pixels:
    print("Warning: The number of pixel columns does not form a perfect square. Check the dataset!")
else:
    print(f"Each image is assumed to be {img_dim}x{img_dim} pixels.")

# Convert the pixel data to a numpy array for reshaping and plotting
image_array = images.values

# Display a few images (first 6) with their labels using matplotlib
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    img = image_array[i].reshape(img_dim, img_dim)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {labels.iloc[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Verify grayscale format by checking a single image
sample_img = image_array[0].reshape(img_dim, img_dim)
print("Sample image shape:", sample_img.shape)
print("Unique pixel values in the sample image:", np.unique(sample_img))
