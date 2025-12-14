import numpy as np

# Load data
img = np.load('data/sailor/sub-17_image.npy')
lbl = np.load('data/sailor/sub-17_label.npy')

# Check shapes
print(f"Image: {img.shape}")  # e.g., (24, 240, 240, 155) = 4 modalities × 6 sessions
print(f"Label: {lbl.shape}")  # e.g., (6, 240, 240, 155) = 6 sessions

# Check value ranges
print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")  # Should be [0, 1]
print(f"Label values: {np.unique(lbl)}")  # Should be [0, 1, 3]

# Check for NaN/Inf
assert not np.isnan(img).any(), "NaN detected in images"
assert not np.isinf(img).any(), "Inf detected in images"