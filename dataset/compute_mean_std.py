import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_dir = 'tiny-imagenet-200'
train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=transforms.ToTensor())

loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)

mean = 0.0
var = 0.0  # For E[X^2]
total_images = 0

for images, _ in loader:
    batch_samples = images.size(0)
    # Flatten spatial dims: [B, C, H*W]
    images_flat = images.view(batch_samples, images.size(1), -1)
    
    # Accumulate for mean: sum(E[X]) over batches
    mean += images_flat.mean(2).sum(0)
    
    # Accumulate for var: sum(E[X^2]) over batches
    var += (images_flat ** 2).mean(2).sum(0)
    
    total_images += batch_samples

mean /= total_images
var /= total_images
std = torch.sqrt(var - mean ** 2)

print(f"Dataset Mean: {mean.tolist()}")
print(f"Dataset Std:  {std.tolist()}")
