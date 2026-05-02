import torchvision.datasets as datasets
import os

clean = datasets.ImageFolder('./dataset/tiny-imagenet-200/val').class_to_idx
corrupt = datasets.ImageFolder('./dataset/tiny-imagenet-c/gaussian_noise/1').class_to_idx
print("Match:", clean == corrupt)