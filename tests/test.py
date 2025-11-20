#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:24:19 2025

@author: sushmitachandel
"""
from torchvision import models, transforms
import torch.nn as nn
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt

    
# Some important paths, class names and essential mappings.
save_dir = '/Users/sushmitachandel/Desktop/portfolio_of_projects/classification_of_ocean_features/resnet_github/models/train2/'
checkpoint_path = f"{save_dir}/bestepoch.pth" # path of the best model to load
class_names = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
index_to_class = {0:'F', 1:'G', 2:'H', 3:'I', 4:'J', 5:'K', 6:'L', 7:'M', 8:'N', 9:'O'}
index_to_classnames = {0:'POW', 1:'WS', 2:'MCC', 3:'RC', 4:'BS', 5:'SI', 6:'IB', 7:'LWA', 8:'AF', 9:'OF'}

# Load the model
device = 'mps'
model = models.resnet18(pretrained=False, num_classes=10).to(device) # Default model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict']) # Trained model
model.eval()

# Using the same transform as used during the training
val_transforms = transforms.Compose([
transforms.Resize((224, 224)),
transforms.Grayscale(num_output_channels=3),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
])

# Prepare the image
image_path = '/Users/sushmitachandel/Desktop/dataset/wang/data/wang2018/wang2018/data/F/s1a-wv1-slc-vv-20160527t023531-20160527t023534-011438-011698-023.png'
sample_img = Image.open(image_path)
sample_img_transform = val_transforms(sample_img)
# print(sample_img.shape)
# print(sample_img.dtype)
# print(sample_img.min())
# print(sample_img.max())
# print(label_idx)

# Predict
with torch.no_grad():
    sample_img_pred = sample_img_transform.unsqueeze(0).to(device)
    y_pred = model(sample_img_pred)
    _, predicted = torch.max(y_pred.data, 1)

# Display
fig, ax = plt.subplots()
ax.imshow(sample_img)
ax.axis('off')  # Hide all axes elements (ticks, labels, spines)
plt.show()
print("Predicted class:", index_to_classnames[predicted.item()])
print("True Class:",'POW')

