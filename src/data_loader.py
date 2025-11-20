#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:04:09 2025

@author: sushmitachandel
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

def get_dataloaders(path, batch_size=128):
    
    """
    This function can be used to create train and val data loaders.
    
    Input
    -----
    path : str
        Denotes path of the directory containing all the images. 
        The directory should contain sub-folders such that each folder contains
        images form a class
    batch_size : int, defaults to 128
        Size of a batch
        
    Returns
    -------
    train_loader : object
        Train loader object
    test_loader : object
        Test loader object
    classes : list of strs
        List containing class abbreviations
    """
    
    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(path)
    # e.g., 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=generator)

    # Apply different transforms to each subset
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    print(f'Number of examples in training dataset are: {len(train_dataset)}')
    print(f'Number of examples in validating dataset are: {len(val_dataset)}')

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # List of class names (subfolder names) and numeric mapping
    print(f' The list of classes are: {dataset.classes}')
    print(f' Dict mapping class name → numeric label: {dataset.class_to_idx}')
    classes = dataset.classes
    
    return train_loader, val_loader, classes