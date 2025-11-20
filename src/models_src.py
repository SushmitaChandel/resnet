#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:29:51 2025

@author: sushmitachandel
"""

from torchvision import models
import torch.nn as nn

def model_level_first(classes, device='mps'):
    
    # Define model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad_(True)
    
    return model

def model_level_second(classes, device='mps'):
    
    model = models.resnet18(pretrained=True)
    # print(model)
    class_names = classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name:
            param.requires_grad = True
    model.fc.requires_grad_(True)
    
    return model
    