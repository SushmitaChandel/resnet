#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:10:58 2025

@author: sushmitachandel
"""
import matplotlib.pyplot as plt
import torch
import os

def save_curves(device, save_dir="exp1"):
    """
    This function can be used to save loss and accuracy train and test curves.
    Input
    -----
    device : str denoting the name of the computing device
        mps for metal, cpu for CPU, gpu for GPU
    save_dir : str denoting a path
        Path of the directory where the information of the last epoch is saved.
        In this directory, the output curves would also be saved.
    """
    
    path_best = save_dir+"/lastepoch.pth"
    checkpoint = torch.load(path_best, map_location=device)
    if os.path.exists(path_best):
        train_losses_epoch = checkpoint['train_losses_epoch']
        val_losses_epoch  = checkpoint['val_losses_epoch']
        train_accuracy_epoch = checkpoint['train_accuracy_epoch']
        val_accuracy_epoch  = checkpoint['val_accuracy_epoch']
    else:
        train_losses_epoch = [] # Trackers - accumulated over epochs
        val_losses_epoch= []
        train_accuracy_epoch = []
        val_accuracy_epoch = []
        
    epochs_list = list(range(1, len(train_losses_epoch)+1))
    plt.figure(figsize=(8,5))
    plt.plot(epochs_list, train_losses_epoch, label="Train Loss")
    plt.plot(epochs_list, val_losses_epoch, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir+"/crossentropyloss.png") 
    # plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(epochs_list, train_accuracy_epoch, label="Train Accuracy")
    plt.plot(epochs_list, val_accuracy_epoch, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir+"/accuracy.png") 
    # plt.sh
    
def save_training_log(file_path, best_val_acc, best_val_loss, total_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    with open(file_path, 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc}\n")
        f.write(f"Best Validation Loss: {best_val_loss}\n")
        f.write(f"Total Epochs: {total_epochs}\n\n")
        
        f.write("Epoch, Train Loss, Validation Loss, Train Accuracy, Validation Accuracy\n")
        for epoch in range(total_epochs):
            f.write(f"{epoch+1}, {train_losses[epoch]:.6f}, {val_losses[epoch]:.6f}, {train_accuracies[epoch]:.6f}, {val_accuracies[epoch]:.6f}\n")
