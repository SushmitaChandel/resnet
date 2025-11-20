#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:33:54 2025

@author: sushmitachandel
"""
import argparse
import os
import sys
import torch.optim as optim
import torch.nn as nn
import torch

import data_loader
import models_src
import utils

# # This function can be used for plotting 
# import matplotlib.pyplot as plt
# def get_plot(epochs_list, train_epochs, val_epochs, xlabel, ylabel, 
#              title):
    
#     # print(f'Epochs till now: {epochs_list}')
#     # print(f'Training metrics till now: {train_epochs}')
#     # print(f'Validation metrics till now: {val_epochs}')
#     plt.figure(figsize=(8,5))
#     plt.plot(epochs_list, train_epochs, label="Train "+ylabel)
#     plt.plot(epochs_list, val_epochs, label="Val "+ylabel)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def train_output_layer(n_epochs, train_loader, val_loader, classes, device, 
                       save_dir="checkpoints"):
    
    # Define model
    model = models_src.model_level_first(classes, device=device)
    
    # Define loss function
    criterion_name = 'CrossEntropyLoss' 
    criterion = nn.CrossEntropyLoss()
    
    # Define Optimizer
    optimizer_name = 'Adam'
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize training losses and accuracy lists
    os.makedirs(save_dir, exist_ok=True)
    # Check whether the best epoc is saved or not.
    path_last = save_dir+"/lastepoch.pth"
    path_best = save_dir+"/bestepoch.pth"
    if os.path.exists(path_best):
        checkpoint_best = torch.load(path_best, map_location=device)
        best_val_loss = checkpoint_best['best_val_loss']
        best_epoch = checkpoint_best['best_epoch']
        checkpoint_last = torch.load(path_last, map_location=device)
        model.load_state_dict(checkpoint_last['model_state_dict'])
        optimizer.load_state_dict(checkpoint_last['optimizer_state_dict'])
        train_losses_epoch = checkpoint_last['train_losses_epoch']
        val_losses_epoch  = checkpoint_last['val_losses_epoch']
        train_accuracy_epoch = checkpoint_last['train_accuracy_epoch']
        val_accuracy_epoch  = checkpoint_last['val_accuracy_epoch']
    else:
        best_val_loss = float('inf') 
        best_epoch = 0
        train_losses_epoch = [] # Trackers - accumulated over epochs
        val_losses_epoch= []
        train_accuracy_epoch = []
        val_accuracy_epoch = []
        
    len_train_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
        
    # Begin training
    start_epoch = len(train_losses_epoch)
    for epoch in range(start_epoch, start_epoch+n_epochs):
        
        epoch = epoch+1
        print(f'Starting Epoch: {epoch}...')
        
        ######### Training #########
        trn_corr = 0
        running_loss = 0
        for i, data in enumerate(train_loader, 0):

            i += 1
            
            X_train, y_train = data
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Predict and collect all the correct predictions
            y_pred = model(X_train)
            _, predicted = torch.max(y_pred.data, 1)
            trn_corr += (predicted == y_train).sum()
            
            # Compute loss
            loss = criterion(y_pred, y_train)
            running_loss += loss.item() * X_train.size(0) 
            
            # Clear the gradients before training by setting to zero
            # Required for a fresh start
            optimizer.zero_grad()
            
            # Backpropogate
            loss.backward()

            # Update weights
            optimizer.step()
            
            if i % 200 == 0: # show interim results every 50 mini-batches
                accuracy = ((trn_corr.item())/(X_train.size(0)*i))*100
                l = running_loss / (X_train.size(0)*i)
                print(
                    f'Epoch: {epoch}, '
                    f'Mini-Batches Completed: {i}, '
                    f'Loss: {l:.3f}, '
                    f'Accuracy: {accuracy:.3f}'
                    )

        epoch_loss = running_loss / len_train_dataset
        train_losses_epoch.append(epoch_loss)
        epoch_accuracy = (trn_corr.item()) / len_train_dataset
        train_accuracy_epoch.append(epoch_accuracy)
        
        ######## Validation #########
        tst_corr = 0
        running_loss = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                X_val, y_val = data
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                # Predict and collect all the correct predictions
                y_pred = model(X_val)
                _, predicted = torch.max(y_pred.data, 1)
                tst_corr += (predicted == y_val).sum()
                    
                loss = criterion(y_pred, y_val)
                running_loss += loss.item() * X_val.size(0)

        epoch_loss = running_loss / len_val_dataset
        val_losses_epoch.append(epoch_loss)
        epoch_accuracy = (tst_corr.item()) / len_val_dataset
        val_accuracy_epoch.append(epoch_accuracy)

        ########### Saving best results #########
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_name': criterion_name,
                'optimizer_name': optimizer_name,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'best_val_acc': epoch_accuracy,
                'level': 'first'
            }, f"{save_dir}/bestepoch.pth")
            print(f"Saved best model at epoch {epoch},"
                  f" with val_loss: {best_val_loss:.4f}")
            
        ########### Saving last epoch #########
        # Save after each 2nd epoch
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses_epoch': train_losses_epoch,
                'val_losses_epoch': val_losses_epoch,
                'val_accuracy_epoch': val_accuracy_epoch,
                'train_accuracy_epoch': train_accuracy_epoch,
                'criterion_name': criterion_name,
                'optimizer_name': optimizer_name,
                'last_epoch': epoch,
                'level': 'first'
            }, f"{save_dir}/lastepoch.pth")
        
        # ########## Plotting on the go #########
        epochs_list = list(range(1, len(train_losses_epoch)+1))
        # get_plot(epochs_list, train_losses_epoch, val_losses_epoch, 
        #          "Epoch", "Loss", "Training & Validation Loss")
        # clear_output(wait=True)
            
        # get_plot(epochs_list, train_accuracy_epoch, val_accuracy_epoch, 
        #             "Epoch", "Accuracy", "Training & Validation Accuracy")
        # clear_output(wait=True)
        if epoch % 10 == 0:
            print(f'Epochs till now: {epochs_list}')
            print(f'Training metrics till now: {train_accuracy_epoch}')
            print(f'Validation metrics till now: {val_accuracy_epoch}')

    print('Finished Training')

def train_last2_and_output_layer(n_epochs, train_loader, val_loader, classes,
                                 device, save_dir="checkpoints"):
    
    model = models_src.model_level_second(classes, device=device)
    
    # Define loss function
    criterion_name = 'CrossEntropyLoss' 
    criterion = nn.CrossEntropyLoss()
    

    # Initialize training losses and accuracy lists
    os.makedirs(save_dir, exist_ok=True)
    # Check whether the best epoc is saved or not.
    path_last = save_dir+"/lastepoch.pth"
    path_best = save_dir+"/bestepoch.pth"
    if os.path.exists(path_best) and os.path.exists(path_last):
        # Initialize a default optimizer
        optimizer_name = 'Adam'
        checkpoint_best = torch.load(path_best, map_location=device)
        best_val_loss = checkpoint_best['best_val_loss']
        best_epoch = checkpoint_best['best_epoch']
        checkpoint_last = torch.load(path_last, map_location=device)
        train_losses_epoch = checkpoint_last['train_losses_epoch']
        val_losses_epoch  = checkpoint_last['val_losses_epoch']
        train_accuracy_epoch = checkpoint_last['train_accuracy_epoch']
        val_accuracy_epoch  = checkpoint_last['val_accuracy_epoch']
        model.load_state_dict(checkpoint_last['model_state_dict'])
        level = checkpoint_last['level']
        if level == 'first':
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = optim.Adam([
                {'params': model.layer3.parameters(), 'lr': 1e-5},
                {'params': model.layer4.parameters(), 'lr': 1e-5},
                {'params': model.fc.parameters(), 'lr': 1e-4}
            ], weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint_last['optimizer_state_dict'])
    else:
        best_val_loss = float('inf') 
        best_epoch = 0
        train_losses_epoch = [] # Trackers - accumulated over epochs
        val_losses_epoch= []
        train_accuracy_epoch = []
        val_accuracy_epoch = []
    
    len_train_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
        
    optimizer_name = 'Adam'
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # Begin training
    start_epoch = len(train_losses_epoch)
    for epoch in range(start_epoch, start_epoch+n_epochs):
        
        epoch = epoch+1
        print(f'Starting Epoch: {epoch}...')
        
        ######### Training #########
        trn_corr = 0
        running_loss = 0
        for i, data in enumerate(train_loader, 0):

            i += 1
            
            X_train, y_train = data
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Predict and collect all the correct predictions
            y_pred = model(X_train)
            _, predicted = torch.max(y_pred.data, 1)
            trn_corr += (predicted == y_train).sum()
            
            # Compute loss
            loss = criterion(y_pred, y_train)
            running_loss += loss.item() * X_train.size(0) 
            
            # Clear the gradients before training by setting to zero
            # Required for a fresh start
            optimizer.zero_grad()
            
            # Backpropogate
            loss.backward()

            # Update weights
            optimizer.step()
            
            if i % 200 == 0: # show interim results every 50 mini-batches
                accuracy = ((trn_corr.item())/(X_train.size(0)*i))*100
                l = running_loss / (X_train.size(0)*i)
                print(
                    f'Epoch: {epoch}, '
                    f'Mini-Batches Completed: {i}, '
                    f'Loss: {l:.3f}, '
                    f'Accuracy: {accuracy:.3f}'
                    )

        epoch_loss = running_loss / len_train_dataset
        train_losses_epoch.append(epoch_loss)
        epoch_accuracy = (trn_corr.item()) / len_train_dataset
        train_accuracy_epoch.append(epoch_accuracy)
        
        ######## Validation #########
        tst_corr = 0
        running_loss = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                X_val, y_val = data
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                # Predict and collect all the correct predictions
                y_pred = model(X_val)
                _, predicted = torch.max(y_pred.data, 1)
                tst_corr += (predicted == y_val).sum()

                loss = criterion(y_pred, y_val)
                running_loss += loss.item() * X_val.size(0)

        epoch_loss = running_loss / len_val_dataset
        val_losses_epoch.append(epoch_loss)
        epoch_accuracy = (tst_corr.item()) / len_val_dataset
        val_accuracy_epoch.append(epoch_accuracy)

        ########### Saving best results #########
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_name': criterion_name,
                'optimizer_name': optimizer_name,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'best_val_acc': epoch_accuracy,
                'level': 'second'
            }, f"{save_dir}/bestepoch.pth")
            print(f"Saved best model at epoch {epoch},"
                  f" with val_loss: {best_val_loss:.4f}")
            
        ########### Saving last epoch #########
        # Save after each 2nd epoch
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses_epoch': train_losses_epoch,
                'val_losses_epoch': val_losses_epoch,
                'val_accuracy_epoch': val_accuracy_epoch,
                'train_accuracy_epoch': train_accuracy_epoch,
                'criterion_name': criterion_name,
                'optimizer_name': optimizer_name,
                'last_epoch': epoch,
                'level': 'second'
            }, f"{save_dir}/lastepoch.pth")
                
        # ########## Plotting on the go #########
        epochs_list = list(range(1, len(train_losses_epoch)+1))
        # get_plot(epochs_list, train_losses_epoch, val_losses_epoch, 
        #          "Epoch", "Loss", "Training & Validation Loss")
        # clear_output(wait=True)
            
        # get_plot(epochs_list, train_accuracy_epoch, val_accuracy_epoch, 
        #             "Epoch", "Accuracy", "Training & Validation Accuracy")
        # clear_output(wait=True)
        if epoch % 10 == 0:
            print(f'Epochs till now: {epochs_list}')
            print(f'Training metrics till now: {train_accuracy_epoch}')
            print(f'Validation metrics till now: {val_accuracy_epoch}')

    print('Finished Training')
    
def main(n_epochs=20, path_data="data/", device="mps", 
         resume_train="False", save_dir="train2", batch_size=128):
    
    
    train_loader, val_loader, classes = data_loader.get_dataloaders(path_data)
    if resume_train == "False":
        if os.path.isdir(save_dir):
            print(f"Error: The save_dir '{save_dir}' already exists. Please specify a different directory.")
            sys.exit(1)
        else:
            print("Folder does not exisxt. Creating the folder.")
        # This is only trained once for a folder.
        train_output_layer(20, train_loader, val_loader, classes, device, 
                           save_dir=save_dir)
            
    if resume_train == "True":
        if os.path.isdir(save_dir) == False:
            print(f"Error: The save_dir '{save_dir}' does not exists. Please specify a different directory.")
            sys.exit(1)
        else:
            print("Continuing training.")
    
    train_last2_and_output_layer(n_epochs, train_loader, val_loader, classes, 
                                 device, save_dir=save_dir)
    
    utils.save_curves(device, save_dir=save_dir)
    
    # path_last = save_dir+"/lastepoch.pth"
    # path_best = save_dir+"/bestepoch.pth"
    # checkpoint_best = torch.load(path_best, map_location=device)
    # checkpoint_last = torch.load(path_last, map_location=device)
    # train_losses_epoch = checkpoint_last['train_losses_epoch']
    # val_losses_epoch  = checkpoint_last['val_losses_epoch']
    # train_accuracy_epoch = checkpoint_last['train_accuracy_epoch']
    # val_accuracy_epoch  = checkpoint_last['val_accuracy_epoch']
    # best_val_loss = checkpoint_best['best_val_loss']
    # best_val_acc = checkpoint_best['best_val_acc']
    # path_file =  save_dir+"/training_log.txt"
    # total_epochs = len(train_losses_epoch)
    # utils.save_training_log(path_file, best_val_acc, best_val_loss, total_epochs, 
    #                         train_losses_epoch, val_losses_epoch, 
    #                         train_accuracy_epoch, val_accuracy_epoch)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train with command line arguments")
    parser.add_argument("n_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("path_data", type=str, help="Path to dataset directory")
    parser.add_argument("device", type=str, help="Device to use for training (e.g., 'cpu' or 'cuda' or 'mps')")
    parser.add_argument("--resume_train", type=str, default="False", help="Flag suggesting whether or not to resume training from the previous step")
    parser.add_argument("--save_dir", type=str, default="train2", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loaders")
    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    path_data = args.path_data
    device = args.device
    resume_train = args.resume_train
    save_dir = args.save_dir
    batch_size = args.batch_size
    
    # Get absolute path of the directory to save all training history.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "../models/"+save_dir)
    save_dir = os.path.abspath(models_dir)
    print(save_dir)
    
    main(n_epochs=n_epochs, path_data=path_data, device=device, 
         resume_train=resume_train, save_dir=save_dir, batch_size=batch_size)