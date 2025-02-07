import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
import yaml

from unet2dmodel import UNet2D
from dataloader2d import get_loaders
from metrics import calculate_metrics, aggregate_metrics

def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    metrics_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Use AMP autocast in evaluation if CUDA is available
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(x)
                loss_val = criterion(outputs, y)
            losses.append(loss_val.item())
            metrics_list.append(calculate_metrics(outputs, y))
    return losses, aggregate_metrics(metrics_list)

def train_model(config, model, loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Ensure the learning rate is a float (in case it was read as a string)
    lr = float(config['lr']) if isinstance(config['lr'], str) else config['lr']
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    # Initialize AMP scaler if using CUDA for FP16 training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    best_dice = 0
    wandb.init(project=config['wandb_project'], config=config)
    
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        train_metrics_list = []
        
        for x, y in tqdm(loaders['train'], desc=f"Epoch {epoch+1} Training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision training if on CUDA
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(x)
                loss = criterion(outputs, y)
            
            # Backward pass using the scaler.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_metrics_list.append(calculate_metrics(outputs, y))
        
        avg_train_loss = np.mean(train_losses)
        train_metrics = aggregate_metrics(train_metrics_list)
        
        # Validation phase
        val_losses, val_metrics = evaluate(model, loaders['val'], device, criterion)
        avg_val_loss = np.mean(val_losses)
        scheduler.step(val_metrics['dice'])
        
        # Log metrics to wandb and print progress.
        log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': train_metrics['dice'],
            'train_jaccard': train_metrics['jaccard'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_dice': val_metrics['dice'],
            'val_jaccard': val_metrics['jaccard'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall']
        }
        wandb.log(log)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Dice = {val_metrics['dice']:.4f}")
        
        # Create checkpoints folder if not exists.
        os.makedirs("checkpoints", exist_ok=True)
        # Save checkpoint every 20 epochs.
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join("checkpoints", f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        # Save the best model (based on validation Dice).
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pth"))
            
if __name__ == "__main__":
    # Load parameters from YAML.
    with open("parameters.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    loaders = get_loaders(config)
    # Initialize the UNet model (assuming single-channel input and config['num_classes'] output channels).
    model = UNet2D(in_channels=1,
                   out_channels=config['num_classes'],
                   base_filters=config['base_filters'],
                   bilinear=config['bilinear'])
    
    train_model(config, model, loaders)
