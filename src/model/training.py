import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import mlflow
from src.config import Config
from logger.logger import get_logger

# Write the training loop for lstm model

logger = get_logger() # Initialize logger for training module to log training process details

def fit_model(model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,epochs:int=8,lr:float=1e-3) -> nn.Module:
    
    """
    Train the LSTM model with early stopping 
    """
    
    # val_loader is used to evaluate the model performance on unseen data after each epoch to monitor for overfitting.this can reduce overfitting
    
    model.to(Config().device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) # adam optimizer for optimizing the model parameters
    criterion = nn.MSELoss() # mean squared error loss for  tasks like stock price prediction and time series data
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3) # scheduler
    best_val_loss = float("inf")
    patience, counter = 5, 0
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0 #float
        for X, Y in train_loader:
            X, Y = X.to(Config().device), Y.to(Config().device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()# backpropagation for improving model weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)# gradient clipping to avoid exploding gradients
            opt.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) # average training loss for the epoch
        logger.info(f"Epoch {ep}/{epochs} - Train Loss: {avg_train_loss:.5f}") # logging epochs and its average training loss
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X , Y in val_loader:
                X, Y = X.to(Config().device), Y.to(Config().device)
                pred_val = model(X)
                loss_val = criterion(pred_val, Y)
                val_loss += loss_val.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {ep}/{epochs} - Val Loss: {avg_val_loss:.5f}")

            # log the model metrics to mlflow
            try:
                current_lr=opt.param_groups[0]['lr']
                mlflow.log_metric("train_loss", avg_train_loss, step=ep)
                mlflow.log_metric("val_loss", avg_val_loss, step=ep)
                mlflow.log_metric("learning_rate", current_lr, step=ep)
            except Exception as e:
                logger.error(f"MLflow logging failed at epoch {ep}: {e}")
                
            # learning rate scheduler step
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info("Early stopping triggered")
                    break
    return model