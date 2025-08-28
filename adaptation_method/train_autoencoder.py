import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm 
from Inform_project_new.adaptation_method.EarlyStopping import EarlyStopping

def train_val_encoder(model, optimizer, Loss_func, num_epochs, train_dataloader, test_dataloader, run):
    avg_loss_train = []
    avg_loss_val = []
    best_val_loss = float('inf')
    best_model_path = 'best_autoencoder.pth'
    earlystopping = EarlyStopping(patience= 10)
    stop_epoch = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #---Training---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for label, train_data, mask in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        #for train_data in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            train_data = train_data.to(device, non_blocking = True)

            optimizer.zero_grad()
            
            outputs, latent = model(train_data)
            loss = Loss_func(outputs, train_data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        train_avg_loss = epoch_loss / len(train_dataloader)
        avg_loss_train.append(train_avg_loss)
  

        print(f"Train encodings: min={latent.min():.4f}, max={latent.max():.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for label, test_data, mask in test_dataloader:
                test_data = test_data.to(device)
                
                val_outputs, val_latent = model(test_data)
                loss = Loss_func(val_outputs, test_data)
                val_loss += loss.item()

        
                  
        val_avg_loss = val_loss / len(test_dataloader)
        avg_loss_val.append(val_avg_loss)
        
        
        print(f"Val latents: min={val_latent.min():.4f}, max={val_latent.max():.4f}")
        
        print(f" Train Loss = {train_avg_loss:.4f} ,Validation Loss = {val_avg_loss:.4f}")
        
        
        #Early stopping
        earlystopping(val_avg_loss)
        if earlystopping.early_stop and stop_epoch is None:
            stop_epoch = epoch
            print(f'Stopping early at epoch {epoch+1}')
            
        #Saving the best model    
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            
            #print(f'Saved new best model at epoch {epoch+1} with val_loss = {val_avg_loss:.4f}')

            
        #Logging Hyperparameters
        run.log({
            'epoch': epoch+1, 
            'train_loss':  train_avg_loss,
            'val_loss': val_avg_loss,
        })

    run.finish()

    return latent, val_latent, avg_loss_train, avg_loss_val, stop_epoch


def plot_loss(num_epochs, avg_loss_train, avg_loss_val, stop_epoch, run = None):
        plt.figure(figsize=(12,8))
        
        plt.plot(range(1, num_epochs+1), avg_loss_train, label = 'Training Loss')
        plt.plot(range(1, num_epochs+1), avg_loss_val, label = 'Validation Loss')
        if stop_epoch is not None:
            plt.axvline(x = stop_epoch, color = 'r', linestyle = '--', label = f'Early stop at Epoch{stop_epoch+1}')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation')
        plt.legend()
        plt.grid(True)    
        if run is not None:
            run.log({'Loss curve': wandb.Image(plt)})
        plt.show()
            
            
        

















