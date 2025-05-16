import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.dataset import Affwild2GraphDataset
from notebooks.models.GTAT.GTAT import GCATopo
from utils.config import (
    NUM_FRAMES, 
    VISUAL_FRAMES_SUBDIR_NAME, 
    AUDIO_FILES_SUBDIR_NAME
)

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Concordance Correlation Coefficient (CCC) for emotion prediction.
    """
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    
    y_true_var = torch.var(y_true, unbiased=False)
    y_pred_var = torch.var(y_pred, unbiased=False)
    
    covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    
    ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return ccc

class CCCLoss(nn.Module):
    """
    Loss based on the Concordance Correlation Coefficient (CCC) for emotion prediction.
    """
    def __init__(self):
        super(CCCLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        return 1 - ccc  # Minimize (1 - CCC) to maximize CCC

class EmotionTrainer:
    def __init__(self,
                 root_dir: str,
                 train_video_ids_csv: str,
                 val_video_ids_csv: str,
                 batch_size: int = 8,
                 epochs: int = 50,
                 lr: float = 0.001,
                 hidden_dim: int = 128,
                 topology_dim: int = 15,
                 num_gtat_layers: int = 2,
                 gtat_heads: int = 4,
                 dropout: float = 0.2,
                 output_dir: str = "outputs"):
        self.root_dir = root_dir
        self.train_video_ids_csv = train_video_ids_csv
        self.val_video_ids_csv = val_video_ids_csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.topology_dim = topology_dim
        self.num_gtat_layers = num_gtat_layers
        self.gtat_heads = gtat_heads
        self.dropout = dropout
        self.output_dir = output_dir

        # Create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load training and validation CSV files
        self.train_data_df = pd.read_csv(self.train_video_ids_csv)
        self.val_data_df = pd.read_csv(self.val_video_ids_csv)
        
        # Extract video ids
        self.train_video_ids = self.train_data_df['video_id'].tolist()
        self.val_video_ids = self.val_data_df['video_id'].tolist()
        
        # Create the datasets
        self.train_dataset = Affwild2GraphDataset(
            video_ids=self.train_video_ids,
            root_dir=self.root_dir,
            annotations_df=self.train_data_df
        )
        self.val_dataset = Affwild2GraphDataset(
            video_ids=self.val_video_ids,
            root_dir=self.root_dir,
            annotations_df=self.val_data_df
        )
        
        # Create DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize the model and move it to the appropriate device
        self.model = GCATopo(
            hidden_dim=self.hidden_dim,
            topology_dim=self.topology_dim,
            num_gtat_layers=self.num_gtat_layers,
            gtat_heads=self.gtat_heads,
            dropout=self.dropout
        ).to(self.device)
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Define loss functions for both valence and arousal predictions
        self.valence_criterion = CCCLoss()
        self.arousal_criterion = CCCLoss()
        
        # Dictionary to store training history for plotting/analysis later
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_valence_ccc': [],
            'val_valence_ccc': [],
            'train_arousal_ccc': [],
            'val_arousal_ccc': []
        }
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        valence_ccc_sum = 0
        arousal_ccc_sum = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for data in progress_bar:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Get target values for valence and arousal
            valence_target = data.valence.view(-1, 1).float()
            arousal_target = data.arousal.view(-1, 1).float()
            
            # Model predictions
            valence_pred, arousal_pred = self.model(data)
            
            # Compute individual losses
            valence_loss = self.valence_criterion(valence_pred, valence_target)
            arousal_loss = self.arousal_criterion(arousal_pred, arousal_target)
            loss = (valence_loss + arousal_loss) / 2.0
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            valence_ccc = 1 - valence_loss.item()
            arousal_ccc = 1 - arousal_loss.item()
            valence_ccc_sum += valence_ccc
            arousal_ccc_sum += arousal_ccc
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'v_ccc': valence_ccc,
                'a_ccc': arousal_ccc
            })
        
        avg_loss = total_loss / num_batches
        avg_valence_ccc = valence_ccc_sum / num_batches
        avg_arousal_ccc = arousal_ccc_sum / num_batches
        
        return avg_loss, avg_valence_ccc, avg_arousal_ccc

    def validate(self):
        self.model.eval()
        total_loss = 0
        valence_ccc_sum = 0
        arousal_ccc_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for data in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                
                # Get target values
                valence_target = data.valence.view(-1, 1).float()
                arousal_target = data.arousal.view(-1, 1).float()
                
                # Get predictions
                valence_pred, arousal_pred = self.model(data)
                
                # Compute losses
                valence_loss = self.valence_criterion(valence_pred, valence_target)
                arousal_loss = self.arousal_criterion(arousal_pred, arousal_target)
                loss = (valence_loss + arousal_loss) / 2.0
                
                total_loss += loss.item()
                valence_ccc_sum += (1 - valence_loss.item())
                arousal_ccc_sum += (1 - arousal_loss.item())
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_valence_ccc = valence_ccc_sum / num_batches
        avg_arousal_ccc = arousal_ccc_sum / num_batches
        
        return avg_loss, avg_valence_ccc, avg_arousal_ccc

    def plot_training_curves(self):
        plt.figure(figsize=(15, 5))
    
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
    
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_valence_ccc'], label='Train')
        plt.plot(self.history['val_valence_ccc'], label='Validation')
        plt.title('CCC - Valence')
        plt.xlabel('Epoch')
        plt.legend()
    
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_arousal_ccc'], label='Train')
        plt.plot(self.history['val_arousal_ccc'], label='Validation')
        plt.title('CCC - Arousal')
        plt.xlabel('Epoch')
        plt.legend()
    
        plt.tight_layout()
        curve_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(curve_path)
        plt.close()

    def run(self):
        best_val_loss = float('inf')
        best_model_path = os.path.join(self.output_dir, 'best_GTAT.pth')
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss, train_valence_ccc, train_arousal_ccc = self.train_one_epoch()
            val_loss, val_valence_ccc, val_arousal_ccc = self.validate()
            
            # Update training history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_valence_ccc'].append(train_valence_ccc)
            self.history['val_valence_ccc'].append(val_valence_ccc)
            self.history['train_arousal_ccc'].append(train_arousal_ccc)
            self.history['val_arousal_ccc'].append(val_arousal_ccc)
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved with validation loss: {val_loss:.4f}")
            
            print(f"Train  - Loss: {train_loss:.4f}, V-CCC: {train_valence_ccc:.4f}, A-CCC: {train_arousal_ccc:.4f}")
            print(f"Valid  - Loss: {val_loss:.4f}, V-CCC: {val_valence_ccc:.4f}, A-CCC: {val_arousal_ccc:.4f}")
        
        # Save training history to CSV
        history_df = pd.DataFrame(self.history)
        history_csv_path = os.path.join(self.output_dir, 'training_history.csv')
        history_df.to_csv(history_csv_path, index=False)
        
        # Plot and save training curves
        self.plot_training_curves()
        
        print(f"Training completed. Results saved in {self.output_dir}")