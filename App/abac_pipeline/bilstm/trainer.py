"""Trainer for BiLSTM model."""

import torch
from torch.utils.data import DataLoader
from typing import Dict
import os
import logging
import random
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BiLSTMTrainer:
    
    def __init__(self, model: torch.nn.Module, config, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        set_seed(self.config.seed)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Train the model with early stopping based on F1 score."""
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("DataLoaders cannot be empty.")
            
        best_val_f1 = -1.0
        patience_counter = 0
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_bilstm.pt")
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y, batch_mask in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                emissions = self.model(batch_x, batch_mask)
                loss = self.model.compute_loss(emissions, batch_y, batch_mask)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.clip_grad_norm
                )
                
                self.optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]
            
            logger.info(
                f"Epoch {epoch+1:02d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )
            
            # Early stopping based on F1 rather than loss
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Use weights_only=True for security
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Saved new best model with val F1 {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best model for inference
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
            
        return {"best_val_f1": best_val_f1}
            
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate loss and span-level NER metrics (Precision, Recall, F1)."""
        self.model.eval()
        total_loss = 0.0
        
        all_true_tags = []
        all_pred_tags = []
        
        # Access idx2tag from vocab
        idx2tag = self.model.vocab.idx2tag
        
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                # Compute loss
                emissions = self.model(batch_x, batch_mask)
                loss = self.model.compute_loss(emissions, batch_y, batch_mask)
                total_loss += loss.item()
                
                # Get predictions
                preds = self.model.predict(batch_x, batch_mask)
                
                # Convert indices to tags for seqeval
                for i in range(len(preds)):
                    seq_len = len(preds[i])
                    # True tags (ignoring padding)
                    true_seq = [idx2tag[idx.item()] for idx in batch_y[i, :seq_len]]
                    pred_seq = [idx2tag[idx] for idx in preds[i]]
                    
                    all_true_tags.append(true_seq)
                    all_pred_tags.append(pred_seq)
                    
        avg_loss = total_loss / len(dataloader)
        
        # Compute metrics via seqeval
        f1 = f1_score(all_true_tags, all_pred_tags)
        p = precision_score(all_true_tags, all_pred_tags)
        r = recall_score(all_true_tags, all_pred_tags)
        
        return {
            "loss": avg_loss,
            "precision": p,
            "recall": r,
            "f1": f1
        }
