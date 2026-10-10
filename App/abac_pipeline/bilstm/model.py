"""BiLSTM-CRF model for constraint sequence labeling."""

import torch
import torch.nn as nn
from torchcrf import CRF
from .config import BiLSTMConfig
from .vocabulary import Vocabulary
from .embeddings import create_embedding_layer

class BiLSTMConstraintModel(nn.Module):
    """BiLSTM-CRF model for BIO tagging of constraints."""
    
    def __init__(self, config: BiLSTMConfig, vocab: Vocabulary, spacy_nlp=None):
        super(BiLSTMConstraintModel, self).__init__()
        self.config = config
        self.vocab = vocab
        
        # 1. Embedding Layer (may override embedding_dim if spaCy vectors have different dim)
        self.embedding = create_embedding_layer(
            vocab=vocab,
            embedding_dim=config.embedding_dim,
            fasttext_path=config.fasttext_model_path,
            glove_path=config.glove_model_path,
            spacy_nlp=spacy_nlp,
            freeze=config.freeze_embeddings
        )
        
        # Actual embedding dim may differ from config if spaCy vectors overrode it
        actual_embedding_dim = self.embedding.embedding_dim
        
        # 2. BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=actual_embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 3. Projection Layer
        self.dropout = nn.Dropout(config.dropout)
        self.hidden2tag = nn.Linear(config.hidden_dim * 2, vocab.num_tags)
        
        # 4. CRF Layer
        if config.use_crf:
            self.crf = CRF(vocab.num_tags, batch_first=True)
        else:
            self.crf = None
            
        # Loss function for non-CRF fallback
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        
    def _get_lstm_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiLSTM to get emission scores."""
        # x: (batch_size, seq_len)
        # mask: (batch_size, seq_len) bool
        
        # Get sequence lengths from mask for packing
        seq_lengths = mask.sum(dim=1).cpu().int()
        
        embeds = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        embeds = self.dropout(embeds)
        
        # Pack padded sequence to avoid computing over padding and fix bidirectional states
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_lstm_out, _ = self.lstm(packed_embeds)
        
        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, total_length=x.size(1)
        )
        
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, num_tags)
        
        return emissions

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns emission scores."""
        return self._get_lstm_features(x, mask)
        
    def compute_loss(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute loss (CRF negative log likelihood or masked CrossEntropy)."""
        if self.crf is not None:
            # CRF expects byte mask in torchcrf
            byte_mask = mask.bool()
            # crf returns log likelihood, we want negative log likelihood (loss)
            loss = -self.crf(emissions, tags, mask=byte_mask, reduction='mean')
            return loss
        else:
            # Fallback to CrossEntropy
            emissions_flat = emissions.reshape(-1, self.vocab.num_tags)
            tags_flat = tags.reshape(-1)
            
            loss = self.loss_fct(emissions_flat, tags_flat)
            
            mask_flat = mask.reshape(-1).float()
            loss = loss * mask_flat
            
            # Avoid division by zero
            denom = torch.clamp(mask_flat.sum(), min=1.0)
            return loss.sum() / denom

    def predict(self, x: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Decode the sequence of tags."""
        emissions = self._get_lstm_features(x, mask)
        
        if self.crf is not None:
            # Viterbi decoding
            byte_mask = mask.bool()
            predictions = self.crf.decode(emissions, mask=byte_mask)
            return predictions
        else:
            # Greedy decoding
            preds = torch.argmax(emissions, dim=-1) # (batch_size, seq_len)
            predictions = []
            for i in range(preds.size(0)):
                seq_len = int(mask[i].sum().item())
                predictions.append(preds[i, :seq_len].tolist())
            return predictions
