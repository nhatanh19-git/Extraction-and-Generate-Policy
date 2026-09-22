"""PyTorch Dataset for BIO Tagging."""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from .vocabulary import Vocabulary


class BIODataset(Dataset):
    """PyTorch Dataset for BIO Sequence Labeling."""
    
    def __init__(self, sentences: List[List[Tuple[str, str]]], vocab: Vocabulary):
        self.sentences = sentences
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        
        word_indices = [self.vocab.get_word_idx(word) for word, tag in sentence]
        tag_indices = [self.vocab.get_tag_idx(tag) for word, tag in sentence]
        
        return (
            torch.tensor(word_indices, dtype=torch.long),
            torch.tensor(tag_indices, dtype=torch.long)
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], vocab: Vocabulary) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad a batch of sequences to the length of the longest sequence in the batch."""
    # batch is a list of tuples (word_tensor, tag_tensor)
    words = [item[0] for item in batch]
    tags = [item[1] for item in batch]
    
    # Pad sequences
    pad_word_idx = vocab.get_word_idx(Vocabulary.PAD_TOKEN)
    pad_tag_idx = vocab.get_tag_idx(Vocabulary.PAD_TOKEN)
    
    padded_words = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=pad_word_idx)
    padded_tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=pad_tag_idx)
    
    # Create mask (1/True for real tokens, 0/False for padding)
    # Using bool for mask is standard in modern PyTorch, especially for CRF
    mask = (padded_words != pad_word_idx)
    
    return padded_words, padded_tags, mask


def load_conll(filepath: str) -> List[List[Tuple[str, str]]]:
    """Load CoNLL dataset into list of sentences."""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_sentence.append((parts[0], parts[-1]))
                    
    if current_sentence:
        sentences.append(current_sentence)
        
    return sentences
