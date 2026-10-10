"""Vocabulary manager for BiLSTM Model."""

import json
from typing import Dict


class Vocabulary:
    """Manages mapping between words/tags and indices."""
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self):
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        
        self.tag2idx: Dict[str, int] = {self.PAD_TOKEN: 0}
        self.idx2tag: Dict[int, str] = {0: self.PAD_TOKEN}
        
    def add_word(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def add_tag(self, tag: str):
        if tag not in self.tag2idx:
            idx = len(self.tag2idx)
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag
            
    def get_word_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
        
    def get_tag_idx(self, tag: str) -> int:
        if tag not in self.tag2idx:
            # For inference, default unknown tags to "O" if present, else fallback
            if "O" in self.tag2idx:
                return self.tag2idx["O"]
            raise ValueError(f"Unknown tag '{tag}' encountered during lookup.")
        return self.tag2idx[tag]
        
    def build_from_conll(self, filepath: str):
        """Build vocabulary from a CoNLL dataset file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on any whitespace to handle both spaces and tabs
                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]  # Support multi-word tokens in broken CoNLL, take last as tag
                    self.add_word(word)
                    self.add_tag(tag)
                    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
        
    @property
    def num_tags(self) -> int:
        return len(self.tag2idx)

    def save(self, filepath: str):
        """Save vocabulary to a JSON file."""
        data = {
            "word2idx": self.word2idx,
            "tag2idx": self.tag2idx
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Load vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.word2idx = data["word2idx"]
        self.tag2idx = data["tag2idx"]
        
        # Reconstruct reverse mappings
        self.idx2word = {int(k): v for v, k in self.word2idx.items()}
        self.idx2tag = {int(k): v for v, k in self.tag2idx.items()}
