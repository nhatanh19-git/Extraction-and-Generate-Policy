"""Embeddings Loader for BiLSTM.
Supports: spaCy vectors, FastText, GloVe, and random initialization.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from .vocabulary import Vocabulary
import logging

logger = logging.getLogger(__name__)


def create_embedding_layer(
    vocab: Vocabulary, 
    embedding_dim: int = 300, 
    fasttext_path: str = None,
    glove_path: str = None,
    spacy_nlp=None,
    freeze: bool = False
) -> nn.Embedding:
    """Creates an Embedding layer initialized with pretrained vectors if provided.
    
    Priority order:
    1. spaCy vectors (if spacy_nlp is provided and has vectors)
    2. FastText binary model
    3. GloVe text file
    4. Random Xavier initialization
    """
    
    vocab_size = vocab.vocab_size
    # Use Xavier/Glorot initialization scale for standard embeddings
    scale = 1.0 / np.sqrt(embedding_dim)
    weights = np.random.normal(scale=scale, size=(vocab_size, embedding_dim))
    
    # PAD token gets zero vector
    weights[vocab.get_word_idx(Vocabulary.PAD_TOKEN)] = np.zeros(embedding_dim)
    
    loaded_count = 0
    
    # 1. Try spaCy vectors first (already loaded in memory, no file I/O)
    if spacy_nlp is not None and spacy_nlp.vocab.vectors.shape[0] > 0:
        spacy_dim = spacy_nlp.vocab.vectors.shape[1]
        if spacy_dim != embedding_dim:
            logger.warning(
                f"spaCy vector dim ({spacy_dim}) != requested embedding_dim ({embedding_dim}). "
                f"Overriding embedding_dim to {spacy_dim}."
            )
            embedding_dim = spacy_dim
            weights = np.random.normal(scale=1.0/np.sqrt(embedding_dim), size=(vocab_size, embedding_dim))
            weights[vocab.get_word_idx(Vocabulary.PAD_TOKEN)] = np.zeros(embedding_dim)
        
        for idx in range(vocab_size):
            word = vocab.idx2word[idx]
            if word in (Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN):
                continue
            lexeme = spacy_nlp.vocab[word]
            if lexeme.has_vector:
                weights[idx] = lexeme.vector
                loaded_count += 1
                
        logger.info(f"Loaded {loaded_count}/{vocab_size - 2} vectors from spaCy model.")
    
    # 2. Try FastText binary model
    elif fasttext_path and os.path.exists(fasttext_path):
        try:
            import fasttext
            logger.info(f"Loading FastText model from {fasttext_path}")
            ft_model = fasttext.load_model(fasttext_path)
            
            if ft_model.get_dimension() != embedding_dim:
                logger.error(f"FastText dim {ft_model.get_dimension()} != requested {embedding_dim}")
            else:
                for idx in range(vocab_size):
                    word = vocab.idx2word[idx]
                    if word in (Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN):
                        continue
                    weights[idx] = ft_model.get_word_vector(word)
                    loaded_count += 1
                logger.info(f"Loaded {loaded_count} vectors from FastText.")
        except Exception as e:
            logger.error(f"Failed to load FastText model: {e}")
            
    # 3. Fallback to GloVe text file
    elif glove_path and os.path.exists(glove_path):
        try:
            logger.info(f"Loading GloVe model from {glove_path}")
            glove_dict = {}
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == embedding_dim + 1:
                        glove_dict[parts[0]] = np.array([float(x) for x in parts[1:]])
            
            for idx in range(vocab_size):
                word = vocab.idx2word[idx]
                if word in glove_dict:
                    weights[idx] = glove_dict[word]
                    loaded_count += 1
                elif word.lower() in glove_dict:
                    weights[idx] = glove_dict[word.lower()]
                    loaded_count += 1
                    
            logger.info(f"Loaded {loaded_count} vectors from GloVe.")
        except Exception as e:
            logger.error(f"Failed to load GloVe model: {e}")
            
    if loaded_count == 0:
        logger.warning("No pretrained embeddings loaded. Using random initialization.")
        
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab.get_word_idx(Vocabulary.PAD_TOKEN))
    embedding.weight.data.copy_(torch.from_numpy(weights).float())
    embedding.weight.requires_grad = not freeze
    return embedding
