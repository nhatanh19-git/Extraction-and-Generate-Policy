"""Training script for the BiLSTM-CRF model."""

import os
import sys
import argparse
import logging
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
import spacy

# Ensure backend module can be imported when running standalone
base_dir = Path(__file__).resolve().parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from backend.abac_pipeline.bilstm import (
    BiLSTMConfig,
    Vocabulary,
    BIODataset,
    load_conll,
    collate_fn,
    BiLSTMConstraintModel,
    BiLSTMTrainer,
    set_seed
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF model for ABAC constraints.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (small for 60 sentences)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_md", help="spaCy model for embeddings (md/lg have vectors)")
    parser.add_argument("--fasttext", type=str, default=None, help="Path to FastText .bin file")
    parser.add_argument("--glove", type=str, default=None, help="Path to GloVe .txt file")
    parser.add_argument("--no_crf", action="store_true", help="Disable CRF layer")
    args = parser.parse_args()

    # 1. Configuration
    dataset_dir = base_dir / "backend" / "abac_pipeline" / "dataset"
    checkpoint_dir = base_dir / "backend" / "abac_pipeline" / "checkpoints"
    
    config = BiLSTMConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        use_crf=not args.no_crf,
        fasttext_model_path=args.fasttext,
        glove_model_path=args.glove,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 2. Load spaCy model for embeddings
    spacy_nlp = None
    try:
        spacy_nlp = spacy.load(args.spacy_model)
        vec_shape = spacy_nlp.vocab.vectors.shape
        logger.info(f"Loaded spaCy model '{args.spacy_model}' with {vec_shape[0]} vectors of dim {vec_shape[1]}")
    except OSError:
        logger.warning(f"spaCy model '{args.spacy_model}' not found. Trying to download...")
        try:
            spacy.cli.download(args.spacy_model)
            spacy_nlp = spacy.load(args.spacy_model)
            vec_shape = spacy_nlp.vocab.vectors.shape
            logger.info(f"Downloaded and loaded spaCy '{args.spacy_model}' with {vec_shape[0]} vectors of dim {vec_shape[1]}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model for embeddings: {e}. Falling back to random init.")
    
    # 3. Load Data
    train_path = dataset_dir / "train.conll"
    val_path = dataset_dir / "val.conll"
    test_path = dataset_dir / "test.conll"
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}. Run generate_dataset.py first.")
        sys.exit(1)
        
    logger.info("Loading CoNLL datasets...")
    train_sentences = load_conll(str(train_path))
    val_sentences = load_conll(str(val_path))
    test_sentences = load_conll(str(test_path)) if test_path.exists() else []
    
    logger.info(f"Train: {len(train_sentences)} sentences, Val: {len(val_sentences)} sentences, Test: {len(test_sentences)} sentences")
    
    # 4. Build Vocabulary (from train set ONLY to prevent data leakage)
    vocab = Vocabulary()
    logger.info("Building vocabulary from training data...")
    vocab.build_from_conll(str(train_path))
    
    # Also add val/test words so OOV doesn't crash during evaluation
    # (only word2idx, not tag2idx - tags come from training only)
    for split_path in [val_path, test_path]:
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            vocab.add_word(parts[0])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = checkpoint_dir / "vocab.json"
    vocab.save(str(vocab_path))
    logger.info(f"Vocabulary saved to {vocab_path} (size: {vocab.vocab_size} words, {vocab.num_tags} tags)")
    
    # 5. Create Datasets and DataLoaders
    train_dataset = BIODataset(train_sentences, vocab)
    val_dataset = BIODataset(val_sentences, vocab)
    
    # Use partial to pass vocab to collate_fn
    collate = partial(collate_fn, vocab=vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    # 6. Initialize Model with spaCy embeddings
    logger.info("Initializing BiLSTM Constraint Model...")
    model = BiLSTMConstraintModel(config, vocab, spacy_nlp=spacy_nlp)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    
    # 7. Train
    logger.info("Starting training...")
    trainer = BiLSTMTrainer(model, config, device=device)
    metrics = trainer.train(train_loader, val_loader)
    
    logger.info(f"Training completed. Best Val F1: {metrics.get('best_val_f1', 0):.4f}")
    
    # 8. Evaluate on Test Set
    if test_sentences:
        logger.info("Evaluating on test set...")
        test_dataset = BIODataset(test_sentences, vocab)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
        test_metrics = trainer.evaluate(test_loader)
        logger.info(
            f"Test Metrics - Loss: {test_metrics['loss']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
