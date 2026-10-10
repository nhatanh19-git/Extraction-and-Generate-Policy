"""BIO Annotation Generator for ABAC constraints.
Converts ABACPolicyRule items into CoNLL-format BIO training data using heuristic alignment.
"""

import logging
import spacy
from typing import List, Tuple, Optional
from .abac_parser import ABACPolicyRule

logger = logging.getLogger(__name__)


class BIOConstraintGenerator:
    """Generates BIO annotations from natural language policies."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Attempting to download...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            
        # Condition indicators for weak labeling
        self.condition_markers = {"if", "when", "unless", "provided", "where", "assuming"}
        
    def _weak_label_constraint_spans(self, doc: spacy.tokens.Doc) -> List[Tuple[int, int]]:
        """Heuristically find ALL constraint span (start, end) token indices in a sentence.
        
        Returns a list of (start, end) tuples, one per constraint span found.
        """
        spans = []
        
        # 1. Look for explicit condition markers (if, when, unless...)
        for token in doc:
            if token.lower_ in self.condition_markers and (token.dep_ == "mark" or token.dep_ == "advmod" or token.pos_ == "SCONJ"):
                head = token.head
                if head.pos_ in ("VERB", "AUX") and head != token:
                    subtree_indices = sorted([t.i for t in head.subtree])
                else:
                    subtree_indices = sorted([t.i for t in token.subtree])
                
                start, end = subtree_indices[0], subtree_indices[-1]
                if token.i >= start:
                    start = token.i
                spans.append((start, end))
                
        # 2. Look for relative clauses (relcl) that act as constraints
        for token in doc:
            if token.dep_ == "relcl":
                head = token.head
                # If relcl modifies a noun that is a pobj of a preposition, include the prep!
                # e.g. "for courses which they are taking"
                if head.dep_ == "pobj" and head.head.dep_ == "prep":
                    subtree_indices = sorted([t.i for t in head.head.subtree])
                else:
                    subtree_indices = sorted([t.i for t in token.subtree])
                spans.append((subtree_indices[0], subtree_indices[-1]))
                
        # 3. Look for participial phrases acting as constraints (acl)
        for token in doc:
            if token.dep_ == "acl" and token.head.dep_ in ("dobj", "pobj", "nsubj", "nsubjpass"):
                subtree_indices = sorted([t.i for t in token.subtree])
                spans.append((subtree_indices[0], subtree_indices[-1]))

        # 4. Handle "Only" patterns — prepositional phrase on the subject
        for token in doc:
            if token.lower_ == "only":
                head = token.head
                if head.dep_ == "nsubj":
                    for child in head.children:
                        if child.dep_ == "prep":
                            subtree_indices = sorted([t.i for t in child.subtree])
                            spans.append((subtree_indices[0], subtree_indices[-1]))
        
        # 5. Look for "with" prepositional phrases attached to the main verb
        # E.g., "... can modify grades with the approval of the head of ..."
        # These are environmental / contextual constraints.
        for token in doc:
            if token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ == "prep" and child.lower_ == "with":
                        subtree_indices = sorted([t.i for t in child.subtree])
                        spans.append((subtree_indices[0], subtree_indices[-1]))
        
        # Deduplicate and merge overlapping spans
        if not spans:
            return []
        
        spans.sort()
        merged = [spans[0]]
        for s, e in spans[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + 1:
                # Overlapping or adjacent — merge
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
                
        return merged

    # Keep the old single-span API for backward compatibility (used by pipeline heuristic fallback)
    def _weak_label_constraint_span(self, doc: spacy.tokens.Doc) -> Tuple[int, int]:
        """Return the first constraint span found (backward-compatible single-span API)."""
        spans = self._weak_label_constraint_spans(doc)
        if spans:
            return spans[0]
        return (-1, -1)

    def generate_bio(self, rules: List[ABACPolicyRule]) -> List[List[Tuple[str, str]]]:
        """Generate BIO tagged sequences for a list of rules.
        Format: [("A", "O"), ("user", "O"), ("can", "O"), ("read", "O"), ("if", "B-CONSTRAINT"), ...]
        """
        dataset = []
        
        for rule in rules:
            if not rule.description:
                continue
                
            doc = self.nlp(rule.description)
            
            for sent in doc.sents:
                constraint_ranges = []
                
                # We only tag constraints if the rule actually has one in its syntax
                if rule.constraint_condition:
                    constraint_ranges = self._weak_label_constraint_spans(sent.as_doc())
                
                tagged_sentence = []
                for i, token in enumerate(sent):
                    label = "O"
                    for start_idx, end_idx in constraint_ranges:
                        if start_idx <= i <= end_idx:
                            # Ignore trailing punctuation for constraint span
                            if token.is_punct and i == end_idx:
                                label = "O"
                            elif i == start_idx:
                                label = "B-CONSTRAINT"
                            else:
                                label = "I-CONSTRAINT"
                            break
                            
                    tagged_sentence.append((token.text, label))
                    
                dataset.append(tagged_sentence)
            
        return dataset

    def export_conll(self, dataset: List[List[Tuple[str, str]]], output_path: str):
        """Export the BIO dataset to a CoNLL format file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in dataset:
                for token, label in sentence:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
