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
        # "only" is tricky because it modifies the subject or action, we don't want to capture the whole sentence.
        
    def _weak_label_constraint_span(self, doc: spacy.tokens.Doc) -> Tuple[int, int]:
        """Heuristically find the start and end token indices of a constraint clause."""
        best_start = -1
        best_end = -1
        
        # 1. Look for explicit condition markers (if, when, unless...)
        for token in doc:
            if token.lower_ in self.condition_markers and (token.dep_ == "mark" or token.dep_ == "advmod" or token.pos_ == "SCONJ"):
                # Use the subtree of the head of this marker, or the subtree of the marker itself
                head = token.head
                if head.pos_ in ("VERB", "AUX") and head != token:
                    subtree_indices = sorted([t.i for t in head.subtree])
                else:
                    subtree_indices = sorted([t.i for t in token.subtree])
                
                # Exclude main clause subject/object if they got dragged in
                # Actually, an advcl subtree is usually pretty self-contained
                start, end = subtree_indices[0], subtree_indices[-1]
                # Try to tighten the span: start at the marker itself
                if token.i >= start:
                    start = token.i
                return start, end
                
        # 2. Look for relative clauses (relcl) that act as constraints
        # E.g., "... documents belonging to tenants they are assigned responsible."
        # E.g., "... tasks that are assigned to them."
        for token in doc:
            if token.dep_ == "relcl":
                subtree_indices = sorted([t.i for t in token.subtree])
                return subtree_indices[0], subtree_indices[-1]
                
        # 3. Look for prepositional phrases acting as constraints
        # E.g., "... invoices sent by the department." (acl or prep)
        for token in doc:
            if token.dep_ == "acl" and token.head.dep_ in ("dobj", "pobj"):
                subtree_indices = sorted([t.i for t in token.subtree])
                return subtree_indices[0], subtree_indices[-1]

        # 4. Handle "Only" patterns specifically
        # E.g., "Only members of the sales department can send invoices."
        # Constraint is "of the sales department". The word "Only" just indicates a restriction.
        for token in doc:
            if token.lower_ == "only":
                # Look for a prepositional phrase modifying the subject
                head = token.head
                if head.dep_ == "nsubj":
                    for child in head.children:
                        if child.dep_ == "prep":
                            subtree_indices = sorted([t.i for t in child.subtree])
                            return subtree_indices[0], subtree_indices[-1]
        
        return best_start, best_end

    def generate_bio(self, rules: List[ABACPolicyRule]) -> List[List[Tuple[str, str]]]:
        """Generate BIO tagged sequences for a list of rules.
        Format: [[("A", "O"), ("user", "O"), ("can", "O"), ("read", "O"), ("if", "B-CONSTRAINT"), ...], ...]
        """
        dataset = []
        
        for rule in rules:
            if not rule.description:
                continue
                
            doc = self.nlp(rule.description)
            
            for sent in doc.sents:
                start_idx, end_idx = -1, -1
                
                # We only tag constraints if the rule actually has one in its syntax
                if rule.constraint_condition:
                    start_idx, end_idx = self._weak_label_constraint_span(sent.as_doc())
                    # Adjust relative indices from sent.as_doc() to original doc indices
                    # sent.as_doc() is an independent Doc, so indices are 0-based
                
                tagged_sentence = []
                for i, token in enumerate(sent):
                    label = "O"
                    if start_idx != -1 and start_idx <= i <= end_idx:
                        # Ignore trailing punctuation for constraint span
                        if token.is_punct and i == end_idx:
                            label = "O"
                        elif i == start_idx:
                            label = "B-CONSTRAINT"
                        else:
                            label = "I-CONSTRAINT"
                            
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
