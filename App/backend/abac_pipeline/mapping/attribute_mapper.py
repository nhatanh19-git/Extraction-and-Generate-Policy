"""Attribute Mapper.
Maps raw extracted text attributes to canonical ABAC attributes using semantic candidate ranking.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from ..core.abac_schemas import CanonicalCondition
from thefuzz import fuzz # type: ignore
import spacy
import logging

logger = logging.getLogger(__name__)

class AttributeMapper:
    """Maps raw text to canonical attribute names defined in the schema."""
    
    def __init__(self, schema_path: str, nlp: Optional[spacy.Language] = None):
        self.schema = self._load_schema(schema_path)
        self.subject_vocab = self.schema.get("subject_attributes", [])
        self.resource_vocab = self.schema.get("resource_attributes", [])
        self.action_vocab = self.schema.get("action_attributes", [])
        self.nlp = nlp
        
    def _load_schema(self, schema_path: str) -> Dict[str, List[str]]:
        path = Path(schema_path)
        if not path.exists():
            logger.warning(f"Schema file {schema_path} not found. Using empty vocab.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
            
    def get_candidates(self, raw_text: str, vocab: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Generate and rank candidate mappings based on string and semantic similarity.
        
        Returns:
            List of (attribute_name, confidence_score) sorted by score descending.
        """
        if not raw_text or not vocab:
            return [(raw_text, 1.0)]
            
        candidates = []
        raw_lower = raw_text.lower()
        
        raw_doc = self.nlp(raw_lower) if self.nlp else None
        
        for attr in vocab:
            attr_lower = attr.lower()
            
            # 1. Exact match (highest confidence)
            if raw_lower == attr_lower:
                candidates.append((attr, 1.0))
                continue
                
            # 2. Fuzzy string similarity (0.0 to 1.0)
            try:
                fuzzy_score = fuzz.token_sort_ratio(raw_lower, attr_lower) / 100.0
            except NameError:
                # thefuzz not installed fallback
                fuzzy_score = 1.0 if attr_lower in raw_lower or raw_lower in attr_lower else 0.0
                
            # 3. Semantic similarity (if spaCy model with vectors is loaded)
            semantic_score = 0.0
            if raw_doc and raw_doc.has_vector:
                attr_doc = self.nlp(attr_lower)
                if attr_doc.has_vector:
                    semantic_score = raw_doc.similarity(attr_doc)
                    
            # Combine scores (weight fuzzy higher for structural ABAC names)
            final_score = (fuzzy_score * 0.7) + (semantic_score * 0.3)
            candidates.append((attr, final_score))
            
        # Sort by score descending and take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _find_best_match(self, raw_text: str, vocab: List[str], threshold: float = 0.6) -> str:
        """Find the single best matching attribute."""
        candidates = self.get_candidates(raw_text, vocab, top_k=1)
        if candidates and candidates[0][1] >= threshold:
            return candidates[0][0]
        return raw_text # Return raw if no good match

    def map_condition(self, condition: CanonicalCondition, is_subject: bool = True) -> CanonicalCondition:
        """Maps attributes in a condition to canonical schema."""
        
        # In ABAC, constraint conditions are often evaluated against the context or the opposite entity.
        # But by default, if it's modifying a subject, we map its attribute against the subject vocab.
        vocab = self.subject_vocab if is_subject else self.resource_vocab
        
        # If the left operand clearly refers to the resource (e.g. "type", "classification"), use resource vocab
        if any(res_attr in condition.attribute.lower() for res_attr in self.resource_vocab):
             vocab = self.resource_vocab
             
        canonical_attr = self._find_best_match(condition.attribute, vocab)
        
        canonical_ref = None
        if condition.attribute_ref:
            # If the reference is something like "resource's department", map just "department"
            ref_clean = condition.attribute_ref.lower().replace("resource's", "").replace("subject's", "").replace("'s", "").strip()
            
            # Determine which vocab to use based on the reference target
            ref_vocab = self.resource_vocab if "resource" in condition.attribute_ref.lower() or "object" in condition.attribute_ref.lower() else self.subject_vocab
            mapped_ref = self._find_best_match(ref_clean, ref_vocab)
            
            # Reconstruct reference type
            prefix = "resource" if "resource" in condition.attribute_ref.lower() or "object" in condition.attribute_ref.lower() else "subject"
            canonical_ref = f"{prefix}.{mapped_ref}"
            
        return CanonicalCondition(
            attribute=canonical_attr,
            operator=condition.operator,
            values=condition.values,
            attribute_ref=canonical_ref
        )
