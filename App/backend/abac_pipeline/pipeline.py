"""End-to-End ABAC Pipeline.

Orchestrates the full extraction flow:
    NLPPreprocessor → SAO Entity Extraction → Constraint Extraction
    → Semantic Analysis → Attribute Mapping → Rule Generation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import spacy
from spacy.tokens import Doc

from .core.abac_schemas import (
    ABACEntity, ConstraintSpan, StructuredExtractionResult,
    CanonicalCondition, ConstraintAnalysis
)
from .core.abac_rule import CanonicalABACRule
from .core.preprocessor import NLPPreprocessor, NLPResult
from .semantic.constraint_analyzer import ConstraintAnalyzer
from .mapping.attribute_mapper import AttributeMapper
from .data.bio_generator import BIOConstraintGenerator

# SAO components (dep-parse based extraction)
from backend.sao_extraction.components.subject_extraction import SubjectExtractor
from backend.sao_extraction.components.action_extraction import ActionExtractor
from backend.sao_extraction.components.object_extraction import ObjectExtractor
from backend.sao_extraction.components.clause_segmentation import ClauseSegmenter
from backend.sao_extraction.core.schemas import PipelineContext

logger = logging.getLogger(__name__)


def _find_span_in_doc(doc: Doc, text: str) -> Tuple[int, int]:
    """Find the token-level (start, end) span for a text substring in doc.
    
    Searches for the best matching contiguous token sequence.
    Returns (-1, -1) if not found.
    """
    if not text or not doc:
        return (-1, -1)
    
    text_lower = text.lower().strip()
    n = len(doc)
    
    # Try exact substring match on character offsets first
    char_start = doc.text.lower().find(text_lower)
    if char_start != -1:
        char_end = char_start + len(text_lower)
        # Find tokens that overlap this character range
        tok_start = -1
        tok_end = -1
        for i, tok in enumerate(doc):
            tok_char_start = tok.idx
            tok_char_end = tok.idx + len(tok.text)
            if tok_char_start >= char_start and tok_start == -1:
                tok_start = i
            if tok_char_end <= char_end:
                tok_end = i + 1
            elif tok_start != -1:
                break
        if tok_start != -1 and tok_end != -1:
            return (tok_start, tok_end)
    
    # Fallback: token-level sliding window
    text_tokens = text_lower.split()
    if not text_tokens:
        return (-1, -1)
    
    for i in range(n):
        if doc[i].text.lower() == text_tokens[0]:
            match = True
            end = i
            for j, tt in enumerate(text_tokens):
                if i + j >= n:
                    match = False
                    break
                if doc[i + j].text.lower() != tt:
                    match = False
                    break
                end = i + j + 1
            if match:
                return (i, end)
    
    return (-1, -1)


class ABACPipeline:
    """The End-to-End pipeline for extracting ABAC rules from text."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        logger.info(f"Initializing ABACPipeline with model: {spacy_model}")
        
        # Unified preprocessor (owns the spaCy model)
        self.preprocessor = NLPPreprocessor(spacy_model)
        self.nlp = self.preprocessor.get_spacy_nlp()
        
        # SAO dependency-based extractors
        self.clause_segmenter = ClauseSegmenter()
        self.subj_extractor = SubjectExtractor()
        self.action_extractor = ActionExtractor()
        self.obj_extractor = ObjectExtractor()
        
        # Constraint extraction (heuristic fallback — will be replaced by BiLSTM)
        self.constraint_heuristic = BIOConstraintGenerator(spacy_model)
        
        # Semantic analysis
        self.analyzer = ConstraintAnalyzer(self.nlp)
        
        # Attribute mapping
        schema_path = Path(__file__).parent / "config" / "abac_schema.yaml"
        self.mapper = AttributeMapper(str(schema_path))
        
        # BiLSTM model (None until trained model is loaded)
        self._bilstm_model = None
        self._bilstm_config = None
        
        # Try to load the BiLSTM model
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        if (checkpoints_dir / "best_bilstm.pt").exists() and (checkpoints_dir / "vocab.json").exists():
            try:
                from .bilstm import BiLSTMConfig, Vocabulary, BiLSTMConstraintModel
                import torch
                
                self._bilstm_config = BiLSTMConfig(use_crf=True)
                vocab = Vocabulary()
                vocab.load(str(checkpoints_dir / "vocab.json"))
                
                self._bilstm_model = BiLSTMConstraintModel(self._bilstm_config, vocab)
                self._bilstm_model.load_state_dict(torch.load(str(checkpoints_dir / "best_bilstm.pt"), map_location="cpu", weights_only=True))
                self._bilstm_model.eval()
                logger.info("Successfully loaded trained BiLSTM-CRF model for constraint extraction.")
            except Exception as e:
                logger.error(f"Failed to load BiLSTM model: {e}")
                self._bilstm_model = None
        
    def _extract_sao_entities(self, doc: Doc) -> List[ABACEntity]:
        """Extract Subject/Action/Object entities with proper span tracking.
        
        Uses the SAO dep-parse extractors and resolves character offsets
        back to token spans in the original document.
        """
        context = PipelineContext(original_text=doc.text, doc=doc)
        
        # Run SAO pipeline
        context = self.clause_segmenter.process(context)
        context = self.subj_extractor.process(context)
        context = self.action_extractor.process(context)
        context = self.obj_extractor.process(context)
        
        entities = []
        
        subjects = context.metadata.get("subjects", [])
        actions = context.metadata.get("actions", [])
        objects = context.metadata.get("objects", [])
        
        for i, clause in enumerate(context.clause_docs or [doc[:]]):
            # Subject
            if i < len(subjects) and subjects[i]:
                subj_text = subjects[i]
                span_start, span_end = _find_span_in_doc(doc, subj_text)
                entities.append(ABACEntity(
                    text=subj_text,
                    lemma=subj_text.lower(),
                    span_start=span_start,
                    span_end=span_end,
                    type="SUBJECT"
                ))
            
            # Action
            if i < len(actions) and actions[i]:
                action_text = actions[i]
                span_start, span_end = _find_span_in_doc(doc, action_text)
                entities.append(ABACEntity(
                    text=action_text,
                    lemma=action_text.lower(),
                    span_start=span_start,
                    span_end=span_end,
                    type="ACTION"
                ))
            
            # Objects
            if i < len(objects) and objects[i]:
                for obj_text in objects[i]:
                    span_start, span_end = _find_span_in_doc(doc, obj_text)
                    entities.append(ABACEntity(
                        text=obj_text,
                        lemma=obj_text.lower(),
                        span_start=span_start,
                        span_end=span_end,
                        type="OBJECT"
                    ))
                    
        return entities
        
    def _extract_constraints_heuristic(self, doc: Doc) -> List[ConstraintSpan]:
        """Extract constraint spans using the keyword heuristic (fallback).
        
        This will be replaced by BiLSTM inference once the model is trained.
        """
        start, end = self.constraint_heuristic._weak_label_constraint_span(doc)
        if start != -1 and end != -1:
            span_text = doc[start:end+1].text
            return [ConstraintSpan(text=span_text, start=start, end=end)]
        return []
    
    def _extract_constraints(self, doc: Doc) -> List[ConstraintSpan]:
        """Extract constraint spans (BiLSTM if available, else heuristic)."""
        if self._bilstm_model is not None:
            import torch
            from .bilstm import Vocabulary
            
            # Prepare input
            vocab = self._bilstm_model.vocab
            words = [vocab.get_word_idx(t.text) for t in doc]
            
            x = torch.tensor([words], dtype=torch.long)
            mask = torch.ones((1, len(words)), dtype=torch.bool)
            
            # Predict
            with torch.no_grad():
                preds = self._bilstm_model.predict(x, mask)[0]
                
            # Convert tags to spans
            spans = []
            current_span_start = -1
            
            for i, tag_idx in enumerate(preds):
                tag = vocab.idx2tag.get(tag_idx, "O")
                
                if tag.startswith("B-"):
                    if current_span_start != -1:
                        # Close previous span
                        span_text = doc[current_span_start:i].text
                        spans.append(ConstraintSpan(text=span_text, start=current_span_start, end=i-1))
                    current_span_start = i
                elif tag.startswith("I-"):
                    if current_span_start == -1:
                        # Invalid transition, but we'll accept it and start a new span
                        current_span_start = i
                else:
                    if current_span_start != -1:
                        # Close span
                        span_text = doc[current_span_start:i].text
                        spans.append(ConstraintSpan(text=span_text, start=current_span_start, end=i-1))
                        current_span_start = -1
                        
            # Handle span at the end
            if current_span_start != -1:
                span_text = doc[current_span_start:].text
                spans.append(ConstraintSpan(text=span_text, start=current_span_start, end=len(doc)-1))
                
            if spans:
                return spans
                
            # Fall back to heuristic if BiLSTM predicted nothing
            
        return self._extract_constraints_heuristic(doc)

    def _detect_effect(self, text: str, doc: Doc) -> str:
        """Detect whether the policy is Permit or Deny.
        
        Looks for negation patterns and deny keywords in the text.
        """
        text_lower = text.lower()
        
        # Explicit deny keywords
        deny_keywords = {
            "deny", "denied", "prohibit", "prohibited", "forbid", "forbidden",
            "restrict", "restricted", "prevent", "prevented", "block", "blocked",
            "shall not", "must not", "cannot", "may not", "is not allowed",
            "is not permitted", "are not allowed", "are not permitted",
        }
        
        for keyword in deny_keywords:
            if keyword in text_lower:
                return "Deny"
        
        # Check for negation on the main verb using dep-parse
        for token in doc:
            if token.dep_ == "ROOT" or token.dep_ == "xcomp":
                for child in token.children:
                    if child.dep_ == "neg":
                        return "Deny"
        
        return "Permit"

    def process_policy(self, text: str, rule_id: str = "rule_1") -> StructuredExtractionResult:
        """Process a policy sentence and output a structured rule.
        
        Pipeline:
            1. NLP Preprocessing (contraction expansion, spaCy parse)
            2. SAO Entity Extraction (dep-parse with span tracking)
            3. Constraint Span Extraction (heuristic / BiLSTM)
            4. Constraint Semantic Analysis
            5. Attribute Mapping
            6. Effect Detection
            7. Rule Assembly
        """
        # 1. Preprocess
        nlp_result = self.preprocessor.process(text)
        doc = nlp_result.doc
        
        # 2. Extract base entities (with real spans)
        entities = self._extract_sao_entities(doc)
        
        # 3. Extract constraint spans
        constraint_spans = self._extract_constraints(doc)
        
        # 4. Analyze constraints
        canonical_constraints = []
        for span in constraint_spans:
            raw_cond = self.analyzer.analyze(span, entities)
            if raw_cond:
                # 5. Map attributes
                mapped_cond = self.mapper.map_condition(raw_cond)
                canonical_constraints.append(mapped_cond)
                
        # 6. Build Canonical Rule
        subjects = []
        resources = []
        actions = []
        
        for ent in entities:
            if ent.type == "SUBJECT":
                mapped_attr = self.mapper._find_best_match(ent.lemma, self.mapper.subject_vocab)
                # Determine the appropriate attribute name via semantic analysis
                attr_name = self._infer_subject_attribute(ent, doc)
                subjects.append(CanonicalCondition(
                    attribute=attr_name,
                    operator="IN",
                    values=[mapped_attr]
                ))
            elif ent.type == "OBJECT":
                mapped_attr = self.mapper._find_best_match(ent.lemma, self.mapper.resource_vocab)
                attr_name = self._infer_resource_attribute(ent, doc)
                resources.append(CanonicalCondition(
                    attribute=attr_name,
                    operator="IN",
                    values=[mapped_attr]
                ))
            elif ent.type == "ACTION":
                mapped_action = self.mapper._find_best_match(ent.lemma, self.mapper.action_vocab)
                actions.append(mapped_action)
        
        # 7. Detect effect
        effect = self._detect_effect(text, doc)
        
        rule = CanonicalABACRule(
            rule_id=rule_id,
            effect=effect,
            subjects=subjects,
            resources=resources,
            actions=actions,
            constraints=canonical_constraints
        )
        
        # Group entities by type for the result
        entity_dict: Dict[str, List[ABACEntity]] = {}
        for ent in entities:
            entity_dict.setdefault(ent.type, []).append(ent)
        
        result = StructuredExtractionResult(
            id=rule_id,
            original_sentence=text,
            entities=entity_dict,
            constraint_spans=constraint_spans,
            rule=rule
        )
        
        return result
    
    def _infer_subject_attribute(self, entity: ABACEntity, doc: Doc) -> str:
        """Infer which subject attribute this entity refers to.
        
        Uses context clues from the surrounding text to determine if the 
        subject maps to 'role', 'position', 'department', etc.
        """
        text_lower = entity.text.lower()
        
        # Position indicators
        position_terms = {
            "nurse", "doctor", "manager", "director", "secretary",
            "accountant", "auditor", "planner", "technician", "operator",
            "faculty", "staff", "student", "applicant", "leader",
            "agent", "patient", "instructor", "chair",
        }
        
        # Department indicators
        department_indicators = {"department", "office", "division", "unit"}
        
        # Check if entity text contains position terms
        for term in position_terms:
            if term in text_lower:
                return "position"
        
        # Check surrounding context for "department" mentions
        for term in department_indicators:
            if term in text_lower:
                return "department"
        
        # Check for "member of" pattern which often indicates role
        if "member" in text_lower:
            return "role"
        
        # Default to role (most common in ABAC)
        return "role"
    
    def _infer_resource_attribute(self, entity: ABACEntity, doc: Doc) -> str:
        """Infer which resource attribute this entity refers to.
        
        Uses context clues from the surrounding text to determine if the 
        resource maps to 'type', 'department', 'classification', etc.
        """
        text_lower = entity.text.lower()
        
        # Resource type indicators
        type_terms = {
            "record", "file", "document", "report", "budget", "schedule",
            "task", "invoice", "contract", "paycheck", "application",
            "gradebook", "roster", "transcript", "item", "offer",
        }
        
        for term in type_terms:
            if term in text_lower:
                return "type"
        
        # Default to type (most common for resources)
        return "type"
