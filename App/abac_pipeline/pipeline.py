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
                if span_start != -1:
                    entities.append(ABACEntity(
                        text=subj_text,
                        lemma=subj_text.lower(),
                        span_start=span_start,
                        span_end=span_end - 1,
                        type="SUBJECT"
                    ))
            
            # Action
            if i < len(actions) and actions[i]:
                action_text = actions[i]
                span_start, span_end = _find_span_in_doc(doc, action_text)
                if span_start != -1:
                    entities.append(ABACEntity(
                        text=action_text,
                        lemma=action_text.lower(),
                        span_start=span_start,
                        span_end=span_end - 1,
                        type="ACTION"
                    ))
            
            # Objects
            if i < len(objects) and objects[i]:
                for obj_text in objects[i]:
                    span_start, span_end = _find_span_in_doc(doc, obj_text)
                    if span_start != -1:
                        entities.append(ABACEntity(
                            text=obj_text,
                            lemma=obj_text.lower(),
                            span_start=span_start,
                            span_end=span_end - 1,
                            type="OBJECT"
                        ))
                    
        return entities
        
    def _extract_constraints_heuristic(self, doc: Doc) -> List[ConstraintSpan]:
        """Extract constraint spans using the keyword heuristic (fallback).
        
        Uses multi-span API to detect all constraint spans in the sentence.
        """
        spans_ranges = self.constraint_heuristic._weak_label_constraint_spans(doc)
        result = []
        for start, end in spans_ranges:
            span_text = doc[start:end+1].text
            result.append(ConstraintSpan(text=span_text, start=start, end=end))
        return result
    
    def _merge_adjacent_spans(self, spans: List[ConstraintSpan], doc: Doc, max_gap: int = 2) -> List[ConstraintSpan]:
        """Merge constraint spans that are adjacent or separated by only a small gap 
        (e.g. conjunction 'and' splitting a single semantic constraint).
        
        Also deduplicates spans that are fully contained within another.
        """
        if len(spans) <= 1:
            return spans
            
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda s: s.start)
        
        merged = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            prev = merged[-1]
            gap = span.start - prev.end - 1
            
            # Check if the gap tokens are just conjunctions/punctuation
            if gap <= max_gap and gap >= 0:
                gap_tokens = doc[prev.end + 1 : span.start]
                gap_text = [t.text.lower() for t in gap_tokens]
                # Merge if gap is only conjunctions, commas, or empty
                if all(t in ("and", "or", ",", "&") for t in gap_text) or len(gap_text) == 0:
                    new_start = prev.start
                    new_end = span.end
                    new_text = doc[new_start:new_end+1].text
                    merged[-1] = ConstraintSpan(text=new_text, start=new_start, end=new_end)
                    continue
            
            # Check if span is fully contained within prev
            if span.start >= prev.start and span.end <= prev.end:
                continue
                
            merged.append(span)
            
        return merged
    
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
                last_end = len(doc) - 1
                # Strip trailing punctuation
                while last_end > current_span_start and doc[last_end].is_punct:
                    last_end -= 1
                span_text = doc[current_span_start:last_end+1].text
                spans.append(ConstraintSpan(text=span_text, start=current_span_start, end=last_end))
                
            if spans:
                # Merge adjacent spans that were split by conjunctions
                spans = self._merge_adjacent_spans(spans, doc)
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
        
        # Remove constraint spans that are exactly an entire entity (false positives from BiLSTM)
        constraint_spans = [
            c for c in constraint_spans 
            if not any(c.start == ent.span_start and c.end == ent.span_end for ent in entities)
        ]
        
        # 3b. Dynamically extract structural qualifiers (prepositional phrases attached to Subject/Object)
        # that were excluded from the base entity extraction.
        # Also classify existing constraint spans.
        subject_qualifier_spans = []
        object_qualifier_spans = []
        true_constraint_spans = []
        
        # Helper to extract prep subtrees from entity roots
        def _extract_prep_qualifiers(ent: ABACEntity) -> List[ConstraintSpan]:
            qualifiers = []
            if ent.span_start != -1 and ent.span_end != -1:
                ent_span = doc[ent.span_start:ent.span_end + 1]
                ent_root = ent_span.root
                for child in ent_root.children:
                    if child.dep_ == "prep" and child.i > ent.span_end:
                        subtree = sorted([t.i for t in child.subtree])
                        qualifiers.append(ConstraintSpan(
                            text=doc[subtree[0]:subtree[-1]+1].text,
                            start=subtree[0],
                            end=subtree[-1]
                        ))
            return qualifiers
            
        for ent in entities:
            if ent.type == "SUBJECT":
                subject_qualifier_spans.extend(_extract_prep_qualifiers(ent))
            elif ent.type == "OBJECT":
                object_qualifier_spans.extend(_extract_prep_qualifiers(ent))
        
        for c_span in constraint_spans:
            is_subject_qualifier = False
            is_object_qualifier = False
            # Check if the constraint span's head token (in dep tree) is part of an entity
            if c_span.start < len(doc):
                span_head = doc[c_span.start]  # typically "of", "in", etc.
                if span_head.dep_ == "prep" and span_head.head is not None:
                    head_idx = span_head.head.i
                    for ent in entities:
                        if ent.type == "SUBJECT" and ent.span_start <= head_idx <= ent.span_end:
                            is_subject_qualifier = True
                            break
                        elif ent.type == "OBJECT" and ent.span_start <= head_idx <= ent.span_end:
                            is_object_qualifier = True
                            break
            
            if is_subject_qualifier:
                # Only add if not already dynamically extracted
                if not any(s.start == c_span.start for s in subject_qualifier_spans):
                    subject_qualifier_spans.append(c_span)
            elif is_object_qualifier:
                if not any(s.start == c_span.start for s in object_qualifier_spans):
                    object_qualifier_spans.append(c_span)
            else:
                true_constraint_spans.append(c_span)
        
        # 4. Analyze only TRUE constraints
        canonical_constraints = []
        for span in true_constraint_spans:
            raw_cond = self.analyzer.analyze(span, doc, entities)
            if raw_cond:
                # Resolve if left operand overlaps with Subject or Resource
                raw_cond = self._resolve_constraint_attribute(raw_cond, entities, doc)
                
                # 5. Map attributes
                mapped_cond = self.mapper.map_condition(raw_cond)
                
                # Filter out empty/useless constraints (no values AND no attribute_ref)
                if mapped_cond.values or mapped_cond.attribute_ref:
                    canonical_constraints.append(mapped_cond)
                
        # 6. Build Canonical Rule
        subjects = []
        objects = []
        actions = []
        
        import dataclasses as dc
        
        for ent in entities:
            if ent.type == "SUBJECT":
                # Strip "Only/only" from subject text
                subj_text = ent.lemma
                if subj_text.lower().startswith("only "):
                    subj_text = subj_text[5:].strip()
                
                mapped_attr = self.mapper._find_best_match(subj_text, self.mapper.subject_vocab)
                attr_name = self._infer_subject_attribute(ent, doc)
                subjects.append(CanonicalCondition(
                    attribute=attr_name,
                    operator="IN",
                    values=[mapped_attr]
                ))
                
                # Add subject qualifier spans to constraints
                for sq_span in subject_qualifier_spans:
                    sq_cond = self.analyzer.analyze(sq_span, doc, entities)
                    if sq_cond and sq_cond.values:
                        # Infer the attribute name from the values (e.g. "sales department" -> department)
                        val_text = " ".join(sq_cond.values)
                        inferred = self._infer_attribute_from_text(val_text, doc)
                        canonical_constraints.append(CanonicalCondition(
                            attribute=f"subject.{inferred}",
                            operator="IN",
                            values=sq_cond.values
                        ))
                        
            elif ent.type == "OBJECT":
                mapped_attr = self.mapper._find_best_match(ent.lemma, self.mapper.resource_vocab)
                attr_name = self._infer_object_attribute(ent, doc)
                objects.append(CanonicalCondition(
                    attribute=attr_name,
                    operator="IN",
                    values=[mapped_attr]
                ))
                
                # Add object qualifier spans to constraints
                for oq_span in object_qualifier_spans:
                    oq_cond = self.analyzer.analyze(oq_span, doc, entities)
                    if oq_cond and oq_cond.values:
                        val_text = " ".join(oq_cond.values)
                        inferred = self._infer_attribute_from_text(val_text, doc)
                        canonical_constraints.append(CanonicalCondition(
                            attribute=f"object.{inferred}",
                            operator="IN",
                            values=oq_cond.values
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
            objects=objects,
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
    
    def _truncate_entities_by_constraints(self, entities: List[ABACEntity], constraint_spans: List[ConstraintSpan], doc: Doc):
        """Truncate entity texts to exclude constraint spans to prevent overlapping."""
        import dataclasses
        for i, ent in enumerate(entities):
            if ent.type not in ("SUBJECT", "OBJECT"):
                continue
                
            for c_span in constraint_spans:
                # If constraint overlaps with entity
                if c_span.start >= ent.span_start and c_span.end <= ent.span_end:
                    # Usually constraints are at the end of the entity (e.g. "rosters [for courses...]")
                    if c_span.start > ent.span_start:
                        new_end = c_span.start - 1
                        new_text = doc[ent.span_start:new_end + 1].text.strip()
                        entities[i] = dataclasses.replace(ent, span_end=new_end, text=new_text, lemma=new_text.lower())
                        ent = entities[i]
                    else:
                        new_start = c_span.end + 1
                        if new_start <= ent.span_end:
                            new_text = doc[new_start:ent.span_end + 1].text.strip()
                            entities[i] = dataclasses.replace(ent, span_start=new_start, text=new_text, lemma=new_text.lower())
                            ent = entities[i]
    
    def _resolve_constraint_attribute(self, cond: CanonicalCondition, entities: List[ABACEntity], doc: Doc) -> CanonicalCondition:
        """Resolves the left operand of a constraint to a fully-qualified ABAC attribute 
        (e.g., subject.department, resource.classification) if it overlaps with an entity.
        """
        import dataclasses
        attr_text = cond.attribute.lower()
        
        is_subject = False
        is_object = False
        is_action = False
        
        for ent in entities:
            # Check overlap
            if ent.type == "SUBJECT" and (ent.lemma in attr_text or attr_text in ent.lemma):
                is_subject = True
            elif ent.type == "OBJECT" and (ent.lemma in attr_text or attr_text in ent.lemma):
                is_object = True
            elif ent.type == "ACTION" and (ent.lemma in attr_text or attr_text in ent.lemma):
                is_action = True
                
        if is_subject:
            # Infer what property of the subject this constraint is about based on the right operand
            right_text = " ".join(cond.values) if cond.values else ""
            dummy_ent = ABACEntity(text=right_text, lemma=right_text, span_start=0, span_end=0, type="SUBJECT")
            inferred_attr = self._infer_subject_attribute(dummy_ent, doc)
            cond = dataclasses.replace(cond, attribute=f"subject.{inferred_attr}")
            
        elif is_object:
            right_text = " ".join(cond.values) if cond.values else ""
            
            # Additional heuristic for resources: "confidential", "secret", "public" -> classification
            if any(w in right_text for w in ["confidential", "secret", "public", "restricted"]):
                inferred_attr = "classification"
            elif any(w in right_text for w in ["active", "pending", "approved", "denied", "assigned"]):
                inferred_attr = "status"
            else:
                dummy_ent = ABACEntity(text=right_text, lemma=right_text, span_start=0, span_end=0, type="OBJECT")
                inferred_attr = self._infer_object_attribute(dummy_ent, doc)
            
            cond = dataclasses.replace(cond, attribute=f"object.{inferred_attr}")
            
        elif is_action:
            # If it overlaps with an ACTION, the constraint is usually an environmental or contextual condition 
            # attached to the verb (e.g. 'with the approval of...').
            # The actual attribute is the main noun of the right operand (e.g. 'approval').
            right_text = " ".join(cond.values) if cond.values else ""
            if right_text:
                dummy_doc = self.analyzer.nlp(right_text)
                main_noun = None
                for token in dummy_doc:
                    if token.dep_ == "ROOT":
                        main_noun = token.lemma_
                        break
                if main_noun:
                    cond = dataclasses.replace(cond, attribute=f"environment.{main_noun}")
            
        return cond
    
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
    
    def _infer_object_attribute(self, entity: ABACEntity, doc: Doc) -> str:
        """Infer which object attribute this entity refers to.
        
        Uses context clues from the surrounding text to determine if the 
        object maps to 'type', 'department', 'classification', etc.
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
    
    def _infer_attribute_from_text(self, text: str, doc: Doc) -> str:
        """Infer an ABAC attribute name from a descriptive text.
        
        E.g. "the educational testing and quality assurance office" -> "department"
             "the sales department" -> "department"
             "engineering team" -> "team"
             "security clearance level" -> "clearance"
        """
        text_lower = text.lower()
        
        # Department/organizational indicators
        dept_indicators = {"department", "office", "division", "unit", "bureau", "branch", "section", "group"}
        for term in dept_indicators:
            if term in text_lower:
                return "department"
        
        # Team indicators
        if "team" in text_lower:
            return "team"
            
        # Level/clearance indicators 
        level_indicators = {"level", "clearance", "classification", "grade", "rank"}
        for term in level_indicators:
            if term in text_lower:
                return term
        
        # Location indicators
        location_indicators = {"ward", "location", "building", "floor", "room", "site", "area", "region"}
        for term in location_indicators:
            if term in text_lower:
                return "location"
        
        # Use spaCy to find the main noun (head of the NP)
        val_doc = self.preprocessor.nlp(text)
        for token in val_doc:
            if token.dep_ == "ROOT" or (token.pos_ == "NOUN" and token.dep_ in ("pobj", "attr", "dobj")):
                return token.lemma_
        
        # Fallback
        return "attribute"
