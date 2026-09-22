"""Constraint Semantic Analyzer.
Parses natural language constraint spans into structured relationships.
"""

import spacy
from typing import List, Optional, Tuple
from ..core.abac_schemas import ConstraintSpan, CanonicalCondition, ABACEntity
from ..core.normalizer import normalize_operator, CanonicalOperator
from .reference_resolver import ReferenceResolver

class ConstraintAnalyzer:
    """Analyzes a constraint span to extract the semantic condition."""
    
    def __init__(self, nlp: spacy.Language):
        self.nlp = nlp
        self.resolver = ReferenceResolver()
        
    def _find_shortest_dependency_path(self, doc: spacy.tokens.Doc, start_token: spacy.tokens.Token, end_token: spacy.tokens.Token) -> List[spacy.tokens.Token]:
        """Find the shortest dependency path between two tokens in a document."""
        # Simple BFS
        queue = [[start_token]]
        visited = set([start_token])
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if current == end_token:
                return path
                
            # Get neighbors (head + children)
            neighbors = [current.head] + list(current.children)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in doc:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
        return []

    def _extract_arguments(self, doc: spacy.tokens.Doc, root: spacy.tokens.Token) -> Tuple[Optional[str], Optional[str]]:
        """Extract left and right arguments using dependency paths from the root verb."""
        left_operand = None
        right_operand = None
        
        # Find left operand (typically nsubj or nsubjpass)
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                left_operand = child.lemma_
                # Expand NP with modifiers
                left_parts = [t.lemma_ for t in child.subtree if t.dep_ in ("compound", "amod") and t.i < child.i]
                if left_parts:
                    left_operand = " ".join(left_parts) + " " + left_operand
                break
                
        # If no subject is found, check if it's a relative clause where the subject is implicit
        if not left_operand and root.dep_ == "relcl":
            left_operand = root.head.lemma_
                
        # Find right operand
        # Check direct objects and attributes
        for child in root.children:
            if child.dep_ in ("dobj", "attr", "acomp"):
                right_operand = child.lemma_
                right_parts = [t.lemma_ for t in child.subtree if t.dep_ in ("compound", "amod", "prep", "pobj") and t.i > child.i]
                if right_parts:
                    right_operand = right_operand + " " + " ".join(right_parts)
                break
            elif child.dep_ == "prep":
                # E.g. "assigned to them"
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        right_operand = grandchild.lemma_
                        # Expand NP
                        right_parts = [t.lemma_ for t in grandchild.subtree if t.dep_ in ("compound", "amod")]
                        if right_parts:
                            right_operand = " ".join(right_parts) + " " + right_operand
                        break
                        
        return left_operand, right_operand

    def analyze(self, constraint: ConstraintSpan, entities: List[ABACEntity]) -> Optional[CanonicalCondition]:
        """Analyze a constraint span and map it to a CanonicalCondition."""
        if not constraint.text:
            return None
            
        # 1. Pronoun Resolution (replaces 'they', 'their', 'them' with Subject/Object refs)
        resolved_text = self.resolver.substitute_in_constraint(constraint.text, entities, self.nlp)
        
        # 2. Dependency Parsing
        doc = self.nlp(resolved_text)
        
        # 3. Find the relational root verb
        root = None
        for token in doc:
            if token.dep_ == "ROOT" or (token.dep_ in ("ccomp", "relcl", "advcl") and not root):
                root = token
                if token.dep_ == "ROOT":
                    break
                    
        if not root:
            # Fallback for non-verbal constraints (e.g. "for active contracts")
            for token in doc:
                if token.pos_ == "ADP":
                    root = token
                    break
            if not root:
                return None
                
        # 4. Determine Operator using normalizer
        # We pass the root lemma and its immediate particle/preposition if present
        operator_text = root.lemma_.lower()
        
        # Handle phrases like "member of", "belong to", "same as", "assigned to"
        for child in root.children:
            if child.dep_ == "prep":
                operator_text += " " + child.lemma_.lower()
            elif child.dep_ == "prt":
                operator_text += " " + child.lemma_.lower()
                
        # Special check for "same" anywhere in the constraint
        for token in doc:
            if token.lemma_.lower() == "same":
                operator_text = "same as"
                break
                
        canonical_operator = normalize_operator(operator_text)
        
        # 5. Extract Arguments
        left_operand, right_operand = self._extract_arguments(doc, root)
        
        if not left_operand:
            # If we still don't have a left operand, try defaulting to the main entity the constraint modifies
            if entities:
                left_operand = entities[0].lemma
            else:
                return None
                
        # Default right operand if none found
        if not right_operand:
            right_operand = ""
            
        # Determine if right operand is a reference to another entity (e.g. "subject's department")
        attribute_ref = None
        values = []
        
        if "subject" in right_operand.lower() or "resource" in right_operand.lower() or "object" in right_operand.lower():
            attribute_ref = right_operand
        else:
            values = [right_operand]
            
        return CanonicalCondition(
            attribute=left_operand,
            operator=str(canonical_operator),
            values=values,
            attribute_ref=attribute_ref
        )
