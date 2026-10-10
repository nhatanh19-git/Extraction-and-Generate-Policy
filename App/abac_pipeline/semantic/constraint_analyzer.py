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
                
        # If no subject is found, check if it's a relative clause or preposition where the subject is implicit
        if not left_operand and root.dep_ in ("relcl", "prep", "acl"):
            left_operand = root.head.lemma_
                
        # Find right operand
        # Check direct objects and attributes
        for child in root.children:
            if child.dep_ in ("dobj", "attr", "acomp", "pobj"):
                right_operand = doc[child.left_edge.i : child.right_edge.i + 1].text
                break
            elif child.dep_ in ("prep", "agent"):
                # E.g. "assigned to them", "works in which", "sent by them"
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        # If pobj is a relative pronoun ('which', 'that'), it refers to the relcl head
                        if grandchild.lemma_.lower() in ("which", "that", "who", "whom") and root.dep_ in ("relcl", "acl"):
                            right_operand = root.head.lemma_
                        else:
                            right_operand = doc[grandchild.left_edge.i : grandchild.right_edge.i + 1].text
                        break
        
        # Another case: if root is relcl and we haven't found a right operand, the head might be the right operand
        # e.g. "tasks that are assigned to them" -> assigned is relcl of tasks. tasks is left operand. them is right.
        if root.dep_ == "relcl":
            if not left_operand or left_operand == root.head.lemma_:
                # If left is the head, we still need right.
                pass
                        
        return left_operand, right_operand

    def analyze(self, constraint: ConstraintSpan, doc: spacy.tokens.Doc, entities: List[ABACEntity]) -> Optional[CanonicalCondition]:
        """Analyze a constraint span and map it to a CanonicalCondition using the full sentence doc."""
        if not constraint.text or constraint.start < 0 or constraint.end >= len(doc):
            return None
            
        span = doc[constraint.start : constraint.end + 1]
        
        # 1. Find the topological root of the constraint span
        # The root is the token whose head is outside the span, or the ROOT of the document.
        root = None
        for token in span:
            if token.head.i < constraint.start or token.head.i > constraint.end or token.dep_ == "ROOT":
                root = token
                break
                
        if not root:
            # Fallback to the old logic if somehow all heads are inside (e.g., circular or isolated parse)
            for token in span:
                if token.dep_ in ("ccomp", "relcl", "advcl", "acl", "xcomp"):
                    root = token
                    break
            if not root:
                for token in span:
                    if token.pos_ == "ADP" or token.dep_ == "prep":
                        root = token
                        break
            if not root:
                return None
                
        # 2. Determine Operator using normalizer
        # We pass the root lemma and its immediate particle/preposition if present
        operator_text = root.lemma_.lower()
        
        # Handle phrases like "member of", "belong to", "same as", "assigned to"
        for child in root.children:
            if child.dep_ == "prep" and child.i >= constraint.start and child.i <= constraint.end:
                operator_text += " " + child.lemma_.lower()
            elif child.dep_ == "prt" and child.i >= constraint.start and child.i <= constraint.end:
                operator_text += " " + child.lemma_.lower()
                
        # Special check for "same" anywhere in the constraint
        for token in span:
            if token.lemma_.lower() == "same":
                operator_text = "same as"
                break
                
        canonical_operator = normalize_operator(operator_text)
        
        # 3. Extract Arguments
        left_operand, right_operand = self._extract_arguments(doc, root)
        
        # 4. Resolve pronouns in the extracted arguments
        if left_operand:
            # Split and resolve to handle things like "his supervisees"
            resolved_left = []
            for word in left_operand.split():
                resolved_ent = self.resolver.resolve_reference(word, entities)
                if resolved_ent:
                    resolved_left.append(resolved_ent.lemma)
                else:
                    resolved_left.append(word)
            left_operand = " ".join(resolved_left)
            
        if right_operand:
            resolved_right = []
            for word in right_operand.split():
                resolved_ent = self.resolver.resolve_reference(word, entities)
                if resolved_ent:
                    resolved_right.append(resolved_ent.lemma)
                elif word.lower() == "own":
                    # implicit reference to the subject
                    subj_ent = self.resolver.resolve_reference("their", entities)
                    if subj_ent:
                        resolved_right.append(subj_ent.lemma + "'s")
                else:
                    resolved_right.append(word)
            right_operand = " ".join(resolved_right)
        
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
        elif right_operand:
            values = [right_operand]
            
        return CanonicalCondition(
            attribute=left_operand,
            operator=str(canonical_operator),
            values=values,
            attribute_ref=attribute_ref
        )
