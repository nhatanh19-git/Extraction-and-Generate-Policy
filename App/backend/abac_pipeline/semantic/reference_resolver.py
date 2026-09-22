"""Reference Resolver for pronouns in constraints."""

import spacy
from typing import Optional, List
from ..core.abac_schemas import ABACEntity

class ReferenceResolver:
    """Resolves pronouns in constraints to their antecedent subject or resource."""
    
    def __init__(self):
        # Pronouns typically referring to the subject
        self.subject_pronouns = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "user"}
        
        # Pronouns typically referring to the resource/object
        self.resource_pronouns = {"it", "its", "which", "that", "this", "file", "document", "record", "item"}
        
    def resolve_reference(self, token_text: str, entities: List[ABACEntity]) -> Optional[ABACEntity]:
        """Attempt to resolve a pronoun to an entity in the sentence.
        
        This is a heuristic resolver based on ABAC domain knowledge.
        In policies, human pronouns (he, she, they) almost always refer to the Subject.
        Object pronouns (it, which, that) almost always refer to the Resource/Object.
        """
        token_lower = token_text.lower()
        
        if token_lower in self.subject_pronouns:
            # Find subject entity
            for ent in entities:
                if ent.type == "SUBJECT":
                    return ent
                    
        elif token_lower in self.resource_pronouns:
            # Find object entity
            for ent in entities:
                if ent.type == "OBJECT":
                    return ent
                    
        return None
        
    def substitute_in_constraint(self, constraint_text: str, entities: List[ABACEntity], nlp: spacy.Language) -> str:
        """Replace pronouns and implicit self-references in the constraint text with their resolved entity texts."""
        doc = nlp(constraint_text)
        resolved_text = []
        
        for token in doc:
            token_lower = token.text.lower()
            
            # Context-aware pronoun resolution
            if token.pos_ == "PRON" or token_lower in self.subject_pronouns.union(self.resource_pronouns):
                resolved_ent = self.resolve_reference(token.text, entities)
                if resolved_ent:
                    # Append the noun instead of the pronoun
                    # Handle possessives: "his department" -> "subject's department"
                    if token.tag_ == "PRP$":
                        resolved_text.append(resolved_ent.lemma + "'s")
                    else:
                        resolved_text.append(resolved_ent.lemma)
                    continue
                    
            # Context-aware resolution for implicit references (e.g. "own", "same")
            if token_lower == "own":
                # "own" implies belonging to the subject
                resolved_ent = self.resolve_reference("their", entities)
                if resolved_ent:
                    resolved_text.append(resolved_ent.lemma + "'s")
                    continue
                    
            resolved_text.append(token.text_with_ws)
            
        return "".join(resolved_text).strip()
