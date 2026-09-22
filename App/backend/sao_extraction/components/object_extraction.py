"""Object extraction component.

[EXTENDED] – Extracts Object noun phrases from each clause using spaCy
dependency relations: ``dobj``, ``attr``, ``oprd`` (direct/attribute
complements), and ``prep → pobj`` patterns (prepositional objects).
Coordinating conjuncts (``conj``) are collected to handle multiple objects.
No domain wordlists are hardcoded here.

[BASELINE: Alohaly et al. 2019] – Object entities are candidates for the
resource-type attribute classification described in the paper; this component
identifies the token spans that would serve as CNN classifier inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.interfaces import PipelineComponent
from ..core.schemas import PipelineContext
from .subject_extraction import expand_noun_phrase  # reuse NP expansion utility

logger = logging.getLogger(__name__)

# Structural dependency constants — not domain knowledge
_DIRECT_OBJ_DEPS = frozenset({"dobj", "attr", "oprd"})
_PREP_OBJ_DEPS = frozenset({"pobj"})
_PREP_DEP = "prep"
_CONJ_DEP = "conj"
_RELCL_DEP = "relcl"

# Prepositions that typically introduce the Object in access-control sentences
# (structural relationship patterns, not domain vocabulary)
_ACCESS_PREPS = frozenset({"to", "on", "of", "for", "from", "into", "about"})


def _find_action_token(clause_span: Any, action_text: Optional[str]) -> Optional[Any]:
    """Return the token in *clause_span* whose lemma matches *action_text*.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
    action_text : str | None
        Lemmatised action string from ActionExtractor.

    Returns
    -------
    Token | None
    """
    if not action_text:
        return None
    # action_text may be "verb particle" — match on first word (the verb lemma)
    action_lemma = action_text.split()[0]
    for token in clause_span:
        if token.lemma_.lower() == action_lemma and token.pos_ in ("VERB", "AUX"):
            return token
    return None


def _collect_conj_objects(obj_token: Any) -> list[Any]:
    """Return *obj_token* plus any coordinating object conjuncts.

    E.g. "access records and files" → [records, files]

    Parameters
    ----------
    obj_token : spacy.tokens.Token

    Returns
    -------
    list[Token]
    """
    result = [obj_token]
    for child in obj_token.children:
        if child.dep_ == _CONJ_DEP:
            result.append(child)
            # Recurse one level for "a, b, and c" patterns
            result.extend(
                gc for gc in child.children if gc.dep_ == _CONJ_DEP
            )
    return result


def extract_objects_from_span(
    clause_span: Any, action_text: Optional[str]
) -> list[str]:
    """Extract all object noun phrases from a clause.

    Algorithm
    ---------
    1. Locate the action verb token in the clause.
    2. Check direct dependents with deps in {dobj, attr, oprd}.
    3. Check ``prep → pobj`` children of the action verb.
    4. Expand each candidate into a full NP (excluding relcl).
    5. Collect conjuncts for coordination (multiple objects).

    If the action verb cannot be located, falls back to scanning the entire
    clause for dobj/attr tokens.

    Parameters
    ----------
    clause_span : spacy.tokens.Span
    action_text : str | None
        Lemmatised action from ActionExtractor.

    Returns
    -------
    list[str]
        Deduplicated list of NP texts (preserving order).
    """
    action_token = _find_action_token(clause_span, action_text)
    candidate_tokens: list[Any] = []

    if action_token is not None:
        # Direct object deps of the action verb
        for child in action_token.children:
            if child.dep_ in _DIRECT_OBJ_DEPS:
                candidate_tokens.extend(_collect_conj_objects(child))

        # Prepositional objects ("access to files", "view records of patients")
        for child in action_token.children:
            if child.dep_ == _PREP_DEP and child.lemma_.lower() in _ACCESS_PREPS:
                for grandchild in child.children:
                    if grandchild.dep_ in _PREP_OBJ_DEPS:
                        candidate_tokens.extend(_collect_conj_objects(grandchild))

        # Also check xcomp verb's objects (for "permitted to access records")
        for child in action_token.children:
            if child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
                for gc in child.children:
                    if gc.dep_ in _DIRECT_OBJ_DEPS:
                        candidate_tokens.extend(_collect_conj_objects(gc))
                    if gc.dep_ == _PREP_DEP and gc.lemma_.lower() in _ACCESS_PREPS:
                        for ggc in gc.children:
                            if ggc.dep_ in _PREP_OBJ_DEPS:
                                candidate_tokens.extend(_collect_conj_objects(ggc))
    else:
        # Fallback: scan entire clause for dobj/attr
        for token in clause_span:
            if token.dep_ in _DIRECT_OBJ_DEPS:
                candidate_tokens.extend(_collect_conj_objects(token))

    # Expand NPs and deduplicate (preserve order)
    seen: set[str] = set()
    objects: list[str] = []
    for tok in candidate_tokens:
        np_text = expand_noun_phrase(tok)
        if np_text and np_text not in seen:
            seen.add(np_text)
            objects.append(np_text)

    return objects


class ObjectExtractor(PipelineComponent):
    """Extract Object noun phrases from each clause.

    Requires
    --------
    context.clause_docs : list[spacy.tokens.Span]
    context.metadata["actions"] : list[str | None]

    Writes
    ------
    context.metadata["objects"] : list[list[str]]
        One list of object strings per clause.
    """

    @property
    def name(self) -> str:
        return "object_extraction"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract objects from all clauses and store in metadata."""
        clauses = context.clause_docs or [context.doc[:]]
        actions = context.metadata.get("actions", [None] * len(clauses))

        objects_per_clause: list[list[str]] = []
        for i, span in enumerate(clauses):
            action = actions[i] if i < len(actions) else None
            objs = extract_objects_from_span(span, action)
            objects_per_clause.append(objs)
            logger.debug("Clause %d objects: %s", i, objs)

        context.metadata["objects"] = objects_per_clause
        return context
