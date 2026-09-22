"""Gazetteer providers for lexical and domain knowledge.

[EXTENDED] – FileBasedGazetteer implements GazetteerProvider by loading
JSON files from disk. CompositeGazetteer combines multiple providers.

Extension hook: replace FileBasedGazetteer with DatabaseGazetteer or
OntologyGazetteer at the injection site (RuleBasedSAOExtractor) without
modifying any component logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..core.interfaces import GazetteerProvider

logger = logging.getLogger(__name__)


class FileBasedGazetteer(GazetteerProvider):
    """Load domain knowledge from local JSON files.

    Each JSON file is expected to be a flat object mapping category names
    to lists of strings, for example::

        {
          "permit": ["allow", "may", "can", ...],
          "deny":   ["prohibit", "forbid", ...]
        }

    Multiple files can be loaded; their namespaces are merged so that the
    same category name in different files is treated as one combined list.

    Parameters
    ----------
    None — call ``load(path)`` for each JSON file to populate.
    """

    def __init__(self) -> None:
        # namespace_stem → { category → [terms] }
        self._data: Dict[str, Dict[str, List[str]]] = {}
        # Merged flat view: category → set of lower-case terms (for fast lookup)
        self._index: Dict[str, set[str]] = {}

    def load(self, source: str) -> None:
        """Load a JSON gazetteer file and merge it into the internal index.

        Parameters
        ----------
        source : str
            Absolute or relative path to a JSON file.
        """
        path = Path(source)
        if not path.exists():
            logger.warning("Gazetteer file not found: %s", source)
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                content: Dict[str, List[str]] = json.load(f)

            self._data[path.stem] = content

            # Rebuild flat index (merge across namespaces)
            for category, terms in content.items():
                if category not in self._index:
                    self._index[category] = set()
                self._index[category].update(t.lower() for t in terms)

            logger.info("Loaded gazetteer '%s' (%d categories).", path.stem, len(content))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load gazetteer %s: %s", source, exc)

    def match(self, span: Any, category: str) -> list[str]:
        """Check whether a spaCy token/span matches a gazetteer category.

        Matching strategy (in order of priority):
        1. **Lemma match**: ``token.lemma_.lower()`` in the category wordlist.
        2. **Surface form match**: ``token.text.lower()`` in the category wordlist.

        Both the lemma and the text are checked to handle irregular forms
        (e.g. lemma "permit" for the token "permitted").

        Parameters
        ----------
        span : spacy.tokens.Token | spacy.tokens.Span
            A single spaCy token or span. For spans, only the root token's
            lemma/text is checked (the Gazetteer is term-level, not phrase-level).
        category : str
            The category key to look up (e.g. ``"permit"``, ``"roles"``).

        Returns
        -------
        list[str]
            A non-empty list of the matching term(s) if found, else an empty list.
            The list is truthy when there is at least one match, falsy when empty.
        """
        if category not in self._index:
            return []

        wordset = self._index[category]

        # Prefer the spaCy lemma; fall back to raw text
        candidates: list[str] = []
        try:
            candidates.append(span.lemma_.lower())
        except AttributeError:
            pass
        try:
            candidates.append(span.text.lower())
        except AttributeError:
            pass

        matched = [c for c in candidates if c in wordset]
        return list(dict.fromkeys(matched))  # deduplicated, order-preserving

    # ------------------------------------------------------------------
    # Introspection helpers (useful for testing)
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return all known category names across all loaded files."""
        return list(self._index.keys())

    def terms(self, category: str) -> list[str]:
        """Return all terms for *category*, sorted."""
        return sorted(self._index.get(category, set()))


class CompositeGazetteer(GazetteerProvider):
    """Combine multiple GazetteerProvider instances.

    Useful for mixing sources (e.g. a file-based provider plus a database
    provider) without changing any component logic.

    # [HOOK] Extension point: add DatabaseGazetteer or OntologyGazetteer
    # here without modifying FileBasedGazetteer or any pipeline component.
    """

    def __init__(self, providers: List[GazetteerProvider]) -> None:
        self.providers = providers

    def load(self, source: str) -> None:
        """Delegate ``load`` to all providers that accept this source."""
        for provider in self.providers:
            try:
                provider.load(source)
            except Exception:  # noqa: BLE001
                pass

    def match(self, span: Any, category: str) -> list[str]:
        """Return deduplicated union of matches from all providers."""
        matches: list[str] = []
        for provider in self.providers:
            matches.extend(provider.match(span, category))
        return list(dict.fromkeys(matches))  # deduplicated, order-preserving
