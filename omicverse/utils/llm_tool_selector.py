"""
LLM-driven tool selection for OmicVerse Agent.

Replaces both ``_analyze_task_complexity`` and ``_select_skill_matches_llm``
with a **single** LLM call that decides complexity, skill selection, and MCP
need simultaneously.

Design
------
* ``ToolSelectionResult`` – structured output from a single LLM call.
* ``LLMToolSelector`` – stateless selector that takes a request + catalog and
  returns a selection result.  Falls back to keyword heuristics when the LLM
  is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolSelectionResult:
    """Structured result of a single LLM tool-selection call."""

    complexity: str                    # "simple" | "complex"
    selected_skills: List[str]         # skill slugs
    needs_external_db: bool            # whether to lazy-connect MCP
    reasoning: str = ""                # LLM's brief rationale

    def __post_init__(self) -> None:
        if self.complexity not in ("simple", "complex"):
            self.complexity = "complex"  # safe default


# ---------------------------------------------------------------------------
# Keyword fallback (used when LLM is unavailable)
# ---------------------------------------------------------------------------

_COMPLEX_KEYWORDS = [
    "complete", "full", "entire", "whole", "comprehensive",
    "pipeline", "workflow", "from start", "end-to-end",
    "step by step", "all steps", "everything",
    "multiple", "several", "various", "different steps",
    "and then", "followed by", "after that", "next",
    "analysis pipeline", "full analysis",
]

_SIMPLE_KEYWORDS = [
    "just", "only", "single", "one", "simply",
    "quick", "fast", "basic",
    "describe", "explain", "print", "display", "show summary", "list",
    "what is", "how many", "count",
]

_SIMPLE_FUNCTIONS = [
    "qc", "quality control",
    "normalize", "normalization",
    "pca", "dimensionality reduction",
    "cluster", "clustering", "leiden", "louvain",
    "plot", "visualize", "show",
    "filter", "subset",
    "scale", "log transform",
]

_MCP_KEYWORDS = [
    "interaction partners", "ppi network", "string",
    "uniprot", "gene function", "protein lookup",
    "kegg pathway", "reactome", "pathway lookup",
    "panglao", "cell markers", "known markers",
    "open targets", "drug target",
    "search papers", "pubmed", "europepmc", "literature",
]


def _keyword_fallback(request: str) -> ToolSelectionResult:
    """Fast keyword-based classification when LLM is unavailable."""
    req_lower = request.lower()

    complex_score = sum(1 for kw in _COMPLEX_KEYWORDS if kw in req_lower)
    simple_score = sum(1 for kw in _SIMPLE_KEYWORDS if kw in req_lower)
    func_matches = sum(1 for fn in _SIMPLE_FUNCTIONS if fn in req_lower)
    mcp_matches = sum(1 for kw in _MCP_KEYWORDS if kw in req_lower)

    if complex_score >= 2:
        complexity = "complex"
    elif func_matches >= 1 and complex_score == 0 and len(request.split()) <= 10:
        complexity = "simple"
    elif simple_score >= 1 and complex_score == 0:
        complexity = "simple"
    else:
        complexity = "complex"  # default to complex for safety

    return ToolSelectionResult(
        complexity=complexity,
        selected_skills=[],
        needs_external_db=mcp_matches > 0,
        reasoning="keyword fallback (LLM unavailable)",
    )


# ---------------------------------------------------------------------------
# LLM Tool Selector
# ---------------------------------------------------------------------------

class LLMToolSelector:
    """LLM-driven tool selection replacing keyword routing.

    Combines task complexity analysis, skill matching, and MCP need detection
    into a single LLM call.

    Parameters
    ----------
    catalog : ToolCatalog
        Pre-built unified tool catalog.
    llm_backend : OmicVerseLLMBackend
        LLM backend for making the selection call.
    fallback_to_keywords : bool
        Whether to fall back to keyword heuristics when LLM fails.
    """

    _SELECTION_PROMPT_TEMPLATE = """You are a bioinformatics tool selector for OmicVerse Agent.
Given a user request and dataset state, decide what tools are needed.

Request: "{request}"
Dataset: {adata_summary}

Available Tools:
{catalog}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "complexity": "simple" or "complex",
  "skills": ["skill-slug-1", "skill-slug-2"],
  "needs_external_db": true or false,
  "reasoning": "brief explanation"
}}

Rules:
- "simple": single operation (QC, normalize, cluster, plot, filter, scale)
- "complex": multi-step workflow, pipeline, vague request, or requires combining multiple operations
- "skills": 0-2 most relevant skill slugs from the [skills] section above. Use [] if none are relevant.
- "needs_external_db": true ONLY when external databases are needed (protein interactions, pathway lookup, literature search, cell-type markers from databases, etc.). false for all standard local analysis.
"""

    def __init__(
        self,
        catalog: Any,  # ToolCatalog
        llm_backend: Any,  # OmicVerseLLMBackend
        *,
        fallback_to_keywords: bool = True,
    ):
        self._catalog = catalog
        self._llm = llm_backend
        self._fallback_to_keywords = fallback_to_keywords
        # Cache the catalog text (only changes when catalog is rebuilt)
        self._cached_catalog_prompt = catalog.get_compact_catalog()

    def refresh_catalog(self) -> None:
        """Re-read the compact catalog (e.g. after MCP connect/disconnect)."""
        self._cached_catalog_prompt = self._catalog.get_compact_catalog()

    async def select(
        self,
        request: str,
        adata_summary: str = "unknown",
    ) -> ToolSelectionResult:
        """Single LLM call to analyze request and select tools.

        Parameters
        ----------
        request : str
            The user's natural language request.
        adata_summary : str
            Brief description of the dataset (shape, columns, etc.).

        Returns
        -------
        ToolSelectionResult
            Structured selection with complexity, skills, and MCP need.
        """
        prompt = self._SELECTION_PROMPT_TEMPLATE.format(
            request=request,
            adata_summary=adata_summary,
            catalog=self._cached_catalog_prompt,
        )

        try:
            response = await self._llm.run(prompt)
            return self._parse_response(response)
        except Exception as exc:
            logger.warning("LLM tool selection failed: %s", exc)
            if self._fallback_to_keywords:
                logger.info("Falling back to keyword-based selection")
                return _keyword_fallback(request)
            raise

    def select_sync(
        self,
        request: str,
        adata_summary: str = "unknown",
    ) -> ToolSelectionResult:
        """Synchronous wrapper around :meth:`select`.

        Useful when called from non-async contexts (e.g. ``stream_async``
        setup before the async generator starts).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self.select(request, adata_summary))

        # Running inside existing loop (Jupyter) – use thread
        import threading
        result = None
        exception = None

        def _run():
            nonlocal result, exception
            try:
                result = asyncio.run(self.select(request, adata_summary))
            except Exception as exc:
                exception = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=60)
        if exception is not None:
            raise exception
        return result

    # -- parsing -------------------------------------------------------------

    def _parse_response(self, response: str) -> ToolSelectionResult:
        """Parse the LLM's JSON response into a ToolSelectionResult."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in LLM response, falling back")
            if self._fallback_to_keywords:
                return _keyword_fallback("")
            raise ValueError(f"Could not parse LLM response: {response[:200]}")

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in LLM response: %s", exc)
            if self._fallback_to_keywords:
                return _keyword_fallback("")
            raise ValueError(f"Invalid JSON: {exc}") from exc

        complexity = data.get("complexity", "complex")
        skills_raw = data.get("skills", [])
        needs_db = data.get("needs_external_db", False)
        reasoning = data.get("reasoning", "")

        # Validate skill slugs against catalog
        valid_slugs = []
        if isinstance(skills_raw, list):
            catalog_slugs = {e.slug for e in self._catalog.entries if e.category == "skill"}
            for slug in skills_raw:
                if isinstance(slug, str) and slug in catalog_slugs:
                    valid_slugs.append(slug)

        return ToolSelectionResult(
            complexity=str(complexity),
            selected_skills=valid_slugs,
            needs_external_db=bool(needs_db),
            reasoning=str(reasoning),
        )
