"""
Unit tests for the LLM-driven tool selector.

All tests are fully mocked — no network or LLM access required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omicverse.utils.llm_tool_selector import (
    LLMToolSelector,
    ToolSelectionResult,
    _keyword_fallback,
)
from omicverse.utils.tool_catalog import ToolCatalog, CatalogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog():
    """Create a ToolCatalog with test entries."""
    catalog = ToolCatalog()
    catalog.add_entry(CatalogEntry(
        name="Bulk DEG Analysis", slug="bulk-deg-analysis",
        category="skill", description="DEG pipeline for bulk RNA-seq",
        source="builtin_skill",
    ))
    catalog.add_entry(CatalogEntry(
        name="Single-cell Clustering", slug="single-clustering",
        category="skill", description="Clustering workflow for scRNA-seq",
        source="builtin_skill",
    ))
    catalog.add_entry(CatalogEntry(
        name="string_interaction_partners", slug="string_interaction_partners",
        category="mcp_tool", description="PPI from STRING",
        source="biocontext",
    ))
    return catalog


def _make_llm_backend(response_text: str):
    """Create a mock LLM backend returning a fixed response."""
    backend = MagicMock()
    backend.run = AsyncMock(return_value=response_text)
    return backend


# ---------------------------------------------------------------------------
# Tests: ToolSelectionResult
# ---------------------------------------------------------------------------

class TestToolSelectionResult:
    def test_valid_simple(self):
        r = ToolSelectionResult(
            complexity="simple", selected_skills=[], needs_external_db=False,
        )
        assert r.complexity == "simple"

    def test_valid_complex(self):
        r = ToolSelectionResult(
            complexity="complex", selected_skills=["bulk-deg-analysis"],
            needs_external_db=True, reasoning="Multi-step DEG pipeline",
        )
        assert r.complexity == "complex"
        assert len(r.selected_skills) == 1

    def test_invalid_complexity_defaults_to_complex(self):
        r = ToolSelectionResult(
            complexity="unknown", selected_skills=[], needs_external_db=False,
        )
        assert r.complexity == "complex"


# ---------------------------------------------------------------------------
# Tests: Keyword fallback
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def test_simple_qc(self):
        result = _keyword_fallback("qc with nUMI>500")
        assert result.complexity == "simple"
        assert not result.needs_external_db

    def test_complex_pipeline(self):
        result = _keyword_fallback("complete full end-to-end DEG analysis pipeline")
        assert result.complexity == "complex"

    def test_mcp_detection(self):
        result = _keyword_fallback("find TP53 interaction partners in STRING")
        assert result.needs_external_db

    def test_simple_plot(self):
        result = _keyword_fallback("plot UMAP")
        assert result.complexity == "simple"

    def test_empty_request(self):
        result = _keyword_fallback("")
        assert result.complexity == "complex"  # default to complex


# ---------------------------------------------------------------------------
# Tests: LLMToolSelector.select
# ---------------------------------------------------------------------------

class TestLLMToolSelectorSelect:
    @pytest.mark.asyncio
    async def test_simple_task(self):
        catalog = _make_catalog()
        llm_response = json.dumps({
            "complexity": "simple",
            "skills": [],
            "needs_external_db": False,
            "reasoning": "Single QC operation",
        })
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend)

        result = await selector.select("quality control with nUMI>500")
        assert result.complexity == "simple"
        assert result.selected_skills == []
        assert not result.needs_external_db

    @pytest.mark.asyncio
    async def test_complex_task_with_skills(self):
        catalog = _make_catalog()
        llm_response = json.dumps({
            "complexity": "complex",
            "skills": ["bulk-deg-analysis"],
            "needs_external_db": False,
            "reasoning": "Multi-step DEG pipeline",
        })
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend)

        result = await selector.select("complete bulk RNA-seq DEG pipeline")
        assert result.complexity == "complex"
        assert "bulk-deg-analysis" in result.selected_skills

    @pytest.mark.asyncio
    async def test_mcp_need_detected(self):
        catalog = _make_catalog()
        llm_response = json.dumps({
            "complexity": "simple",
            "skills": [],
            "needs_external_db": True,
            "reasoning": "Need STRING interaction data",
        })
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend)

        result = await selector.select("find TP53 interaction partners")
        assert result.needs_external_db

    @pytest.mark.asyncio
    async def test_invalid_skill_slugs_filtered(self):
        catalog = _make_catalog()
        llm_response = json.dumps({
            "complexity": "complex",
            "skills": ["bulk-deg-analysis", "nonexistent-skill"],
            "needs_external_db": False,
            "reasoning": "test",
        })
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend)

        result = await selector.select("DEG analysis")
        assert result.selected_skills == ["bulk-deg-analysis"]

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        catalog = _make_catalog()
        backend = MagicMock()
        backend.run = AsyncMock(side_effect=RuntimeError("API error"))
        selector = LLMToolSelector(catalog, backend, fallback_to_keywords=True)

        result = await selector.select("quality control")
        assert result.reasoning == "keyword fallback (LLM unavailable)"

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self):
        catalog = _make_catalog()
        backend = MagicMock()
        backend.run = AsyncMock(side_effect=RuntimeError("API error"))
        selector = LLMToolSelector(catalog, backend, fallback_to_keywords=False)

        with pytest.raises(RuntimeError):
            await selector.select("quality control")


# ---------------------------------------------------------------------------
# Tests: Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_json_with_markdown_wrapper(self):
        catalog = _make_catalog()
        llm_response = '```json\n{"complexity": "simple", "skills": [], "needs_external_db": false, "reasoning": "test"}\n```'
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend)

        result = await selector.select("plot UMAP")
        assert result.complexity == "simple"

    @pytest.mark.asyncio
    async def test_no_json_in_response(self):
        catalog = _make_catalog()
        llm_response = "I think this is a simple task with no JSON"
        backend = _make_llm_backend(llm_response)
        selector = LLMToolSelector(catalog, backend, fallback_to_keywords=True)

        result = await selector.select("plot UMAP")
        # Falls back to keyword-based
        assert result.reasoning == "keyword fallback (LLM unavailable)"


# ---------------------------------------------------------------------------
# Tests: Catalog refresh
# ---------------------------------------------------------------------------

class TestCatalogRefresh:
    def test_refresh_updates_prompt(self):
        catalog = _make_catalog()
        backend = _make_llm_backend("{}")
        selector = LLMToolSelector(catalog, backend)

        original = selector._cached_catalog_prompt
        catalog.add_entry(CatalogEntry(
            name="new", slug="new", category="skill",
            description="new skill", source="user_skill",
        ))
        selector.refresh_catalog()
        assert selector._cached_catalog_prompt != original
