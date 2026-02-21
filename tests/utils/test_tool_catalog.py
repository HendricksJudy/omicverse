"""
Unit tests for the unified ToolCatalog.

All tests are fully self-contained — no network or LLM access required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omicverse.utils.tool_catalog import ToolCatalog, CatalogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSkillMeta:
    """Simple fake skill metadata (MagicMock.name is special, so we avoid it)."""
    def __init__(self, name, slug, description, path):
        self.name = name
        self.slug = slug
        self.description = description
        self.path = path


def _make_skill_registry(skills: dict | None = None):
    """Create a mock SkillRegistry with skill_metadata."""
    reg = MagicMock()
    if skills is None:
        skills = {
            "bulk-deg-analysis": _FakeSkillMeta(
                name="Bulk DEG Analysis",
                slug="bulk-deg-analysis",
                description="Differential expression for bulk RNA-seq",
                path=Path("/pkg/.claude/skills/bulk-deg-analysis"),
            ),
            "single-clustering": _FakeSkillMeta(
                name="Single-cell Clustering",
                slug="single-clustering",
                description="Clustering workflow for scRNA-seq",
                path=Path("/pkg/.claude/skills/single-clustering"),
            ),
        }
    reg.skill_metadata = skills
    return reg


def _make_mcp_manager(tools: list | None = None):
    """Create a mock MCPClientManager with list_tools()."""
    mgr = MagicMock()
    if tools is None:
        tool1 = MagicMock(name="string_interaction_partners", description="PPI from STRING", server_name="biocontext")
        tool1.name = "string_interaction_partners"
        tool1.description = "PPI from STRING"
        tool1.server_name = "biocontext"
        tool2 = MagicMock(name="kegg_pathway_info", description="KEGG pathways", server_name="biocontext")
        tool2.name = "kegg_pathway_info"
        tool2.description = "KEGG pathways"
        tool2.server_name = "biocontext"
        tools = [tool1, tool2]
    mgr.list_tools.return_value = tools
    return mgr


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------

class TestToolCatalogConstruction:
    def test_empty_catalog(self):
        catalog = ToolCatalog()
        assert len(catalog) == 0
        assert catalog.get_compact_catalog() == ""

    def test_from_skills(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        skills = catalog.get_entries_by_category("skill")
        assert len(skills) == 2
        slugs = {e.slug for e in skills}
        assert "bulk-deg-analysis" in slugs
        assert "single-clustering" in slugs

    def test_from_mcp(self):
        catalog = ToolCatalog(mcp_manager=_make_mcp_manager())
        mcp_tools = catalog.get_entries_by_category("mcp_tool")
        assert len(mcp_tools) == 2
        assert mcp_tools[0].source == "biocontext"

    def test_combined(self):
        catalog = ToolCatalog(
            skill_registry=_make_skill_registry(),
            mcp_manager=_make_mcp_manager(),
        )
        assert len(catalog) == 4
        assert len(catalog.get_entries_by_category("skill")) == 2
        assert len(catalog.get_entries_by_category("mcp_tool")) == 2


# ---------------------------------------------------------------------------
# Tests: Compact catalog text
# ---------------------------------------------------------------------------

class TestCompactCatalog:
    def test_skills_section(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        text = catalog.get_compact_catalog()
        assert "[skills]" in text
        assert "bulk-deg-analysis:" in text
        assert "single-clustering:" in text

    def test_mcp_section(self):
        catalog = ToolCatalog(mcp_manager=_make_mcp_manager())
        text = catalog.get_compact_catalog()
        assert "[mcp_tools]" in text
        assert "string_interaction_partners:" in text

    def test_cached(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        text1 = catalog.get_compact_catalog()
        text2 = catalog.get_compact_catalog()
        assert text1 is text2  # same object (cached)


# ---------------------------------------------------------------------------
# Tests: Entry lookup
# ---------------------------------------------------------------------------

class TestEntryLookup:
    def test_get_entries_by_slugs(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        entries = catalog.get_entries_by_slugs(["bulk-deg-analysis"])
        assert len(entries) == 1
        assert entries[0].slug == "bulk-deg-analysis"

    def test_get_entries_by_slugs_unknown(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        entries = catalog.get_entries_by_slugs(["nonexistent"])
        assert len(entries) == 0

    def test_get_entries_by_category(self):
        catalog = ToolCatalog(
            skill_registry=_make_skill_registry(),
            mcp_manager=_make_mcp_manager(),
        )
        skills = catalog.get_entries_by_category("skill")
        assert all(e.category == "skill" for e in skills)


# ---------------------------------------------------------------------------
# Tests: Mutation
# ---------------------------------------------------------------------------

class TestCatalogMutation:
    def test_add_entry(self):
        catalog = ToolCatalog()
        entry = CatalogEntry(
            name="test_tool",
            slug="test-tool",
            category="mcp_tool",
            description="A test tool",
            source="custom_mcp",
        )
        catalog.add_entry(entry)
        assert len(catalog) == 1
        assert catalog.get_entries_by_slugs(["test-tool"])[0].name == "test_tool"

    def test_add_entry_invalidates_cache(self):
        catalog = ToolCatalog(skill_registry=_make_skill_registry())
        text1 = catalog.get_compact_catalog()
        catalog.add_entry(CatalogEntry(
            name="new", slug="new", category="skill", description="new skill", source="user_skill",
        ))
        text2 = catalog.get_compact_catalog()
        assert text1 is not text2  # cache invalidated

    def test_remove_entries_by_source(self):
        catalog = ToolCatalog(mcp_manager=_make_mcp_manager())
        assert len(catalog) == 2
        removed = catalog.remove_entries_by_source("biocontext")
        assert removed == 2
        assert len(catalog) == 0

    def test_rebuild_mcp_entries(self):
        catalog = ToolCatalog(mcp_manager=_make_mcp_manager())
        assert len(catalog.get_entries_by_category("mcp_tool")) == 2

        # Simulate disconnect: rebuild with empty manager
        empty_mgr = MagicMock()
        empty_mgr.list_tools.return_value = []
        catalog.rebuild_mcp_entries(empty_mgr)
        assert len(catalog.get_entries_by_category("mcp_tool")) == 0


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        catalog = ToolCatalog(
            skill_registry=_make_skill_registry(),
            mcp_manager=_make_mcp_manager(),
        )
        r = repr(catalog)
        assert "skills=2" in r
        assert "mcp_tools=2" in r
