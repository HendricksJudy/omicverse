"""
Unified Tool Catalog for OmicVerse Agent.

Provides a single registry that unifies Skills, MCP tools, and Registry
functions into one catalog for LLM-driven tool selection.  The LLM receives
a compact text representation of the catalog and decides which tools to use.

Design
------
* ``CatalogEntry`` – lightweight descriptor for any tool (skill, MCP, function).
* ``ToolCatalog`` – aggregates entries from :class:`SkillRegistry`,
  :class:`MCPClientManager`, and the global function registry.
* The compact catalog (~2K tokens) is pre-built at init time and cached until
  an MCP server connects or disconnects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    """Lightweight descriptor for a single tool in the unified catalog."""

    name: str
    slug: str
    category: str          # "skill" | "mcp_tool" | "registry_function"
    description: str
    source: str            # "builtin_skill" | "user_skill" | "biocontext" | "custom_mcp" | "registry"
    requires_connection: bool = False  # True for MCP tools not yet connected


class ToolCatalog:
    """Unified catalog of all available tools for LLM selection.

    Aggregates:
    - Project skills (from :class:`SkillRegistry`)
    - MCP tools (from :class:`MCPClientManager`)
    - Registry functions (from ``_global_registry``)

    The compact catalog string is cached and rebuilt only when the underlying
    sources change (e.g. MCP server connects/disconnects).
    """

    def __init__(
        self,
        skill_registry: Optional[Any] = None,
        mcp_manager: Optional[Any] = None,
        function_registry: Optional[Any] = None,
    ):
        self._entries: List[CatalogEntry] = []
        self._slug_index: Dict[str, CatalogEntry] = {}
        self._compact_cache: Optional[str] = None

        if skill_registry is not None:
            self._build_from_skills(skill_registry)
        if mcp_manager is not None:
            self._build_from_mcp(mcp_manager)
        if function_registry is not None:
            self._build_from_registry(function_registry)

        self._rebuild_index()

    # -- builders ------------------------------------------------------------

    def _build_from_skills(self, skill_registry: Any) -> None:
        """Import skill metadata into the catalog."""
        metadata_dict = getattr(skill_registry, "skill_metadata", None)
        if not metadata_dict:
            return
        for slug, meta in metadata_dict.items():
            # Determine source based on path
            path_str = str(getattr(meta, "path", ""))
            source = "user_skill" if ".claude/skills" in path_str and "omicverse" not in path_str else "builtin_skill"
            self._entries.append(CatalogEntry(
                name=meta.name,
                slug=meta.slug,
                category="skill",
                description=meta.description,
                source=source,
            ))

    def _build_from_mcp(self, mcp_manager: Any) -> None:
        """Import MCP tool descriptors into the catalog."""
        tools = []
        if hasattr(mcp_manager, "list_tools"):
            tools = mcp_manager.list_tools()
        for tool in tools:
            server_name = getattr(tool, "server_name", "unknown")
            source = "biocontext" if server_name == "biocontext" else "custom_mcp"
            self._entries.append(CatalogEntry(
                name=tool.name,
                slug=tool.name,
                category="mcp_tool",
                description=getattr(tool, "description", ""),
                source=source,
                requires_connection=False,  # already connected
            ))

    def _build_from_registry(self, function_registry: Any) -> None:
        """Import function registry entries into the catalog.

        The function registry can be very large, so we only include
        category-level summaries rather than individual functions.
        """
        if not hasattr(function_registry, "list_categories"):
            return
        try:
            categories = function_registry.list_categories()
            for cat_name, cat_info in categories.items():
                count = cat_info.get("count", 0)
                desc = cat_info.get("description", f"{count} functions")
                self._entries.append(CatalogEntry(
                    name=f"registry:{cat_name}",
                    slug=f"registry-{cat_name}",
                    category="registry_function",
                    description=desc,
                    source="registry",
                ))
        except Exception:
            # If registry doesn't support list_categories, skip it
            pass

    def _rebuild_index(self) -> None:
        """Rebuild the slug index and invalidate cached text."""
        self._slug_index = {e.slug: e for e in self._entries}
        self._compact_cache = None

    # -- public API ----------------------------------------------------------

    def add_entry(self, entry: CatalogEntry) -> None:
        """Add a single entry and invalidate the cache."""
        self._entries.append(entry)
        self._slug_index[entry.slug] = entry
        self._compact_cache = None

    def remove_entries_by_source(self, source: str) -> int:
        """Remove all entries from a given source and invalidate cache.

        Returns the number of entries removed.
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.source != source]
        self._rebuild_index()
        return before - len(self._entries)

    def rebuild_mcp_entries(self, mcp_manager: Any) -> None:
        """Replace MCP tool entries with current state of the MCP manager."""
        self._entries = [e for e in self._entries if e.category != "mcp_tool"]
        self._build_from_mcp(mcp_manager)
        self._rebuild_index()

    @property
    def entries(self) -> List[CatalogEntry]:
        return list(self._entries)

    def get_entries_by_slugs(self, slugs: List[str]) -> List[CatalogEntry]:
        """Look up catalog entries by their slugs."""
        return [self._slug_index[s] for s in slugs if s in self._slug_index]

    def get_entries_by_category(self, category: str) -> List[CatalogEntry]:
        """Return all entries of a given category."""
        return [e for e in self._entries if e.category == category]

    def get_compact_catalog(self) -> str:
        """Compact version for LLM prompt: name + one-line description.

        Format optimized for token efficiency::

            [skills]
            - bulk-deg-analysis: Bulk RNA-seq DEG pipeline with DESeq2
            - single-clustering: Single-cell clustering workflow
            [mcp_tools]
            - string_interaction_partners: Protein-protein interactions from STRING
            [registry_functions]
            (available via function registry - see system prompt)

        Returns
        -------
        str
            Compact catalog text suitable for LLM prompt injection.
        """
        if self._compact_cache is not None:
            return self._compact_cache

        sections: List[str] = []

        # Skills
        skills = sorted(
            [e for e in self._entries if e.category == "skill"],
            key=lambda e: e.name.lower(),
        )
        if skills:
            lines = ["[skills]"]
            for e in skills:
                lines.append(f"- {e.slug}: {e.description}")
            sections.append("\n".join(lines))

        # MCP tools
        mcp_tools = sorted(
            [e for e in self._entries if e.category == "mcp_tool"],
            key=lambda e: e.name.lower(),
        )
        if mcp_tools:
            lines = ["[mcp_tools]"]
            for e in mcp_tools:
                lines.append(f"- {e.slug}: {e.description}")
            sections.append("\n".join(lines))

        # Registry functions (compact summary only)
        reg_entries = [e for e in self._entries if e.category == "registry_function"]
        if reg_entries:
            sections.append(
                "[registry_functions]\n"
                "(available via function registry - see system prompt)"
            )

        self._compact_cache = "\n\n".join(sections)
        return self._compact_cache

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        skills = sum(1 for e in self._entries if e.category == "skill")
        mcp = sum(1 for e in self._entries if e.category == "mcp_tool")
        reg = sum(1 for e in self._entries if e.category == "registry_function")
        return f"ToolCatalog(skills={skills}, mcp_tools={mcp}, registry={reg})"
