# OmicVerse Agent: MCP Integration & LLM-Driven Tool Selection Plan

## 1. Current Architecture Analysis

### What Exists Today

**OmicVerse Agent (`smart_agent.py` — `OmicVerseAgent`)**
- Two-tier priority system:
  - **Priority 1 (Fast)**: Registry-only workflow for simple tasks (single function calls)
  - **Priority 2 (Comprehensive)**: Skills-guided workflow for complex multi-step tasks
- LLM-based task analysis (`_analyze_task_complexity`) — classifies requests as simple/complex + whether MCP is needed
- Multi-provider support: OpenAI, Anthropic, Google, DashScope

**Skill System (`skill_registry.py`)**
- 24+ skills in `.claude/skills/` (SKILL.md files with YAML frontmatter)
- `SkillRegistry`: Progressive disclosure — loads metadata at startup, full body on-demand
- `SkillRouter`: **Keyword-based** cosine similarity matching (TF-IDF over tokens) — **This is what the user wants to replace**
- `_select_skill_matches_llm()`: Already exists! Uses LLM to pick skills from a catalog. But only used in Priority 2

**MCP Integration (already partially built)**
- `MCPClientManager` (`mcp_client.py`): Generic MCP client supporting HTTP + stdio transports
- `BioContextBridge` (`biocontext_bridge.py`): Pre-configured wrapper for BioContext.ai
- `MCPConfig` in `agent_config.py`: Configuration with `enable_biocontext` (auto/yes/no)
- `biocontext-mcp` skill in `.claude/skills/`
- MCP auto-detection via LLM in `_analyze_task_complexity`
- Lazy connection: BioContext connects on demand in "auto" mode

**OvIntelligence (RAG backend)**
- `rag_mcp_server.py`: MCP server wrapping the two-stage RAG system for code search
- `rag_adk_agent.py`: Google ADK agent version

### Core Problems

1. **`SkillRouter` is keyword-based** — Uses cosine similarity on token frequencies. Semantically shallow and fragile. The user correctly identifies this as "not intelligent."

2. **`_select_skill_matches_llm` exists but is underutilized** — Only called in Priority 2. Priority 1 skips skills entirely. The `stream_async` path still uses the old keyword router via `_select_skill_matches`.

3. **MCP and Skills are separate decision paths** — The LLM decides "needs_mcp?" separately from "which skill?" in different prompts. This wastes tokens and creates inconsistency.

4. **No unified tool selection** — Skills, MCP tools, and registry functions are three separate concerns managed with hardcoded routing logic.

5. **BioContext is the only MCP server** — No easy way to add servers from the 52+ on biocontext.ai registry.

6. **Two separate LLM calls for analysis** — `_analyze_task_complexity` (complexity + MCP) and `_select_skill_matches_llm` (skills) should be one call.

---

## 2. Proposed Architecture: Unified LLM-Driven Tool Selection

### Core Principle
> **The LLM decides everything.** No keyword matching, no cosine similarity, no hardcoded pattern detection. The agent presents a unified "tool catalog" to the LLM and lets it decide which tools (skills, MCP tools, registry functions) to use in a **single LLM call**.

### Architecture Overview

```
User Request
    │
    ▼
┌─────────────────────────────────┐
│  Unified Tool Catalog (UTC)     │  ← Single source of truth
│                                 │
│  ┌─────────┐ ┌──────┐ ┌──────┐ │
│  │ Skills  │ │ MCP  │ │ Reg. │ │
│  │ (24+)   │ │Tools │ │Funcs │ │
│  └─────────┘ └──────┘ └──────┘ │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  LLM Tool Selector              │  ← Single LLM call replaces
│  "Given this request, which     │     _analyze_task_complexity +
│   tools/skills do you need?"    │     _select_skill_matches_llm
│                                 │
│  Returns: {                     │
│    complexity: simple|complex,  │
│    skills: ["slug1", "slug2"],  │
│    mcp_tools: ["tool1"],        │
│    needs_external_db: true,     │
│    reasoning: "..."             │
│  }                              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Execution Engine               │
│  - Lazy-load selected skills    │
│  - Lazy-connect required MCP    │
│  - Generate & execute code      │
└─────────────────────────────────┘
```

---

## 3. Implementation Plan

### Phase 1: Unified Tool Catalog

**New file: `omicverse/utils/tool_catalog.py`**

A single registry that unifies Skills + MCP tools + Registry functions:

```python
@dataclass
class CatalogEntry:
    name: str
    slug: str
    category: str          # "skill" | "mcp_tool" | "registry_function"
    description: str
    source: str            # "builtin_skill" | "biocontext" | "custom_mcp" | "registry"
    requires_connection: bool  # True for MCP tools not yet connected

class ToolCatalog:
    """Unified catalog of all available tools for LLM selection."""

    def __init__(self, skill_registry, mcp_manager, function_registry):
        self._entries: List[CatalogEntry] = []
        self._build_from_skills(skill_registry)
        self._build_from_mcp(mcp_manager)
        self._build_from_registry(function_registry)

    def get_compact_catalog(self) -> str:
        """Compact version for LLM prompt: name + one-line description.

        Format optimized for token efficiency:
        [skills]
        - bulk-deg-analysis: Bulk RNA-seq DEG pipeline with DESeq2
        - single-clustering: Single-cell clustering workflow
        [mcp_tools]
        - string_interaction_partners: Protein-protein interactions from STRING
        - panglao_cell_markers: Cell-type markers from PanglaoDB
        [functions]
        (omitted from compact — too many, already in system prompt)
        """

    def get_entries_by_slugs(self, slugs: List[str]) -> List[CatalogEntry]:
        """Look up catalog entries by their slugs."""
```

### Phase 2: LLM Tool Selector

**New file: `omicverse/utils/llm_tool_selector.py`**

Replaces both `_analyze_task_complexity` and `_select_skill_matches_llm` with a **single** LLM call:

```python
@dataclass
class ToolSelectionResult:
    complexity: str                    # "simple" | "complex"
    selected_skills: List[str]         # skill slugs
    needs_external_db: bool            # whether to lazy-connect MCP
    reasoning: str                     # LLM's brief rationale

class LLMToolSelector:
    """LLM-driven tool selection replacing keyword routing."""

    def __init__(self, catalog: ToolCatalog, llm_backend):
        self._catalog = catalog
        self._llm = llm_backend
        self._cached_catalog_prompt = catalog.get_compact_catalog()

    async def select(self, request: str, adata_summary: str) -> ToolSelectionResult:
        """Single LLM call to analyze request and select tools.

        The prompt includes the compact catalog and asks the LLM to return
        a structured JSON response.
        """
        prompt = f"""You are a bioinformatics tool selector for OmicVerse Agent.
Given a user request and dataset state, decide what tools are needed.

Request: "{request}"
Dataset: {adata_summary}

{self._cached_catalog_prompt}

Respond as JSON only:
{{
  "complexity": "simple" or "complex",
  "skills": ["skill-slug-1", "skill-slug-2"],
  "needs_external_db": true/false,
  "reasoning": "brief explanation"
}}

Rules:
- "simple": single operation (QC, normalize, cluster, plot)
- "complex": multi-step workflow, pipeline, or vague request
- "skills": 0-2 most relevant skill slugs from the catalog
- "needs_external_db": true only when external databases needed
  (protein interactions, pathway lookup, literature search, etc.)
"""
        response = await self._llm.run(prompt)
        return self._parse_response(response)
```

### Phase 3: Refactor `OmicVerseAgent.run_async`

**Modify: `omicverse/utils/smart_agent.py`**

Replace the multi-step analysis with the unified selector:

```python
async def run_async(self, request, adata):
    # 1. Single LLM call for everything: complexity + skills + MCP need
    selection = await self._tool_selector.select(request, self._summarize_adata(adata))

    # 2. Lazy-load resources based on LLM's decision
    if selection.needs_external_db:
        self._lazy_connect_biocontext()

    if selection.selected_skills:
        skill_guidance = self._load_selected_skills(selection.selected_skills)

    # 3. Execute based on complexity
    if selection.complexity == "simple":
        return await self._run_registry_workflow(request, adata)
    else:
        return await self._run_skills_workflow(request, adata, skill_guidance)
```

Key changes:
- Remove `_analyze_task_complexity` (or keep as offline fallback)
- Remove `_select_skill_matches` (keyword-based, already deprecated)
- Combine `_select_skill_matches_llm` into `LLMToolSelector.select`
- Build `ToolCatalog` once at `__init__` time, not per-request
- `stream_async` also uses `LLMToolSelector` instead of old `_select_skill_matches`

### Phase 4: MCP Server Registry

**New file: `omicverse/utils/mcp_registry.py`**

Enable users to discover and add MCP servers from the BioContext.ai registry:

```python
class MCPServerRegistry:
    """Registry of available MCP servers (lazy — not connected by default)."""

    BUILTIN_SERVERS = {
        "biocontext": {
            "url": "https://mcp.biocontext.ai/mcp/",
            "description": "STRING, UniProt, KEGG, Reactome, PanglaoDB, etc.",
            "category": "knowledgebase",
        },
    }

    def register(self, name, url=None, command=None, description=""):
        """Register an MCP server for lazy connection."""

    def from_biocontext_registry(self, server_slug):
        """Fetch server config from biocontext.ai registry."""

    def get_tool_preview(self) -> List[CatalogEntry]:
        """Return tool previews for unconnected servers (for the catalog)."""
```

### Phase 5: Deprecate SkillRouter

**Modify: `omicverse/utils/skill_registry.py`**

```python
class SkillRouter:
    """DEPRECATED: Use LLMToolSelector instead.

    This keyword-based router used cosine similarity on token frequencies.
    It has been replaced by LLM-driven tool selection which understands
    semantic intent rather than keyword overlap.
    """
    def __init__(self, ...):
        warnings.warn(
            "SkillRouter is deprecated. Use LLMToolSelector for intelligent "
            "skill matching. SkillRouter will be removed in a future version.",
            DeprecationWarning, stacklevel=2,
        )
        ...
```

### Phase 6: Update AgentConfig

**Modify: `omicverse/utils/agent_config.py`**

```python
@dataclass
class MCPConfig:
    servers: List[Dict[str, Any]] = field(default_factory=list)
    enable_biocontext: str = "auto"
    biocontext_mode: str = "remote"
    cache_ttl: int = 3600
    inject_tools_in_prompt: bool = True
    # NEW
    auto_discover_servers: bool = False
    registry_url: str = "https://biocontext.ai/api/registry"

@dataclass
class SelectionConfig:
    """Tool selection strategy configuration."""
    use_llm_selector: bool = True        # Use LLM for tool selection
    fallback_to_keywords: bool = True    # Fall back to SkillRouter if LLM fails
```

---

## 4. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `omicverse/utils/tool_catalog.py` | **NEW** | Unified tool catalog |
| `omicverse/utils/llm_tool_selector.py` | **NEW** | LLM-driven selection (replaces keyword routing) |
| `omicverse/utils/mcp_registry.py` | **NEW** | MCP server discovery registry |
| `omicverse/utils/smart_agent.py` | **MODIFY** | Use LLMToolSelector in run_async and stream_async |
| `omicverse/utils/skill_registry.py` | **MODIFY** | Deprecate SkillRouter |
| `omicverse/utils/agent_config.py` | **MODIFY** | Add SelectionConfig, extend MCPConfig |
| `omicverse/utils/mcp_client.py` | **MODIFY** | Add lazy connection support |
| `omicverse/utils/biocontext_bridge.py` | **MODIFY** | Integrate with MCPServerRegistry |
| `tests/utils/test_tool_catalog.py` | **NEW** | Tests for ToolCatalog |
| `tests/utils/test_llm_tool_selector.py` | **NEW** | Tests for LLMToolSelector |

---

## 5. Migration Strategy

1. **Additive first** — Old code paths remain functional; new paths are opt-in via feature flag
2. **Feature flag**: `use_llm_selector: bool = True` in AgentConfig (default True)
3. **`SkillRouter`** marked deprecated but not removed — backward compat
4. **`_analyze_task_complexity`** kept as offline fallback when LLM unreachable
5. **Tests for both paths** — old keyword routing tests still pass, new LLM routing tested with mocks

---

## 6. Key Design Decisions

1. **Single LLM call for selection** — Replacing two separate calls (complexity + skills + MCP) saves ~50% tokens on the analysis step
2. **Catalog is pre-built at init** — No per-request catalog construction. Rebuilt only when MCP servers connect/disconnect
3. **Lazy MCP connections** — Only connect when the LLM actually selects MCP tools
4. **Progressive disclosure** — Skill bodies loaded only when LLM selects them
5. **Fallback to keywords** — Only when LLM backend is unavailable (network error, no API key)
6. **LLM sees tool descriptions, not tool implementations** — The compact catalog is ~2K tokens, not 50K

---

## 7. User-Facing API (No Change Needed)

The existing API already supports everything:

```python
import omicverse as ov

# Auto mode — LLM decides when to use MCP (default)
agent = ov.Agent(model="gemini-3-flash-preview")
result = agent.run("find TP53 interaction partners and overlap with DEGs", adata)

# Explicit BioContext
agent = ov.Agent(model="gemini-3-flash-preview", enable_biocontext=True)

# Custom MCP servers
agent = ov.Agent(
    model="gemini-3-flash-preview",
    mcp_servers=[
        {"name": "biocontext", "url": "https://mcp.biocontext.ai/mcp/"},
        {"name": "scmcp", "url": "https://scmcp.example.com/mcp/"},
    ],
)
```

---

## 8. Implementation Order

| Priority | Step | Description |
|----------|------|-------------|
| P0 | Phase 1 | `ToolCatalog` — unified catalog class |
| P0 | Phase 2 | `LLMToolSelector` — single LLM call for all decisions |
| P0 | Phase 3 | Refactor `run_async` to use new selector |
| P1 | Phase 5 | Deprecate `SkillRouter` |
| P1 | Phase 6 | Update `AgentConfig` with SelectionConfig |
| P2 | Phase 4 | `MCPServerRegistry` for multi-server discovery |
| P2 | Tests | Full test coverage for new modules |
