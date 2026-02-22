"""
MCP Context Enricher for OmicVerse Agent.

Bridges the :class:`BioContextMCPClient` with the agent's LLM workflow.
Before the agent generates analysis code, the enricher:

1. Asks the LLM which MCP tools (if any) would provide useful biological
   context for the user's request.
2. Executes those tool calls against the connected MCP servers.
3. Formats the results into a concise context block that is injected into the
   code-generation prompt.

This is a **non-blocking enrichment** step — failures are caught and logged
so the agent can always fall back to code generation without external context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .mcp_client import BioContextMCPClient, MCPToolInfo, MCPToolResult
from .mcp_config import MCPConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: token approximation
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

class MCPContextEnricher:
    """Uses the agent's LLM to select and call MCP tools, then formats the
    results as context for code generation.

    Parameters
    ----------
    mcp_client : BioContextMCPClient
        Connected MCP client.
    llm : object
        The agent's LLM backend (must have an async ``run(prompt)`` method).
    config : MCPConfig
        MCP settings (token budget, max tools, etc.).
    """

    def __init__(
        self,
        mcp_client: BioContextMCPClient,
        llm: Any,
        config: MCPConfig,
    ) -> None:
        self._client = mcp_client
        self._llm = llm
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich(
        self,
        request: str,
        adata_meta: Dict[str, Any],
    ) -> str:
        """Return a formatted context string to inject into the code-generation
        prompt, or an empty string if no enrichment is needed / available.

        Parameters
        ----------
        request : str
            The user's natural-language request.
        adata_meta : dict
            Lightweight metadata about the AnnData object (shape, var_names
            sample, obs columns, etc.).

        Returns
        -------
        str
            Formatted biological context block, or ``""``.
        """
        # Step 1: discover available tools
        try:
            tools = await self._client.list_all_tools()
        except Exception as exc:
            logger.warning("MCP tool discovery failed: %s", exc)
            return ""

        if not tools:
            return ""

        # Step 2: ask LLM which tools to call
        try:
            planned_calls = await self._select_tools(request, adata_meta, tools)
        except Exception as exc:
            logger.warning("MCP tool selection failed: %s", exc)
            return ""

        if not planned_calls:
            return ""

        # Step 3: execute tool calls
        results = await self._execute_calls(planned_calls)

        # Step 4: format into context
        successful = [r for r in results if not r.is_error]
        if not successful:
            return ""

        return self._format_context(successful)

    # ------------------------------------------------------------------
    # Internal: LLM-based tool selection
    # ------------------------------------------------------------------

    async def _select_tools(
        self,
        request: str,
        adata_meta: Dict[str, Any],
        tools: List[MCPToolInfo],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which MCP tools to call and with what arguments.

        Returns a list of dicts, each with keys:
        ``server_name``, ``tool_name``, ``arguments``.
        """
        # Build a compact tool catalogue
        catalogue_lines: List[str] = []
        for t in tools:
            schema_summary = ""
            if t.input_schema:
                props = t.input_schema.get("properties", {})
                if props:
                    params = ", ".join(
                        f"{k}: {v.get('type', 'any')}"
                        for k, v in list(props.items())[:6]
                    )
                    schema_summary = f" | params: ({params})"
            catalogue_lines.append(
                f"- [{t.server_name}] {t.tool_name}: {t.description[:120]}{schema_summary}"
            )

        catalogue = "\n".join(catalogue_lines)

        # Build adata description
        shape_str = f"{adata_meta.get('shape', ('?', '?'))}"
        genes_sample = adata_meta.get("var_names", [])[:10]
        obs_cols = adata_meta.get("obs_columns", [])

        prompt = f"""You are a biological context advisor. A user is about to run a bioinformatics analysis.
Decide whether querying external biological databases would help, and if so, which MCP tools to call.

User request: "{request}"

Dataset info:
- Shape: {shape_str}
- Sample gene names: {genes_sample}
- Obs columns: {obs_cols}

Available MCP tools (server/tool_name: description):
{catalogue}

Rules:
1. Only select tools that provide genuinely useful biological context for this request.
2. Select at most {self._config.max_tools_per_query} tools.
3. If the request is purely computational (e.g. "run PCA", "cluster cells") and needs no external biology, return an empty list.
4. Provide concrete arguments for each tool call based on the request and dataset.

Respond with ONLY a JSON array. Each element must have:
  {{"server_name": "...", "tool_name": "...", "arguments": {{...}}}}

If no tools are needed, return: []"""

        response = await self._llm.run(prompt)

        # Parse JSON array from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            calls = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse MCP tool selection JSON")
            return []

        # Validate and cap
        valid: List[Dict[str, Any]] = []
        known_servers = set(self._client.server_names)
        known_tools = {t.tool_name for t in tools}

        for call in calls[: self._config.max_tools_per_query]:
            sn = call.get("server_name", "")
            tn = call.get("tool_name", "")
            args = call.get("arguments", {})
            if sn in known_servers and tn in known_tools and isinstance(args, dict):
                valid.append({"server_name": sn, "tool_name": tn, "arguments": args})

        return valid

    # ------------------------------------------------------------------
    # Internal: execute calls
    # ------------------------------------------------------------------

    async def _execute_calls(
        self,
        planned_calls: List[Dict[str, Any]],
    ) -> List[MCPToolResult]:
        """Run planned tool calls via the MCP client."""
        return await self._client.call_tools_parallel(planned_calls)

    # ------------------------------------------------------------------
    # Internal: format context
    # ------------------------------------------------------------------

    def _format_context(self, results: List[MCPToolResult]) -> str:
        """Format successful MCP results into a prompt-friendly block.

        Respects the configured token budget by truncating per-result and
        in aggregate.
        """
        budget = self._config.max_context_tokens
        blocks: List[str] = []
        total_tokens = 0

        for r in results:
            content_str = str(r.content or "")

            # Per-result budget: fair share of the total
            per_result_budget = budget // max(len(results), 1)
            if _approx_tokens(content_str) > per_result_budget:
                # Truncate to fit
                char_limit = per_result_budget * 4
                content_str = content_str[:char_limit] + "... [truncated]"

            block = (
                f"[{r.server_name}/{r.tool_name}]\n"
                f"{content_str}"
            )
            block_tokens = _approx_tokens(block)

            if total_tokens + block_tokens > budget:
                # Would exceed budget — stop adding
                break

            blocks.append(block)
            total_tokens += block_tokens

        if not blocks:
            return ""

        header = "Biological Context (from external knowledge databases via MCP):"
        return header + "\n\n" + "\n\n".join(blocks)
