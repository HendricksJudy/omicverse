# How Registry Functions Work with ov.agent

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Complete Workflow](#complete-workflow)
3. [Function Registration](#function-registration)
4. [Agent Integration](#agent-integration)
5. [Priority System](#priority-system)
6. [Usage Examples](#usage-examples)
7. [Advanced Features](#advanced-features)

---

## Architecture Overview

The OmicVerse agent system connects natural language requests to registered functions through a sophisticated registry and LLM-based routing system.

```
┌─────────────────┐
│  User Request   │ "quality control with nUMI>500, mito<0.2"
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────────────────────┐
│                     OmicVerseAgent                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Task Complexity Analysis                          │  │
│  │     - Simple (1-3 functions) → Priority 1            │  │
│  │     - Complex (workflow) → Priority 2                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  2. Function Registry Lookup (_global_registry)       │  │
│  │     - Query by alias: "qc", "质控", "quality_control" │  │
│  │     - Get function metadata, signature, examples      │  │
│  │     - Multi-language search support                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3. LLM Code Generation                               │  │
│  │     - Inject registry info into prompt                │  │
│  │     - Extract parameters from natural language        │  │
│  │     - Generate executable Python code                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  4. Code Validation (Reflection)                      │  │
│  │     - Review generated code                           │  │
│  │     - Check for errors/improvements                   │  │
│  │     - Iterate if needed                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  5. Execution                                          │  │
│  │     - Sandboxed code execution                        │  │
│  │     - Return modified adata                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  6. Result Review                                      │  │
│  │     - Validate output matches intent                  │  │
│  │     - Report confidence score                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────┐
│  Modified adata │
└─────────────────┘
```

---

## Complete Workflow

### Step-by-Step: How a User Request Becomes Executable Code

**User Request**: `"quality control with nUMI>500, mito<0.2"`

#### Step 1: Agent Initialization
```python
import omicverse as ov

agent = ov.Agent(model="gpt-4o", api_key="your-key")
# Internally: agent._setup_agent() loads all registered functions
```

#### Step 2: Registry Loading
```python
# In _get_available_functions_info():
functions_info = []
for entry in _global_registry._registry.values():
    functions_info.append({
        'name': 'qc',
        'full_name': 'omicverse.pp.qc',
        'description': 'Perform quality control on single-cell data',
        'aliases': ['质控', 'qc', 'quality_control'],
        'category': 'preprocessing',
        'signature': '(adata, tresh={...}, ...)',
        'examples': [...]
    })
```

#### Step 3: Prompt Construction
```python
# Agent builds this prompt dynamically:
instructions = """
You are an intelligent OmicVerse assistant.

## Available OmicVerse Functions

[
  {
    "name": "qc",
    "full_name": "omicverse.pp.qc",
    "description": "Perform quality control...",
    "aliases": ["质控", "qc", "quality_control"],
    "signature": "(adata, tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250})",
    "examples": [
      "ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})"
    ]
  },
  ... (all other registered functions)
]

## Your Task
When given request "quality control with nUMI>500, mito<0.2":
1. Find function with aliases containing "qc" or "quality"
2. Extract parameters: nUMI>500 → tresh['nUMIs']=500, mito<0.2 → tresh['mito_perc']=0.2
3. Generate Python code using exact signature from registry
"""
```

#### Step 4: LLM Code Generation
```python
# LLM generates:
generated_code = """
import omicverse as ov
adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})
print(f"QC completed: {adata.shape[0]} cells × {adata.shape[1]} genes")
"""
```

#### Step 5: Code Validation (Reflection)
```python
# Agent reviews the code:
reflection_prompt = f"""
Review this generated code for the request "{request}":

{generated_code}

Check for:
- Correct function usage
- Parameter extraction accuracy
- Syntax errors
- Missing imports

Return: {{'needs_revision': False, 'confidence': 0.95, 'issues_found': []}}
"""
```

#### Step 6: Execution
```python
# Agent executes in sandboxed environment:
result_adata = agent._execute_generated_code(generated_code, adata)
# Returns modified adata
```

---

## Function Registration

### How to Register a New Function

**File**: `omicverse/utils/_cluster.py` (example)

```python
from omicverse.utils.registry import register_function

@register_function(
    aliases=[
        "聚类",              # Chinese
        "cluster",           # English
        "clustering",        # Alternative
        "细胞聚类",          # Chinese variant
        "单细胞聚类"         # Chinese single-cell
    ],
    category="utils",
    description="Perform clustering using various algorithms including Leiden, Louvain, GMM, K-means, and scICE",
    examples=[
        "# Leiden clustering (recommended)",
        "sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)",
        "ov.utils.cluster(adata, method='leiden', resolution=1.0)",
        "# Gaussian Mixture Model clustering",
        "ov.utils.cluster(adata, method='GMM', n_components=10)",
    ],
    related=["pp.neighbors", "pl.embedding", "utils.refine_label"]
)
def cluster(adata: anndata.AnnData,
            method: str = 'leiden',
            use_rep: str = 'X_pca',
            random_state: int = 1024,
            n_components = None,
            **kwargs):
    """
    Perform clustering using various algorithms.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    method : str
        Clustering algorithm: 'leiden', 'louvain', 'GMM', 'kmeans'
    use_rep : str
        Representation to use for clustering
    ...

    Returns
    -------
    None or model object
        Adds cluster labels to adata.obs
    """

    if method == 'leiden':
        sc.tl.leiden(adata, **kwargs)
    elif method == 'louvain':
        sc.tl.louvain(adata, **kwargs)
    # ... implementation
```

### What Happens During Registration

1. **Decorator Execution**: `@register_function` is called immediately when the module loads
2. **Metadata Extraction**:
   ```python
   # Inside FunctionRegistry.register():
   sig = inspect.signature(func)  # Get signature
   docstring = inspect.getdoc(func)  # Get documentation
   module_name = func.__module__  # "omicverse.utils._cluster"
   func_name = func.__name__  # "cluster"
   full_name = f"{module_name}.{func_name}"
   ```

3. **Multi-Alias Indexing**:
   ```python
   # Creates multiple index entries:
   _registry["聚类"] = entry
   _registry["cluster"] = entry
   _registry["clustering"] = entry
   _registry["cluster"] = entry  # function name
   _registry["omicverse.utils._cluster.cluster"] = entry  # full name
   ```

4. **Category Mapping**:
   ```python
   _categories["utils"] = [..., "omicverse.utils._cluster.cluster", ...]
   ```

### Registry Entry Structure

```python
entry = {
    'function': <function cluster at 0x...>,
    'full_name': 'omicverse.utils._cluster.cluster',
    'short_name': 'cluster',
    'module': 'omicverse.utils._cluster',
    'aliases': ['聚类', 'cluster', 'clustering', '细胞聚类', '单细胞聚类'],
    'category': 'utils',
    'description': 'Perform clustering using various algorithms...',
    'examples': [...],
    'related': ['pp.neighbors', 'pl.embedding', 'utils.refine_label'],
    'signature': '(adata, method=\'leiden\', use_rep=\'X_pca\', ...)',
    'parameters': ['adata', 'method=\'leiden\'', 'use_rep=\'X_pca\'', ...],
    'docstring': 'Perform clustering using various algorithms...'
}
```

---

## Agent Integration

### How Agent Uses the Registry

#### Method 1: Direct Registry Query
```python
# In smart_agent.py:
def _search_functions(self, query: str) -> str:
    """Search for functions matching query."""
    results = _global_registry.find(query)  # Uses difflib for fuzzy matching
    return json.dumps({
        "found": len(results),
        "functions": results
    })
```

#### Method 2: System Prompt Injection
```python
# In _setup_agent():
functions_info = self._get_available_functions_info()

instructions = f"""
You are an intelligent OmicVerse assistant.

## Available OmicVerse Functions

{functions_info}  # ← ENTIRE REGISTRY INJECTED HERE

## Your Task
When given a natural language request:
1. Analyze the request
2. Find the most appropriate function from above
3. Extract parameters
4. Generate Python code
"""
```

#### Method 3: Parameter Extraction Rules
```python
# Agent prompt includes parameter extraction patterns:
"""
## Parameter Extraction Rules

Extract parameters dynamically:
- "nUMI>X", "umi>X" → tresh={'nUMIs': X, ...}
- "mito<X" → tresh={'mito_perc': X}
- "genes>X" → tresh={'detected_genes': X}
- "resolution=X" → resolution=X
- "n_pcs=X" → n_pcs=X
"""
```

---

## Priority System

### Two-Tier Execution Strategy

The agent intelligently routes requests based on complexity:

#### **Priority 1: Fast Registry-Only** (60-70% faster)

**Use Case**: Simple tasks requiring 1-3 function calls

**Characteristics**:
- Uses ONLY function registry (no skill guidance)
- Single LLM call
- Optimized for direct parameter extraction
- 50% lower token usage

**Example Requests**:
- "quality control with nUMI>500"
- "PCA with 50 components"
- "leiden clustering resolution=1.0"

**Workflow**:
```python
async def _run_priority1(self, request: str, adata) -> anndata.AnnData:
    # Build registry-only prompt
    functions_info = self._get_available_functions_info()

    prompt = f"""
    Request: "{request}"
    Available Functions: {functions_info}

    INSTRUCTIONS:
    - Generate 1-3 function calls maximum
    - No complex workflows
    - Direct parameter mapping
    """

    # Single LLM call
    code = await self._llm.run(prompt)

    # Optional reflection
    if self.enable_reflection:
        code = await self._reflect_on_code(code, request, adata)

    # Execute
    result = self._execute_generated_code(code, adata)

    return result
```

#### **Priority 2: Skills-Guided** (Comprehensive)

**Use Case**: Complex tasks requiring multi-step workflows

**Characteristics**:
- Matches relevant skills using LLM
- Loads full skill guidance (lazy loading)
- Injects both registry + skills
- Generates multi-step pipelines

**Example Requests**:
- "complete bulk RNA-seq DEG pipeline"
- "full single-cell preprocessing workflow"
- "analyze and annotate cell types"

**Workflow**:
```python
async def _run_priority2(self, request: str, adata) -> anndata.AnnData:
    # Step 1: Match skills
    matched_skills = await self._select_skill_matches_llm(request, top_k=2)

    # Step 2: Load skill guidance
    skill_guidance = self._format_skill_guidance(matched_skills)

    # Step 3: Build comprehensive prompt
    functions_info = self._get_available_functions_info()

    prompt = f"""
    Request: "{request}"
    Available Functions: {functions_info}
    Skill Guidance: {skill_guidance}

    INSTRUCTIONS:
    - Generate complete multi-step workflow
    - Follow skill best practices
    - Use registry functions for implementation
    """

    # Generate code
    code = await self._llm.run(prompt)

    # Multi-iteration reflection
    for i in range(self.reflection_iterations):
        code = await self._reflect_on_code(code, request, adata)

    # Execute
    result = self._execute_generated_code(code, adata)

    return result
```

### Task Complexity Classification

The agent determines which priority to use:

```python
def _classify_task_complexity(self, request: str) -> str:
    """
    Classify request as 'simple' or 'complex'.

    Simple indicators:
    - Single action verbs: "qc", "cluster", "normalize"
    - 1-3 parameters
    - No workflow keywords

    Complex indicators:
    - "complete", "full", "pipeline", "workflow"
    - Multiple steps: "and then", "followed by"
    - Ambiguous requirements
    """

    # Pattern matching + LLM reasoning
    if any(word in request.lower() for word in ['complete', 'pipeline', 'workflow', 'full']):
        return 'complex'

    # LLM-based classification for ambiguous cases
    classification = await self._llm.run(f"Classify this task: {request}")
    return classification  # 'simple' or 'complex'
```

---

## Usage Examples

### Example 1: Simple QC (Priority 1)

```python
import omicverse as ov
import scanpy as sc

# Load data
adata = sc.datasets.pbmc3k()

# Initialize agent
agent = ov.Agent(model="gpt-4o", api_key="your-key")

# Natural language request
result = agent.run("quality control with nUMI>500, mito<0.2", adata)

# Output:
# 🚀 Priority 1: Fast registry-based workflow
#    💭 Generating code with registry functions only...
#    🧬 Generated code:
#    ----------------------------------------------
#    import omicverse as ov
#    adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})
#    print(f"QC completed: {adata.shape[0]} cells × {adata.shape[1]} genes")
#    ----------------------------------------------
#    🔍 Validating code...
#       ✅ Code validated (confidence: 95.0%)
#    ⚡ Executing code...
#    ✅ Execution successful!
#    📊 Result: 2638 cells × 1838 genes
```

**What Happened**:
1. Agent identified "quality control" → searched registry for "qc"
2. Extracted parameters: nUMI>500 → `tresh['nUMIs']=500`, mito<0.2 → `tresh['mito_perc']=0.2`
3. Generated code using function signature from registry
4. Validated code with reflection
5. Executed in sandboxed environment

### Example 2: Clustering (Priority 1)

```python
# After preprocessing...
result = agent.run("leiden clustering resolution=1.0", adata)

# Generated code:
# import omicverse as ov
# ov.utils.cluster(adata, method='leiden', resolution=1.0)
# print(f"Clustering completed: {len(adata.obs['leiden'].unique())} clusters")
```

### Example 3: Chinese Language (Priority 1)

```python
# Chinese request
result = agent.run("质控 nUMI>500", adata)

# Agent searches registry aliases: ['质控', 'qc', 'quality_control']
# Finds omicverse.pp.qc
# Generates same code as English request
```

### Example 4: Complex Workflow (Priority 2)

```python
# Complex multi-step request
result = agent.run(
    "complete single-cell preprocessing: qc, normalization, HVG selection, PCA, and clustering",
    adata
)

# Output:
# 🧠 Priority 2: Skills-guided workflow for complex tasks
#    🎯 Matching relevant skills...
#    📚 Loading skill guidance:
#       - single-preprocessing
#    💭 Generating multi-step workflow code...
#    🧬 Generated workflow code:
#    ==============================================
#    import omicverse as ov
#    import scanpy as sc
#
#    # Step 1: Quality control
#    adata = ov.pp.qc(adata, tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250})
#
#    # Step 2: Normalization
#    sc.pp.normalize_total(adata, target_sum=1e4)
#    sc.pp.log1p(adata)
#
#    # Step 3: Highly variable genes
#    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#
#    # Step 4: PCA
#    adata = ov.pp.pca(adata, n_comps=50)
#
#    # Step 5: Clustering
#    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
#    ov.utils.cluster(adata, method='leiden', resolution=1.0)
#
#    print(f"Workflow completed: {adata.shape[0]} cells × {len(adata.var['highly_variable'].sum())} HVGs")
#    ==============================================
```

### Example 5: Parameter Variations

```python
# Different ways to express the same thing:

# English
agent.run("qc with nUMI>500", adata)

# Chinese
agent.run("质控 nUMI>500", adata)

# Full words
agent.run("quality control minimum UMI 500", adata)

# Abbreviations
agent.run("qc nUMI>500", adata)

# All generate the same code!
```

---

## Advanced Features

### 1. Reflection (Code Validation)

**Purpose**: Review and improve generated code before execution

```python
agent = ov.Agent(
    model="gpt-4o",
    enable_reflection=True,      # Enable validation
    reflection_iterations=1      # Number of review cycles
)
```

**How It Works**:
```python
async def _reflect_on_code(self, code, request, adata, iteration):
    """Review generated code for issues."""

    reflection_prompt = f"""
    Review this code for request "{request}":

    {code}

    Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes

    Check for:
    1. Correct function usage
    2. Parameter extraction accuracy
    3. Syntax errors
    4. Missing imports
    5. Logic errors

    Return JSON:
    {{
        "needs_revision": true/false,
        "confidence": 0.0-1.0,
        "issues_found": ["issue1", "issue2"],
        "improved_code": "...",
        "explanation": "..."
    }}
    """

    result = await self._llm.run(reflection_prompt)
    return result
```

### 2. Result Review (Output Validation)

**Purpose**: Validate that execution result matches user intent

```python
agent = ov.Agent(
    model="gpt-4o",
    enable_result_review=True    # Enable output validation
)
```

**How It Works**:
```python
async def _review_result(self, original_adata, result_adata, request, code):
    """Validate result matches intent."""

    review_prompt = f"""
    Request: "{request}"
    Code executed: {code}

    Original: {original_adata.shape[0]} cells × {original_adata.shape[1]} genes
    Result: {result_adata.shape[0]} cells × {result_adata.shape[1]} genes

    Does the result match the user's intent?

    Return JSON:
    {{
        "matched": true/false,
        "confidence": 0.0-1.0,
        "issues": ["issue1", "issue2"],
        "explanation": "..."
    }}
    """

    result = await self._llm.run(review_prompt)
    return result
```

### 3. Token Usage Tracking

```python
agent = ov.Agent(model="gpt-4o")
result = agent.run("qc with nUMI>500", adata)

# Check token usage
print(agent.last_usage)
# Usage(input_tokens=1234, output_tokens=567, total_tokens=1801, model='gpt-4o', provider='openai')

# Detailed breakdown
print(agent.last_usage_breakdown)
# {
#     'generation': Usage(...),
#     'reflection': [Usage(...)],
#     'review': [Usage(...)],
#     'total': Usage(...)
# }
```

### 4. Registry Search API

```python
import omicverse as ov

# Search functions
results = ov.find_function("质控")
# Returns: [{'name': 'qc', 'full_name': 'omicverse.pp.qc', ...}]

# List all functions
all_funcs = ov.list_functions()

# List by category
preprocessing_funcs = ov.list_functions("preprocessing")

# Get help
ov.help("qc")
# Prints: Full documentation for ov.pp.qc

# Get recommendations
recommendations = ov.recommend_function("我想过滤低质量细胞")
# Returns: Functions related to cell filtering
```

### 5. Direct Function Access

You can still call functions directly without the agent:

```python
import omicverse as ov

# Direct call (traditional way)
adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})

# Agent call (natural language)
agent = ov.Agent(model="gpt-4o")
adata = agent.run("qc with nUMI>500, mito<0.2", adata)

# Both do the same thing!
```

---

## Key Takeaways

### For Users:

1. **Natural Language**: Use Chinese, English, or abbreviations - all work
2. **Simple Tasks**: Fast execution with Priority 1 (1-3 function calls)
3. **Complex Tasks**: Comprehensive workflows with Priority 2 (multi-step)
4. **Validation**: Reflection and result review ensure correctness
5. **Transparency**: Generated code is shown before execution

### For Developers:

1. **Easy Registration**: Just add `@register_function` decorator
2. **Multi-Language**: Support Chinese and English aliases
3. **Auto-Discovery**: Functions are automatically available to agent
4. **Metadata-Rich**: Examples, related functions, categories
5. **Extensible**: Add new functions without modifying agent code

### Architecture Benefits:

1. **Separation of Concerns**: Registry (data) separate from Agent (logic)
2. **Dynamic Loading**: Functions registered at module import time
3. **LLM-Friendly**: JSON format perfect for prompt injection
4. **Searchable**: Fuzzy matching, aliases, categories
5. **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Alibaba

---

## Next Steps

- **Add New Functions**: See [Function Registration](#function-registration)
- **Create Skills**: See `.claude/skills/` for workflow templates
- **Customize Agent**: See `model_config.py` for provider settings
- **Run Tests**: See `tests/utils/test_smart_agent.py` for examples

For more information:
- Registry: `omicverse/utils/registry.py`
- Agent: `omicverse/utils/smart_agent.py`
- Skills: `omicverse/utils/skill_registry.py`
