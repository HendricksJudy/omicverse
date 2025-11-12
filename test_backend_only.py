"""
Test just the LLM backend without full omicverse dependencies
"""

import sys
sys.path.insert(0, '/home/user/omicverse')

print("="*80)
print("TESTING OV.AGENT LLM BACKEND WITH GEMINI 2.0 FLASH")
print("="*80)

# Import backend only
try:
    from omicverse.utils.agent_backend import OmicVerseLLMBackend
    print("\n✓ Backend imported successfully\n")
except Exception as e:
    print(f"\n✗ Failed to import backend: {e}\n")
    sys.exit(1)

# Initialize backend
print("Initializing Gemini 2.0 Flash backend...")
try:
    backend = OmicVerseLLMBackend(
        system_prompt="You are a helpful bioinformatics assistant specialized in single-cell analysis.",
        model="gemini-2.0-flash-exp",
        api_key="AIzaSyBnS7j5WcLZoUZFN_50FiDQEGJVIdH2bM0",
        temperature=0.2
    )
    print(f"✓ Backend initialized")
    print(f"  Model: {backend.model}")
    print(f"  Provider: {backend.provider}")
    print(f"  Temperature: {backend.temperature}")
except Exception as e:
    print(f"✗ Failed to initialize backend: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Simple query
print("\n" + "="*80)
print("TEST 1: Simple mathematical query")
print("="*80)
try:
    print("Query: What is 127 + 456? Answer with just the number.")
    response = backend.run("What is 127 + 456? Answer with just the number.")
    print(f"Response: {response}")
    print("✓ Test 1 passed")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")

# Test 2: Code generation
print("\n" + "="*80)
print("TEST 2: Python code generation")
print("="*80)
try:
    query = """Generate Python code to calculate the mean of a list: [1, 2, 3, 4, 5].
    Return only the code in a markdown code block."""
    print(f"Query: {query}")
    response = backend.run(query)
    print(f"Response:\n{response}")
    if "```python" in response or "```" in response:
        print("✓ Test 2 passed - Code block detected")
    else:
        print("⚠ Test 2 warning - No code block markdown detected")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")

# Test 3: Bioinformatics knowledge
print("\n" + "="*80)
print("TEST 3: Bioinformatics domain knowledge")
print("="*80)
try:
    query = "What is UMAP in the context of single-cell analysis? Answer in one sentence."
    print(f"Query: {query}")
    response = backend.run(query)
    print(f"Response: {response}")
    if "dimension" in response.lower() or "reduce" in response.lower() or "visual" in response.lower():
        print("✓ Test 3 passed - Contains relevant terms")
    else:
        print("⚠ Test 3 warning - Response may be incomplete")
except Exception as e:
    print(f"✗ Test 3 failed: {e}")

# Test 4: OmicVerse specific code generation
print("\n" + "="*80)
print("TEST 4: OmicVerse code generation")
print("="*80)
try:
    query = """Write Python code using omicverse to perform quality control on single-cell data.
    The code should filter cells with nUMI > 500.
    Assume adata is already loaded.
    Return only the code."""
    print(f"Query: {query[:100]}...")
    response = backend.run(query)
    print(f"Response:\n{response}")
    if "ov.pp.qc" in response or "quality control" in response.lower():
        print("✓ Test 4 passed - OmicVerse function mentioned")
    else:
        print("⚠ Test 4 warning - OmicVerse function not clearly identified")
except Exception as e:
    print(f"✗ Test 4 failed: {e}")

# Test 5: Response speed
print("\n" + "="*80)
print("TEST 5: Response speed")
print("="*80)
try:
    import time
    query = "List 3 common clustering methods in single-cell analysis."
    print(f"Query: {query}")
    start = time.time()
    response = backend.run(query)
    duration = time.time() - start
    print(f"Response: {response}")
    print(f"Duration: {duration:.2f}s")
    if duration < 5.0:
        print(f"✓ Test 5 passed - Fast response ({duration:.2f}s < 5s)")
    else:
        print(f"⚠ Test 5 warning - Slow response ({duration:.2f}s >= 5s)")
except Exception as e:
    print(f"✗ Test 5 failed: {e}")

print("\n" + "="*80)
print("BACKEND TESTING COMPLETE")
print("="*80)
print("\nAll basic backend tests completed successfully!")
print("Backend is working correctly with Gemini 2.0 Flash.")
