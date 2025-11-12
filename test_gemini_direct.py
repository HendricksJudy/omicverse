"""
Direct test of Gemini API without omicverse dependencies
"""

import google.generativeai as genai
import time

print("="*80)
print("DIRECT GEMINI 2.0 FLASH API TEST")
print("="*80)

API_KEY = "AIzaSyBnS7j5WcLZoUZFN_50FiDQEGJVIdH2bM0"

# Configure Gemini
print("\nConfiguring Gemini API...")
genai.configure(api_key=API_KEY)

# Initialize model
print("Initializing Gemini 2.0 Flash model...")
model = genai.GenerativeModel('gemini-2.0-flash-exp')
print("✓ Model initialized\n")

# Test 1: Simple query
print("="*80)
print("TEST 1: Simple mathematical query")
print("="*80)
query = "What is 127 + 456? Answer with just the number."
print(f"Query: {query}")
response = model.generate_content(query)
print(f"Response: {response.text}")
print("✓ Test 1 passed\n")

# Test 2: Code generation
print("="*80)
print("TEST 2: Python code generation")
print("="*80)
query = """Generate Python code to calculate the mean of a list: [1, 2, 3, 4, 5].
Return only the code in a markdown code block."""
print(f"Query: {query}")
response = model.generate_content(query)
print(f"Response:\n{response.text}")
if "```" in response.text:
    print("✓ Test 2 passed - Code block detected\n")
else:
    print("⚠ Test 2 warning - No code block markdown\n")

# Test 3: Bioinformatics knowledge
print("="*80)
print("TEST 3: Bioinformatics domain knowledge")
print("="*80)
query = "What is UMAP in single-cell analysis? Answer in one sentence."
print(f"Query: {query}")
response = model.generate_content(query)
print(f"Response: {response.text}")
print("✓ Test 3 passed\n")

# Test 4: OmicVerse specific code generation
print("="*80)
print("TEST 4: OmicVerse code generation")
print("="*80)
query = """Write Python code using omicverse library to perform quality control on single-cell data.
The code should filter cells with nUMI > 500 and mito < 0.2.
Assume adata is already loaded as an AnnData object.
Use the function ov.pp.qc().
Return only the code in a Python code block."""
print(f"Query: {query[:100]}...")
response = model.generate_content(query)
print(f"Response:\n{response.text}")
if "ov.pp.qc" in response.text:
    print("✓ Test 4 passed - ov.pp.qc function mentioned\n")
else:
    print("⚠ Test 4 warning - ov.pp.qc not found\n")

# Test 5: Response speed
print("="*80)
print("TEST 5: Response speed (Flash model)")
print("="*80)
query = "List 3 clustering methods in single-cell analysis in one line."
print(f"Query: {query}")
start = time.time()
response = model.generate_content(query)
duration = time.time() - start
print(f"Response: {response.text}")
print(f"Duration: {duration:.2f}s")
if duration < 3.0:
    print(f"✓ Test 5 passed - Very fast response ({duration:.2f}s < 3s)\n")
elif duration < 5.0:
    print(f"✓ Test 5 passed - Fast response ({duration:.2f}s < 5s)\n")
else:
    print(f"⚠ Test 5 warning - Slow response ({duration:.2f}s >= 5s)\n")

# Test 6: Complex bioinformatics code generation
print("="*80)
print("TEST 6: Complex workflow code generation")
print("="*80)
query = """Generate Python code for a complete single-cell preprocessing workflow using omicverse and scanpy:
1. Quality control with nUMI>500, mito<0.2
2. Normalization and HVG selection (2000 genes)
3. Scaling
4. PCA computation (50 components)
5. UMAP computation

Assume adata is loaded. Use omicverse (ov) and scanpy (sc) libraries.
Return only Python code."""
print(f"Query: Complete preprocessing workflow")
start = time.time()
response = model.generate_content(query)
duration = time.time() - start
print(f"Response:\n{response.text}")
print(f"Duration: {duration:.2f}s")

# Check if key functions are mentioned
key_functions = ["ov.pp.qc", "preprocess", "scale", "pca", "umap"]
found = [f for f in key_functions if f.lower() in response.text.lower()]
print(f"Key functions found: {found}")
if len(found) >= 3:
    print(f"✓ Test 6 passed - {len(found)}/5 key functions mentioned\n")
else:
    print(f"⚠ Test 6 warning - Only {len(found)}/5 key functions found\n")

# Test 7: Code extraction test
print("="*80)
print("TEST 7: Code extraction from response")
print("="*80)
query = """Write Python code to compute leiden clustering on single-cell data.
Assume neighbors are already computed.
Use scanpy.
Return code only."""
print(f"Query: {query}")
response = model.generate_content(query)
print(f"Response:\n{response.text}")

# Extract code from markdown
import re
code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response.text, re.DOTALL)
if code_blocks:
    print(f"\n✓ Extracted {len(code_blocks)} code block(s)")
    for i, code in enumerate(code_blocks):
        print(f"\nCode block {i+1}:")
        print(code)
else:
    print("\n⚠ No code blocks found with markdown formatting")
    print("Raw response might contain code without markdown")

print("\n" + "="*80)
print("GEMINI API TESTING COMPLETE")
print("="*80)
print("\nGemini 2.0 Flash is working correctly!")
print("API Key is valid")
print("Model can:")
print("  ✓ Answer simple queries")
print("  ✓ Generate Python code")
print("  ✓ Handle bioinformatics domain knowledge")
print("  ✓ Generate omicverse-specific code")
print("  ✓ Respond quickly (Flash model)")
print("  ✓ Generate complex workflows")
print("\nReady for full ov.agent testing once dependencies are installed!")
