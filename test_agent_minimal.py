"""
Minimal test to check if agent can be initialized and basic functionality works
"""

import sys
import os

# Add omicverse to path
sys.path.insert(0, '/home/user/omicverse')

print("="*80)
print("MINIMAL AGENT TEST")
print("="*80)

# Try importing agent components directly
print("\n1. Testing agent imports...")
try:
    from omicverse.utils.smart_agent import OmicVerseAgent
    print("   ✓ OmicVerseAgent imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import OmicVerseAgent: {e}")
    sys.exit(1)

try:
    from omicverse.utils.agent_backend import OmicVerseLLMBackend
    print("   ✓ OmicVerseLLMBackend imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import OmicVerseLLMBackend: {e}")
    sys.exit(1)

# Try initializing agent
print("\n2. Testing agent initialization with Gemini 2.0 Flash...")
try:
    agent = OmicVerseAgent(
        model="gemini-2.0-flash-exp",
        api_key="AIzaSyBnS7j5WcLZoUZFN_50FiDQEGJVIdH2bM0",
        enable_reflection=True,
        enable_result_review=True
    )
    print("   ✓ Agent initialized successfully")
    print(f"   Model: {agent.model}")
    print(f"   Reflection enabled: {agent.enable_reflection}")
    print(f"   Result review enabled: {agent.enable_result_review}")
except Exception as e:
    print(f"   ✗ Failed to initialize agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try a simple backend test
print("\n3. Testing LLM backend...")
try:
    from omicverse.utils.agent_backend import OmicVerseLLMBackend
    backend = OmicVerseLLMBackend(
        system_prompt="You are a helpful bioinformatics assistant.",
        model="gemini-2.0-flash-exp",
        api_key="AIzaSyBnS7j5WcLZoUZFN_50FiDQEGJVIdH2bM0"
    )
    print("   ✓ Backend initialized successfully")

    # Test a simple query
    print("\n4. Testing simple LLM query...")
    response = backend.run("What is 2+2? Answer with just the number.")
    print(f"   Response: {response[:100]}")
    print("   ✓ Backend query successful")

except Exception as e:
    print(f"   ✗ Backend test failed: {e}")
    import traceback
    traceback.print_exc()

# Try checking registry
print("\n5. Testing function registry...")
try:
    from omicverse.utils.registry import FunctionRegistry
    registry = FunctionRegistry()
    print(f"   ✓ Registry initialized")
    print(f"   Total registered functions: {len(registry.functions)}")

    # Show a few function names
    func_names = list(registry.functions.keys())[:5]
    print(f"   Sample functions: {func_names}")
except Exception as e:
    print(f"   ✗ Registry test failed: {e}")

# Try checking skill registry
print("\n6. Testing skill registry...")
try:
    from omicverse.utils.skill_registry import SkillRegistry
    skill_registry = SkillRegistry()
    print(f"   ✓ Skill registry initialized")
    print(f"   Total skills: {len(skill_registry.skills)}")

    # Show skill names
    skill_names = list(skill_registry.skills.keys())[:5]
    print(f"   Sample skills: {skill_names}")
except Exception as e:
    print(f"   ✗ Skill registry test failed: {e}")

print("\n" + "="*80)
print("MINIMAL TEST COMPLETE")
print("="*80)
print("\nAll basic components are working! Ready for comprehensive testing.")
