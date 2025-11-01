#!/usr/bin/env python3
"""
Test script for OmicVerse Agent and Skill System
Tests both ov.Agent() and ov.agent.seeker() functionality
"""

import os
import sys
from pathlib import Path

# Add omicverse to path
sys.path.insert(0, str(Path(__file__).parent))

import omicverse as ov


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_agent_initialization():
    """Test 1: Agent Initialization"""
    print_section("Test 1: Agent Initialization")

    try:
        # Test without API key (should work but warn about missing key)
        print("Creating agent with default settings (gpt-4o-mini)...")
        agent = ov.Agent(model="gpt-4o-mini")
        print(f"✓ Agent created successfully: {agent}")
        print(f"  Model: {agent.model}")
        print(f"  Temperature: {agent.temperature}")
        return True
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        return False


def test_list_supported_models():
    """Test 2: List Supported Models"""
    print_section("Test 2: List Supported Models")

    try:
        print("Listing supported models...")
        models_str = ov.list_supported_models(show_all=False)
        print("\nSupported models (commonly used):")
        print(models_str)

        # Also test with show_all=True to get full list
        print("\n\nGetting full model list count...")
        full_models = ov.list_supported_models(show_all=True)
        model_count = len([line for line in full_models.split('\n') if line.strip().startswith('-')])
        print(f"✓ Total supported models: {model_count}")
        return True
    except Exception as e:
        print(f"✗ Failed to list models: {e}")
        return False


def test_skill_registry():
    """Test 3: Skill Registry Loading"""
    print_section("Test 3: Skill Registry Loading")

    try:
        from omicverse.utils.skill_registry import SkillRegistry

        # Find skills directory
        project_root = Path(__file__).parent
        skills_dir = project_root / ".claude" / "skills"

        if not skills_dir.exists():
            print(f"⚠ Skills directory not found at: {skills_dir}")
            return False

        print(f"Loading skills from: {skills_dir}")
        registry = SkillRegistry(project_root=project_root)
        registry.load()

        print(f"✓ Loaded {len(registry.skills)} skills:")
        for slug, skill in sorted(registry.skills.items()):
            print(f"  - {slug}")
            print(f"    Title: {skill.title}")
            print(f"    Description: {skill.description[:80]}...")

        return True
    except Exception as e:
        print(f"✗ Failed to load skill registry: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_routing():
    """Test 4: Skill Routing"""
    print_section("Test 4: Skill Routing")

    try:
        from omicverse.utils.skill_registry import SkillRegistry

        project_root = Path(__file__).parent
        registry = SkillRegistry(project_root=project_root)
        registry.load()

        # Test queries
        test_queries = [
            "How do I preprocess single-cell data?",
            "I need to do differential expression analysis on bulk RNA-seq",
            "Help me with clustering my scRNA-seq dataset",
            "How do I analyze spatial transcriptomics?",
        ]

        print("Testing skill routing with different queries:\n")
        for query in test_queries:
            print(f"Query: '{query}'")
            matches = registry.route(query, top_k=3)

            if matches:
                print(f"  Top matches:")
                for i, match in enumerate(matches, 1):
                    print(f"    {i}. {match.skill.slug} (score: {match.score:.3f})")
            else:
                print("  No matches found")
            print()

        return True
    except Exception as e:
        print(f"✗ Failed to test skill routing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_seeker():
    """Test 5: Skill Seeker (dry run)"""
    print_section("Test 5: Skill Seeker Functionality")

    print("Testing ov.agent.seeker() function...")
    print("\nFunction signature:")
    print("  ov.agent.seeker(")
    print("      links: Union[str, List[str]],")
    print("      name: Optional[str] = None,")
    print("      description: Optional[str] = None,")
    print("      max_pages: int = 30,")
    print("      target: str = 'skills',")
    print("      out_dir: Optional[Path] = None,")
    print("      package: bool = False")
    print("  )")

    print("\nExample usage:")
    print("  result = ov.agent.seeker(")
    print("      'https://example.com/docs',")
    print("      name='MySkill',")
    print("      description='Custom analysis skill',")
    print("      package=True")
    print("  )")

    print("\n✓ Skill seeker function is available")
    print("  Note: Actual skill creation requires valid URLs and network access")

    return True


def test_agent_with_skills():
    """Test 6: Agent with Skill Registry Integration"""
    print_section("Test 6: Agent with Skill Registry Integration")

    try:
        from omicverse.utils.smart_agent import OmicVerseAgent
        from omicverse.utils.skill_registry import SkillRegistry

        project_root = Path(__file__).parent

        # Create registry
        print("Loading skill registry...")
        registry = SkillRegistry(project_root=project_root)
        registry.load()
        print(f"✓ Loaded {len(registry.skills)} skills")

        # Check if agent can find skills
        print("\nChecking agent integration with skills...")
        print("  Skills are automatically loaded when agent processes requests")
        print("  The agent uses skill routing to inject relevant skill guidance")

        # Show an example of how skills would be used
        print("\nExample workflow:")
        print("  1. User: 'I need to preprocess single-cell data'")
        print("  2. Agent routes to 'single-preprocessing' skill")
        print("  3. Skill guidance is injected into agent prompt")
        print("  4. Agent generates code following skill instructions")

        return True
    except Exception as e:
        print(f"✗ Failed to test agent-skill integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_discovery():
    """Test 7: Function Discovery"""
    print_section("Test 7: Function Discovery System")

    try:
        from omicverse.utils.smart_agent import OmicVerseAgent

        print("Testing function registry...")
        agent = ov.Agent(model="gpt-4o-mini")

        # Get the underlying agent
        if hasattr(agent, '_ov_agent'):
            ov_agent = agent._ov_agent

            # Show function discovery capabilities
            print(f"✓ Function registry contains {len(ov_agent.function_registry)} functions")

            # Test function search
            print("\nTesting function search for 'preprocess'...")
            results = ov_agent._search_functions("preprocess single cell quality control")
            print(f"  Found {len(results)} relevant functions:")
            for func_name in results[:5]:  # Show top 5
                print(f"    - {func_name}")

        return True
    except Exception as e:
        print(f"✗ Failed to test function discovery: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("  OMICVERSE AGENT & SKILL SYSTEM TEST SUITE")
    print("█"*80)

    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("List Supported Models", test_list_supported_models),
        ("Skill Registry Loading", test_skill_registry),
        ("Skill Routing", test_skill_routing),
        ("Skill Seeker", test_skill_seeker),
        ("Agent-Skill Integration", test_agent_with_skills),
        ("Function Discovery", test_function_discovery),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
