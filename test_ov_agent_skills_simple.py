#!/usr/bin/env python3
"""
Simplified test script for OmicVerse Agent and Skill System
Tests the agent module directly without requiring full omicverse installation
"""

import sys
from pathlib import Path

# Add omicverse to path
sys.path.insert(0, str(Path(__file__).parent))


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_skill_registry_loading():
    """Test 1: Direct Skill Registry Loading"""
    print_section("Test 1: Skill Registry Loading")

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

        print(f"✓ Loaded {len(registry.skills)} skills\n")

        # Display skills in categorized format
        bulk_skills = []
        single_skills = []
        other_skills = []

        for slug, skill in sorted(registry.skills.items()):
            if slug.startswith('bulk-'):
                bulk_skills.append((slug, skill))
            elif slug.startswith('single-'):
                single_skills.append((slug, skill))
            else:
                other_skills.append((slug, skill))

        if bulk_skills:
            print("📊 Bulk Analysis Skills:")
            for slug, skill in bulk_skills:
                print(f"  • {slug}")
                print(f"    {skill.description[:100]}...")
                print()

        if single_skills:
            print("🧬 Single-Cell Analysis Skills:")
            for slug, skill in single_skills:
                print(f"  • {slug}")
                print(f"    {skill.description[:100]}...")
                print()

        if other_skills:
            print("🔬 Other Skills:")
            for slug, skill in other_skills:
                print(f"  • {slug}")
                print(f"    {skill.description[:100]}...")
                print()

        return True
    except Exception as e:
        print(f"✗ Failed to load skill registry: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_routing():
    """Test 2: Skill Routing"""
    print_section("Test 2: Skill Routing")

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
            "Cell-cell communication analysis",
            "WGCNA co-expression network",
            "Combat batch correction",
            "Trajectory inference pseudotime",
        ]

        print("Testing skill routing with different queries:\n")
        for query in test_queries:
            print(f"📝 Query: '{query}'")
            matches = registry.route(query, top_k=3)

            if matches:
                print(f"   Top {len(matches)} matches:")
                for i, match in enumerate(matches, 1):
                    print(f"      {i}. {match.skill.slug:<40} (score: {match.score:.3f})")
            else:
                print("   ⚠ No matches found")
            print()

        return True
    except Exception as e:
        print(f"✗ Failed to test skill routing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_content():
    """Test 3: Skill Content Display"""
    print_section("Test 3: Skill Content Inspection")

    try:
        from omicverse.utils.skill_registry import SkillRegistry

        project_root = Path(__file__).parent
        registry = SkillRegistry(project_root=project_root)
        registry.load()

        # Pick a skill to inspect
        if 'single-preprocessing' in registry.skills:
            skill = registry.skills['single-preprocessing']
            print(f"Inspecting skill: {skill.slug}")
            print(f"Title: {skill.title}")
            print(f"Name: {skill.name}")
            print(f"Description: {skill.description}\n")

            # Show skill body (first 500 chars)
            instructions = skill.prompt_instructions(max_chars=500)
            print("Skill Instructions (preview):")
            print("-" * 80)
            print(instructions)
            print("-" * 80)

            # Show metadata
            print(f"\nMetadata:")
            for key, value in skill.metadata.items():
                if key not in ['name', 'title', 'description']:
                    print(f"  {key}: {value}")

            return True
        else:
            print("⚠ single-preprocessing skill not found")
            return False

    except Exception as e:
        print(f"✗ Failed to inspect skill content: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_config():
    """Test 4: Model Configuration"""
    print_section("Test 4: Model Configuration")

    try:
        from omicverse.utils.model_config import ModelConfig

        print("Testing ModelConfig class...\n")

        # Test some common models
        test_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "anthropic/claude-sonnet-4-20250514",
            "gemini/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ]

        print("Testing model support:")
        for model in test_models:
            is_supported = ModelConfig.is_model_supported(model)
            description = ModelConfig.get_model_description(model) if is_supported else "N/A"
            provider = ModelConfig.get_provider_from_model(model) if is_supported else "N/A"
            status = "✓" if is_supported else "✗"
            print(f"  {status} {model:<45} | {description:<25} | Provider: {provider}")

        # List all supported models (sample)
        print("\n\nListing commonly used models:")
        models_str = ModelConfig.list_supported_models(show_all=False)
        print(models_str)

        # Count total models
        full_list = ModelConfig.list_supported_models(show_all=True)
        model_count = len([line for line in full_list.split('\n') if '│' in line and not line.strip().startswith('│ Provider')])
        print(f"\n✓ Total supported models: {model_count}")

        return True
    except Exception as e:
        print(f"✗ Failed to test model config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_seeker_api():
    """Test 5: Agent Seeker API"""
    print_section("Test 5: Agent Seeker API (ov.agent.seeker)")

    try:
        # Import the agent module directly
        from omicverse import agent

        print("Testing agent.seeker() availability...\n")

        # Check if seeker function exists
        if hasattr(agent, 'seeker'):
            print("✓ agent.seeker() function is available\n")

            # Show the function signature
            import inspect
            sig = inspect.signature(agent.seeker)
            print("Function signature:")
            print(f"  agent.seeker{sig}\n")

            # Show docstring
            if agent.seeker.__doc__:
                print("Documentation:")
                print("-" * 80)
                print(agent.seeker.__doc__)
                print("-" * 80)

            print("\n📚 Example usage:")
            print("  # Create skill from single URL")
            print("  result = ov.agent.seeker(")
            print("      'https://example.com/docs/feature',")
            print("      name='New Analysis',")
            print("      description='Custom analysis workflow',")
            print("      package=True")
            print("  )")
            print("\n  # Create skill from multiple URLs")
            print("  result = ov.agent.seeker(")
            print("      ['https://docs.site-a.com/', 'https://docs.site-b.com/guide'],")
            print("      name='multi-source',")
            print("      max_pages=50,")
            print("      package=True")
            print("  )")

            return True
        else:
            print("✗ agent.seeker() not found")
            return False

    except Exception as e:
        print(f"✗ Failed to test agent.seeker API: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_files():
    """Test 6: Skill File Structure"""
    print_section("Test 6: Skill File Structure")

    try:
        project_root = Path(__file__).parent
        skills_dir = project_root / ".claude" / "skills"

        if not skills_dir.exists():
            print(f"⚠ Skills directory not found")
            return False

        # List all skill directories
        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        print(f"Found {len(skill_dirs)} skill directories:\n")

        for skill_dir in sorted(skill_dirs):
            skill_md = skill_dir / "SKILL.md"
            reference_md = skill_dir / "reference.md"

            skill_exists = skill_md.exists()
            ref_exists = reference_md.exists()

            status = "✓" if skill_exists else "✗"
            print(f"{status} {skill_dir.name}")

            if skill_exists:
                # Check file size
                size = skill_md.stat().st_size
                print(f"   SKILL.md: {size:,} bytes")
            else:
                print(f"   ⚠ SKILL.md missing!")

            if ref_exists:
                ref_size = reference_md.stat().st_size
                print(f"   reference.md: {ref_size:,} bytes")

        return True

    except Exception as e:
        print(f"✗ Failed to check skill files: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("  OMICVERSE AGENT & SKILL SYSTEM - SIMPLIFIED TEST SUITE")
    print("█"*80)

    tests = [
        ("Skill Registry Loading", test_skill_registry_loading),
        ("Skill Routing", test_skill_routing),
        ("Skill Content Inspection", test_skill_content),
        ("Model Configuration", test_model_config),
        ("Agent Seeker API", test_agent_seeker_api),
        ("Skill File Structure", test_skill_files),
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
