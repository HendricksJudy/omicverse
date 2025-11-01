#!/usr/bin/env python3
"""
Direct test of OmicVerse Agent modules without package initialization
This bypasses the full omicverse package import to avoid dependency issues
"""

import sys
import importlib.util
from pathlib import Path

project_root = Path(__file__).parent


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def import_module_from_path(module_name, file_path):
    """Import a module directly from its file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_skill_registry_direct():
    """Test 1: Direct Skill Registry Loading"""
    print_section("Test 1: Skill Registry (Direct Import)")

    try:
        # Import skill_registry directly
        skill_registry_path = project_root / "omicverse" / "utils" / "skill_registry.py"
        skill_registry = import_module_from_path("skill_registry", skill_registry_path)

        SkillRegistry = skill_registry.SkillRegistry

        # Load skills
        skill_root = project_root / ".claude" / "skills"
        registry = SkillRegistry(skill_root=skill_root)
        registry.load()

        print(f"✓ Loaded {len(registry.skills)} skills\n")

        # Categorize and display skills
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
            print(f"📊 Bulk Analysis Skills ({len(bulk_skills)}):")
            for slug, skill in bulk_skills:
                print(f"  • {slug}")

        if single_skills:
            print(f"\n🧬 Single-Cell Analysis Skills ({len(single_skills)}):")
            for slug, skill in single_skills:
                print(f"  • {slug}")

        if other_skills:
            print(f"\n🔬 Other Skills ({len(other_skills)}):")
            for slug, skill in other_skills:
                print(f"  • {slug}")

        return True, registry
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_skill_routing(registry):
    """Test 2: Skill Routing"""
    print_section("Test 2: Skill Routing")

    if registry is None:
        print("⚠ Skipping - registry not loaded")
        return False

    try:
        # Import SkillRouter
        skill_registry_path = project_root / "omicverse" / "utils" / "skill_registry.py"
        skill_registry = import_module_from_path("skill_registry_routing", skill_registry_path)
        SkillRouter = skill_registry.SkillRouter

        # Create router
        router = SkillRouter(registry)

        test_queries = [
            ("Single-cell preprocessing", "How do I preprocess single-cell data?"),
            ("Bulk DEG analysis", "I need to do differential expression analysis on bulk RNA-seq"),
            ("Clustering", "Help me with clustering my scRNA-seq dataset"),
            ("Spatial analysis", "How do I analyze spatial transcriptomics?"),
            ("Cell communication", "Cell-cell communication analysis"),
            ("WGCNA", "WGCNA co-expression network"),
            ("Batch correction", "Combat batch correction"),
            ("Trajectory", "Trajectory inference pseudotime"),
        ]

        print("Testing skill routing:\n")
        for label, query in test_queries:
            matches = router.route(query, top_k=3)

            if matches:
                top_match = matches[0]
                print(f"📝 {label}")
                print(f"   Query: '{query}'")
                print(f"   → {top_match.skill.slug} (score: {top_match.score:.3f})")
            else:
                print(f"⚠ {label}: No matches")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_content(registry):
    """Test 3: Skill Content"""
    print_section("Test 3: Skill Content Details")

    if registry is None:
        print("⚠ Skipping - registry not loaded")
        return False

    try:
        # Check a few skills in detail
        test_skills = [
            'single-cell-preprocessing-with-omicverse',
            'bulk-rna-seq-differential-expression-with-omicverse',
            'spatial-transcriptomics-tutorials-with-omicverse'
        ]

        for skill_slug in test_skills:
            if skill_slug in registry.skills:
                skill = registry.skills[skill_slug]
                print(f"\n{'─'*80}")
                print(f"Skill: {skill.slug}")
                print(f"Name: {skill.name}")
                print(f"Description: {skill.description}")

                # Get instructions preview
                instructions = skill.prompt_instructions(max_chars=300)
                print(f"\nInstructions (preview):")
                print(instructions[:300] + "...")
            else:
                print(f"⚠ {skill_slug} not found")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_config_direct():
    """Test 4: Model Configuration (Direct Import)"""
    print_section("Test 4: Model Configuration")

    try:
        # Import model_config directly
        model_config_path = project_root / "omicverse" / "utils" / "model_config.py"
        model_config = import_module_from_path("model_config", model_config_path)

        ModelConfig = model_config.ModelConfig

        # Test common models
        test_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "anthropic/claude-sonnet-4-20250514",
            "gemini/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ]

        print("Testing model configuration:\n")
        for model in test_models:
            is_supported = ModelConfig.is_model_supported(model)
            if is_supported:
                description = ModelConfig.get_model_description(model)
                provider = ModelConfig.get_provider_from_model(model)
                print(f"✓ {model}")
                print(f"  Description: {description}")
                print(f"  Provider: {provider}\n")
            else:
                print(f"✗ {model} - Not supported\n")

        # List models
        models_str = ModelConfig.list_supported_models(show_all=False)
        print("\nCommonly used models:")
        print(models_str)

        # Count total
        full_list = ModelConfig.list_supported_models(show_all=True)
        model_count = len([line for line in full_list.split('\n') if '│' in line and 'Provider' not in line])
        print(f"\n✓ Total supported models: ~{model_count}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skill_files():
    """Test 5: Skill File Structure"""
    print_section("Test 5: Skill File Structure")

    try:
        skills_dir = project_root / ".claude" / "skills"

        if not skills_dir.exists():
            print(f"⚠ Skills directory not found at: {skills_dir}")
            return False

        skill_dirs = sorted([d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

        print(f"Found {len(skill_dirs)} skill directories:\n")

        total_skill_size = 0
        total_ref_size = 0

        for skill_dir in skill_dirs:
            skill_md = skill_dir / "SKILL.md"
            reference_md = skill_dir / "reference.md"

            skill_exists = skill_md.exists()
            ref_exists = reference_md.exists()

            status = "✓" if skill_exists else "✗"
            print(f"{status} {skill_dir.name}")

            if skill_exists:
                size = skill_md.stat().st_size
                total_skill_size += size
                print(f"   SKILL.md: {size:,} bytes", end="")

            if ref_exists:
                ref_size = reference_md.stat().st_size
                total_ref_size += ref_size
                print(f" | reference.md: {ref_size:,} bytes", end="")

            print()

        print(f"\nTotal documentation:")
        print(f"  SKILL.md files: {total_skill_size:,} bytes ({total_skill_size/1024:.1f} KB)")
        print(f"  reference.md files: {total_ref_size:,} bytes ({total_ref_size/1024:.1f} KB)")
        print(f"  Combined: {(total_skill_size + total_ref_size):,} bytes ({(total_skill_size + total_ref_size)/1024:.1f} KB)")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_seeker_api():
    """Test 6: Agent Seeker API Structure"""
    print_section("Test 6: Agent Seeker API")

    try:
        # Check agent module exists
        agent_init = project_root / "omicverse" / "agent" / "__init__.py"

        if not agent_init.exists():
            print(f"⚠ Agent module not found at: {agent_init}")
            return False

        # Read the file to check for seeker function
        with open(agent_init) as f:
            content = f.read()

        if 'def seeker' in content:
            print("✓ agent.seeker() function found in module")

            # Extract function signature
            import re
            match = re.search(r'def seeker\((.*?)\):', content, re.DOTALL)
            if match:
                params = match.group(1)
                print(f"\nFunction signature:")
                print(f"  def seeker({params})")

            print("\n📚 Purpose:")
            print("  Creates Claude Agent skills from documentation links")
            print("  Scaffolds SKILL.md with YAML frontmatter and reference docs")
            print("  Optional .zip packaging for Claude upload")

            print("\n📝 Example usage:")
            print("  import omicverse as ov")
            print("  result = ov.agent.seeker(")
            print("      'https://example.com/docs',")
            print("      name='MySkill',")
            print("      description='Custom analysis',")
            print("      package=True")
            print("  )")

            return True
        else:
            print("⚠ seeker function not found in agent module")
            return False

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("  OMICVERSE AGENT & SKILL SYSTEM - DIRECT TEST SUITE")
    print("  (Bypassing full package import to avoid dependencies)")
    print("█"*80)

    results = []
    registry = None

    # Test 1: Skill Registry
    result, registry = test_skill_registry_direct()
    results.append(("Skill Registry Loading", result))

    # Test 2: Skill Routing (needs registry)
    result = test_skill_routing(registry)
    results.append(("Skill Routing", result))

    # Test 3: Skill Content (needs registry)
    result = test_skill_content(registry)
    results.append(("Skill Content Details", result))

    # Test 4: Model Config
    result = test_model_config_direct()
    results.append(("Model Configuration", result))

    # Test 5: Skill Files
    result = test_skill_files()
    results.append(("Skill File Structure", result))

    # Test 6: Agent Seeker
    result = test_agent_seeker_api()
    results.append(("Agent Seeker API", result))

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
