#!/usr/bin/env python3
"""
Generate reference data for agent integration tests.

This script executes key tutorial notebook workflows and saves
intermediate outputs as reference data for validating ov.agent functionality.

Usage:
    python scripts/generate_reference_data.py [--output DIR] [--force]

Options:
    --output DIR    Output directory (default: tests/integration/agent/data/)
    --force         Overwrite existing references
    --dataset NAME  Generate only specific dataset (pbmc3k, bulk_deg, or all)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from tests.integration.agent.utils.data_generators import (
    generate_pbmc3k_references,
    generate_bulk_deg_reference,
    generate_all_references
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate reference data for agent integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all reference data
  python scripts/generate_reference_data.py

  # Generate only PBMC3k references
  python scripts/generate_reference_data.py --dataset pbmc3k

  # Force regeneration
  python scripts/generate_reference_data.py --force

  # Custom output directory
  python scripts/generate_reference_data.py --output /path/to/output
        """
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'tests' / 'integration' / 'agent' / 'data',
        help='Output directory (default: tests/integration/agent/data/)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing references'
    )

    parser.add_argument(
        '--dataset',
        choices=['pbmc3k', 'bulk_deg', 'all'],
        default='all',
        help='Which dataset to generate (default: all)'
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("OmicVerse Agent - Reference Data Generator")
    print("="*70)
    print(f"Output directory: {args.output}")
    print(f"Force overwrite: {args.force}")
    print(f"Dataset: {args.dataset}")
    print("="*70)
    print()

    results = {}

    try:
        if args.dataset == 'all':
            results = generate_all_references(args.output, force=args.force)

        elif args.dataset == 'pbmc3k':
            print("Generating PBMC3k references...")
            pbmc_dir = args.output / 'pbmc3k'
            results['pbmc3k'] = generate_pbmc3k_references(
                pbmc_dir,
                force=args.force
            )
            results['pbmc3k']['status'] = 'success'

        elif args.dataset == 'bulk_deg':
            print("Generating bulk DEG references...")
            bulk_dir = args.output / 'bulk_deg'
            results['bulk_deg'] = generate_bulk_deg_reference(
                bulk_dir,
                force=args.force
            )
            results['bulk_deg']['status'] = 'success'

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "="*70)
    print("Generation Complete")
    print("="*70)

    success_count = sum(
        1 for r in results.values()
        if r.get('status') in ['success', 'exists']
    )
    total_count = len(results)

    print(f"Status: {success_count}/{total_count} datasets generated successfully")

    if success_count < total_count:
        print("\n⚠️  Some datasets failed to generate")
        for name, result in results.items():
            if result.get('status') == 'failed':
                print(f"  - {name}: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print("\n✅ All reference data generated successfully!")
        print("\nNext steps:")
        print("  1. Review generated files in:", args.output)
        print("  2. Run integration tests:")
        print("     pytest tests/integration/agent/ -m quick")
        print("  3. Commit reference data if acceptable:")
        print("     git add tests/integration/agent/data/")
        print('     git commit -m "Add agent test reference data"')

    return 0


if __name__ == '__main__':
    sys.exit(main())
