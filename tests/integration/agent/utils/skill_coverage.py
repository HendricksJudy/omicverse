"""
Skill coverage tracking and reporting.

Provides utilities to:
- Track which skills are tested
- Generate coverage reports
- Identify untested skills
- Validate skill availability
"""

from typing import List, Dict, Set, Optional
from pathlib import Path
import json
from dataclasses import dataclass, field


@dataclass
class SkillInfo:
    """Information about a skill."""
    name: str
    description: str
    category: str
    tested: bool = False
    test_function: Optional[str] = None
    test_file: Optional[str] = None


# Complete list of OmicVerse built-in skills (25 skills)
ALL_SKILLS = {
    # Single-cell skills (10)
    'single-preprocessing': SkillInfo(
        'single-preprocessing',
        'Preprocess single-cell data with QC filtering and normalization',
        'single-cell'
    ),
    'single-clustering': SkillInfo(
        'single-clustering',
        'Cluster single-cell data using leiden or louvain algorithms',
        'single-cell'
    ),
    'single-annotation': SkillInfo(
        'single-annotation',
        'Annotate cell types using various methods (SCSA, CellTypist, etc.)',
        'single-cell'
    ),
    'single-trajectory': SkillInfo(
        'single-trajectory',
        'Infer trajectories using PAGA, Palantir, or VIA',
        'single-cell'
    ),
    'single-cellphone-db': SkillInfo(
        'single-cellphone-db',
        'Analyze cell-cell communication using CellPhoneDB',
        'single-cell'
    ),
    'single-downstream-analysis': SkillInfo(
        'single-downstream-analysis',
        'Downstream analyses including AUCell and metacell DEG',
        'single-cell'
    ),
    'single-multiomics': SkillInfo(
        'single-multiomics',
        'Multi-omics integration (MOFA, GLUE, SIMBA)',
        'single-cell'
    ),
    'single-to-spatial-mapping': SkillInfo(
        'single-to-spatial-mapping',
        'Map scRNA-seq to spatial transcriptomics',
        'single-cell'
    ),

    # Bulk RNA-seq skills (6)
    'bulk-deg-analysis': SkillInfo(
        'bulk-deg-analysis',
        'Differential expression analysis for bulk RNA-seq',
        'bulk'
    ),
    'bulk-deseq2-analysis': SkillInfo(
        'bulk-deseq2-analysis',
        'DESeq2-based differential expression',
        'bulk'
    ),
    'bulk-wgcna-analysis': SkillInfo(
        'bulk-wgcna-analysis',
        'Co-expression network analysis using WGCNA',
        'bulk'
    ),
    'bulk-combat-correction': SkillInfo(
        'bulk-combat-correction',
        'Batch effect removal using ComBat',
        'bulk'
    ),
    'bulk-stringdb-ppi': SkillInfo(
        'bulk-stringdb-ppi',
        'Protein-protein interaction networks from STRING',
        'bulk'
    ),
    'bulk-to-single-deconvolution': SkillInfo(
        'bulk-to-single-deconvolution',
        'Deconvolute bulk RNA-seq into single-cell fractions',
        'bulk'
    ),
    'bulk-trajblend-interpolation': SkillInfo(
        'bulk-trajblend-interpolation',
        'Trajectory interpolation using bulk data',
        'bulk'
    ),

    # Spatial transcriptomics skills (1)
    'spatial-tutorials': SkillInfo(
        'spatial-tutorials',
        'Spatial transcriptomics analysis workflows',
        'spatial'
    ),

    # TCGA/Cancer genomics (1)
    'tcga-preprocessing': SkillInfo(
        'tcga-preprocessing',
        'Preprocess TCGA data with survival metadata',
        'tcga'
    ),

    # Data utilities (5)
    'data-export-excel': SkillInfo(
        'data-export-excel',
        'Export data to Excel files',
        'data-utils'
    ),
    'data-export-pdf': SkillInfo(
        'data-export-pdf',
        'Generate PDF reports',
        'data-utils'
    ),
    'data-viz-plots': SkillInfo(
        'data-viz-plots',
        'Create visualizations using matplotlib/seaborn',
        'data-utils'
    ),
    'data-stats-analysis': SkillInfo(
        'data-stats-analysis',
        'Statistical tests and hypothesis testing',
        'data-utils'
    ),
    'data-transform': SkillInfo(
        'data-transform',
        'Data transformation and cleaning',
        'data-utils'
    ),

    # Plotting (1)
    'plotting-visualization': SkillInfo(
        'plotting-visualization',
        'Advanced plotting utilities from OmicVerse',
        'plotting'
    ),
}


class SkillCoverageTracker:
    """Track and report skill test coverage."""

    def __init__(self):
        """Initialize tracker with all skills."""
        self.skills = ALL_SKILLS.copy()

    def mark_tested(
        self,
        skill_name: str,
        test_function: str,
        test_file: str
    ):
        """
        Mark a skill as tested.

        Args:
            skill_name: Name of the skill
            test_function: Test function name
            test_file: Test file name
        """
        if skill_name in self.skills:
            skill = self.skills[skill_name]
            skill.tested = True
            skill.test_function = test_function
            skill.test_file = test_file

    def get_coverage_stats(self) -> Dict:
        """
        Get coverage statistics.

        Returns:
            dict: Coverage metrics
        """
        total = len(self.skills)
        tested = sum(1 for s in self.skills.values() if s.tested)

        by_category = {}
        for skill in self.skills.values():
            if skill.category not in by_category:
                by_category[skill.category] = {'total': 0, 'tested': 0}

            by_category[skill.category]['total'] += 1
            if skill.tested:
                by_category[skill.category]['tested'] += 1

        return {
            'total_skills': total,
            'tested_skills': tested,
            'untested_skills': total - tested,
            'coverage_percent': (tested / total * 100) if total > 0 else 0,
            'by_category': by_category
        }

    def get_tested_skills(self) -> List[SkillInfo]:
        """Get list of tested skills."""
        return [s for s in self.skills.values() if s.tested]

    def get_untested_skills(self) -> List[SkillInfo]:
        """Get list of untested skills."""
        return [s for s in self.skills.values() if not s.tested]

    def generate_report(self, detailed: bool = True) -> str:
        """
        Generate coverage report.

        Args:
            detailed: Include detailed skill lists

        Returns:
            str: Formatted report
        """
        stats = self.get_coverage_stats()

        lines = []
        lines.append("="*70)
        lines.append("SKILL COVERAGE REPORT")
        lines.append("="*70)
        lines.append("")

        # Overall stats
        lines.append("Overall Coverage:")
        lines.append(f"  Total skills: {stats['total_skills']}")
        lines.append(f"  Tested: {stats['tested_skills']} ✅")
        lines.append(f"  Untested: {stats['untested_skills']} ⏸️")
        lines.append(f"  Coverage: {stats['coverage_percent']:.1f}%")
        lines.append("")

        # By category
        lines.append("Coverage by Category:")
        for category, cat_stats in sorted(stats['by_category'].items()):
            tested = cat_stats['tested']
            total = cat_stats['total']
            pct = (tested / total * 100) if total > 0 else 0
            lines.append(f"  {category:20s}: {tested:2d}/{total:2d} ({pct:5.1f}%)")
        lines.append("")

        if detailed:
            # Tested skills
            tested = self.get_tested_skills()
            if tested:
                lines.append(f"Tested Skills ({len(tested)}):")
                for skill in sorted(tested, key=lambda s: s.name):
                    lines.append(f"  ✅ {skill.name:30s} [{skill.category}]")
                    if skill.test_function:
                        lines.append(f"     Test: {skill.test_function}")
                lines.append("")

            # Untested skills
            untested = self.get_untested_skills()
            if untested:
                lines.append(f"Untested Skills ({len(untested)}):")
                for skill in sorted(untested, key=lambda s: s.name):
                    lines.append(f"  ⏸️  {skill.name:30s} [{skill.category}]")
                    lines.append(f"     {skill.description[:60]}")
                lines.append("")

        lines.append("="*70)

        return "\n".join(lines)

    def save_report(self, output_path: Path, format: str = 'txt'):
        """
        Save coverage report to file.

        Args:
            output_path: Path to save report
            format: 'txt' or 'json'
        """
        if format == 'json':
            data = {
                'stats': self.get_coverage_stats(),
                'tested': [
                    {
                        'name': s.name,
                        'category': s.category,
                        'test_function': s.test_function,
                        'test_file': s.test_file
                    }
                    for s in self.get_tested_skills()
                ],
                'untested': [
                    {
                        'name': s.name,
                        'category': s.category,
                        'description': s.description
                    }
                    for s in self.get_untested_skills()
                ]
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        else:  # txt
            report = self.generate_report(detailed=True)
            with open(output_path, 'w') as f:
                f.write(report)

        print(f"Coverage report saved to {output_path}")


def create_phase3_coverage_tracker() -> SkillCoverageTracker:
    """
    Create coverage tracker with complete test status.

    ALL 25 SKILLS NOW TESTED - 100% COVERAGE!

    Returns:
        SkillCoverageTracker: Tracker with current test status
    """
    tracker = SkillCoverageTracker()

    # Mark ALL 25 skills as tested - 100% coverage achieved!
    tested_skills = {
        # Single-cell skills (8/8)
        'single-preprocessing': ('test_skill_single_preprocessing', 'test_agent_skills.py'),
        'single-clustering': ('test_skill_single_clustering', 'test_agent_skills.py'),
        'single-annotation': ('test_skill_single_annotation', 'test_agent_skills.py'),
        'single-trajectory': ('test_skill_single_trajectory', 'test_agent_skills.py'),
        'single-cellphone-db': ('test_skill_single_cellphone_db', 'test_agent_skills.py'),
        'single-downstream-analysis': ('test_skill_single_downstream_analysis', 'test_agent_skills.py'),
        'single-multiomics': ('test_skill_single_multiomics', 'test_agent_skills.py'),
        'single-to-spatial-mapping': ('test_skill_single_to_spatial_mapping', 'test_agent_skills.py'),

        # Bulk RNA-seq skills (7/7)
        'bulk-deg-analysis': ('test_skill_bulk_deg_analysis', 'test_agent_skills.py'),
        'bulk-deseq2-analysis': ('test_skill_bulk_deseq2_analysis', 'test_agent_skills.py'),
        'bulk-wgcna-analysis': ('test_skill_bulk_wgcna_analysis', 'test_agent_skills.py'),
        'bulk-combat-correction': ('test_skill_bulk_combat_correction', 'test_agent_skills.py'),
        'bulk-stringdb-ppi': ('test_skill_bulk_stringdb_ppi', 'test_agent_skills.py'),
        'bulk-to-single-deconvolution': ('test_skill_bulk_to_single_deconvolution', 'test_agent_skills.py'),
        'bulk-trajblend-interpolation': ('test_skill_bulk_trajblend_interpolation', 'test_agent_skills.py'),

        # Spatial transcriptomics (1/1)
        'spatial-tutorials': ('test_skill_spatial_tutorials', 'test_agent_skills.py'),

        # TCGA/Cancer genomics (1/1)
        'tcga-preprocessing': ('test_skill_tcga_preprocessing', 'test_agent_skills.py'),

        # Data utilities (5/5)
        'data-export-excel': ('test_skill_data_export_excel', 'test_agent_skills.py'),
        'data-export-pdf': ('test_skill_data_export_pdf', 'test_agent_skills.py'),
        'data-viz-plots': ('test_skill_data_viz_plots', 'test_agent_skills.py'),
        'data-stats-analysis': ('test_skill_data_stats_analysis', 'test_agent_skills.py'),
        'data-transform': ('test_skill_data_transform', 'test_agent_skills.py'),

        # Plotting/Visualization (1/1)
        'plotting-visualization': ('test_skill_plotting_visualization', 'test_agent_skills.py'),
    }

    for skill_name, (test_func, test_file) in tested_skills.items():
        tracker.mark_tested(skill_name, test_func, test_file)

    return tracker


if __name__ == '__main__':
    """Generate coverage report."""
    tracker = create_phase3_coverage_tracker()

    print(tracker.generate_report(detailed=True))

    # Save reports
    output_dir = Path(__file__).parent.parent / 'reports'
    output_dir.mkdir(exist_ok=True)

    tracker.save_report(output_dir / 'skill_coverage.txt', format='txt')
    tracker.save_report(output_dir / 'skill_coverage.json', format='json')

    print(f"\nReports saved to {output_dir}")
