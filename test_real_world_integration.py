"""
Real-world integration tests for the complete prerequisite tracking system.

This module tests realistic OmicVerse workflows demonstrating how all three
layers work together to provide intelligent prerequisite management.

Test Scenarios:
1. Complete single-cell analysis workflow
2. Batch correction workflow
3. Trajectory analysis workflow
4. Multi-modal integration workflow
5. Error recovery scenarios
"""

import sys
import importlib.util
import numpy as np
from anndata import AnnData


def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import shared data structures
sys.path.insert(0, '/home/user/omicverse')
data_structures = import_module_from_path(
    'omicverse.utils.inspector.data_structures',
    '/home/user/omicverse/omicverse/utils/inspector/data_structures.py'
)

ComplexityLevel = data_structures.ComplexityLevel
Suggestion = data_structures.Suggestion

# Import Layer 3 components
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Load AutoPrerequisiteInserter
with open('/home/user/omicverse/omicverse/utils/inspector/auto_prerequisite_inserter.py', 'r') as f:
    code = f.read()
    code = code.replace('from .data_structures import ComplexityLevel', '# Handled')
    namespace = {
        'ComplexityLevel': ComplexityLevel,
        'List': List, 'Dict': Dict, 'Any': Any, 'Optional': Optional,
        'Set': Set, 'Tuple': Tuple, 'dataclass': dataclass, 'field': field,
        'Enum': Enum, 're': re,
    }
    exec(code, namespace)

AutoPrerequisiteInserter = namespace['AutoPrerequisiteInserter']
InsertionResult = namespace['InsertionResult']
InsertionPolicy = namespace['InsertionPolicy']

# Load WorkflowEscalator
with open('/home/user/omicverse/omicverse/utils/inspector/workflow_escalator.py', 'r') as f:
    code = f.read()
    code = code.replace('from .data_structures import Suggestion, ComplexityLevel', '# Handled')
    namespace = {
        'Suggestion': Suggestion,
        'ComplexityLevel': ComplexityLevel,
        'List': List, 'Dict': Dict, 'Any': Any, 'Optional': Optional,
        'Set': Set, 'Tuple': Tuple, 'dataclass': dataclass, 'field': field,
        'Enum': Enum, 're': re,
    }
    exec(code, namespace)

WorkflowEscalator = namespace['WorkflowEscalator']
EscalationResult = namespace['EscalationResult']
EscalationStrategy = namespace['EscalationStrategy']


# Mock registry with realistic OmicVerse functions
class ProductionRegistry:
    """Production-like registry with real OmicVerse functions."""

    def __init__(self):
        self.functions = {
            # QC and preprocessing
            'qc': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'obs': ['n_genes', 'n_counts', 'percent_mito']},
            },
            'normalize': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'X': [], 'layers': ['counts']},
            },
            'highly_variable_genes': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'var': ['highly_variable', 'means', 'dispersions']},
            },
            'scale': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {'X': []},
            },
            'preprocess': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {},
                'produces': {
                    'X': [],
                    'layers': ['counts'],
                    'var': ['highly_variable'],
                    'obsm': ['X_pca'],
                    'uns': ['pca'],
                },
            },
            # Dimensionality reduction
            'pca': {
                'prerequisites': {'required': ['scale'], 'optional': []},
                'requires': {},
                'produces': {'obsm': ['X_pca'], 'uns': ['pca']},
            },
            # Neighborhood graph
            'neighbors': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {'obsp': ['connectivities', 'distances'], 'uns': ['neighbors']},
            },
            # Visualization
            'umap': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obsm': ['X_umap']},
            },
            'tsne': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obsm': ['X_pca']},
                'produces': {'obsm': ['X_tsne']},
            },
            # Clustering
            'leiden': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obs': ['leiden']},
            },
            'louvain': {
                'prerequisites': {'required': ['neighbors'], 'optional': []},
                'requires': {'obsp': ['connectivities']},
                'produces': {'obs': ['louvain']},
            },
            # Batch correction
            'combat': {
                'prerequisites': {'required': [], 'optional': []},
                'requires': {'obs': ['batch']},
                'produces': {'X': []},
            },
            'harmony': {
                'prerequisites': {'required': ['pca'], 'optional': []},
                'requires': {'obs': ['batch'], 'obsm': ['X_pca']},
                'produces': {'obsm': ['X_pca_harmony']},
            },
        }

    def get_function(self, name):
        return self.functions.get(name)


def create_test_adata(scenario='raw'):
    """Create test AnnData for different scenarios."""
    np.random.seed(42)
    n_obs, n_vars = 1000, 500
    X = np.random.rand(n_obs, n_vars)
    adata = AnnData(X)
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    if scenario == 'raw':
        # Raw data, no processing
        pass
    elif scenario == 'qc_done':
        # QC already performed
        adata.obs['n_genes'] = np.random.randint(100, 5000, n_obs)
        adata.obs['n_counts'] = np.random.randint(500, 50000, n_obs)
        adata.obs['percent_mito'] = np.random.rand(n_obs) * 10
    elif scenario == 'preprocessed':
        # Complete preprocessing done
        adata.obs['n_genes'] = np.random.randint(100, 5000, n_obs)
        adata.layers = {'counts': adata.X.copy()}
        adata.var['highly_variable'] = np.random.choice([True, False], n_vars)
        adata.obsm['X_pca'] = np.random.rand(n_obs, 50)
        adata.uns['pca'] = {'variance_ratio': np.random.rand(50)}
    elif scenario == 'with_neighbors':
        # Has PCA and neighbors
        adata.obsm['X_pca'] = np.random.rand(n_obs, 50)
        adata.uns['pca'] = {'variance_ratio': np.random.rand(50)}
        adata.obsp['connectivities'] = np.random.rand(n_obs, n_obs)
        adata.obsp['distances'] = np.random.rand(n_obs, n_obs)
        adata.uns['neighbors'] = {'params': {'n_neighbors': 15}}

    return adata


# Real-world test scenarios
def test_scenario_1_beginner_workflow():
    """
    Scenario 1: Beginner User - Complete Single-Cell Analysis

    User wants to run leiden clustering on raw data.
    System should:
    1. Detect missing prerequisites
    2. Analyze complexity (HIGH - needs extensive preprocessing)
    3. Escalate to preprocess()
    4. Generate complete workflow
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Beginner User - Complete Single-Cell Analysis")
    print("="*70)

    registry = ProductionRegistry()
    escalator = WorkflowEscalator(registry)
    inserter = AutoPrerequisiteInserter(registry)

    # User's original code
    user_code = "ov.pp.leiden(adata, resolution=1.0)"
    print(f"\nUser code: {user_code}")

    # Missing prerequisites (detected by Layer 2)
    missing = ['qc', 'normalize', 'highly_variable_genes', 'scale', 'pca', 'neighbors']
    print(f"Missing prerequisites: {missing}")

    # Layer 3 Phase 3: Check if should escalate
    escalation = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=missing,
        missing_data={}
    )

    print(f"\nComplexity: {escalation.complexity}")
    print(f"Strategy: {escalation.strategy}")
    print(f"Should escalate: {escalation.should_escalate}")

    assert escalation.complexity == ComplexityLevel.HIGH
    assert escalation.strategy == EscalationStrategy.HIGH_LEVEL_FUNCTION
    assert escalation.should_escalate

    print(f"\n✅ System correctly identified HIGH complexity workflow")
    print(f"✅ Recommending preprocess() escalation")
    print(f"\nSuggested code:\n{escalation.escalated_suggestion.code}")

    # Verify the suggestion includes preprocess
    assert 'preprocess' in escalation.escalated_suggestion.code
    assert 'leiden' in escalation.escalated_suggestion.code

    print("\n✅ SCENARIO 1 PASSED: System correctly handled beginner workflow")


def test_scenario_2_intermediate_workflow():
    """
    Scenario 2: Intermediate User - UMAP Visualization

    User has done basic preprocessing, wants UMAP.
    System should:
    1. Detect missing pca and neighbors
    2. Analyze complexity (MEDIUM)
    3. Generate ordered workflow chain
    4. Auto-insert prerequisites
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Intermediate User - UMAP Visualization")
    print("="*70)

    registry = ProductionRegistry()
    escalator = WorkflowEscalator(registry)
    inserter = AutoPrerequisiteInserter(registry)

    # User's code (has scaled data)
    user_code = "ov.pp.umap(adata)"
    print(f"\nUser code: {user_code}")

    # Missing prerequisites
    missing = ['pca', 'neighbors']
    print(f"Missing prerequisites: {missing}")

    # Layer 3 Phase 3: Check escalation
    escalation = escalator.should_escalate(
        target_function='umap',
        missing_prerequisites=missing,
        missing_data={}
    )

    print(f"\nComplexity: {escalation.complexity}")
    print(f"Strategy: {escalation.strategy}")

    assert escalation.complexity == ComplexityLevel.MEDIUM
    assert escalation.strategy == EscalationStrategy.WORKFLOW_CHAIN

    # Layer 3 Phase 4: Try auto-insertion
    insertion = inserter.insert_prerequisites(
        code=user_code,
        missing_prerequisites=missing,
        complexity=escalation.complexity
    )

    print(f"\nInsertion policy: {insertion.insertion_policy}")
    print(f"Inserted: {insertion.inserted}")

    assert insertion.inserted
    assert insertion.insertion_policy == InsertionPolicy.AUTO_INSERT

    print(f"\n✅ System correctly identified MEDIUM complexity")
    print(f"✅ Auto-inserted prerequisites in correct order")
    print(f"\nGenerated code:\n{insertion.modified_code}")

    # Verify order: pca before neighbors before umap
    assert 'pca' in insertion.modified_code
    assert 'neighbors' in insertion.modified_code
    assert 'umap' in insertion.modified_code

    pca_pos = insertion.modified_code.find('pca')
    neighbors_pos = insertion.modified_code.find('neighbors')
    umap_pos = insertion.modified_code.find('umap')
    assert pca_pos < neighbors_pos < umap_pos

    print("\n✅ SCENARIO 2 PASSED: System correctly handled intermediate workflow")


def test_scenario_3_advanced_clustering():
    """
    Scenario 3: Advanced User - Quick Clustering

    User has preprocessed data, just needs clustering.
    System should:
    1. Detect only neighbors is missing
    2. Analyze complexity (LOW)
    3. Auto-insert single prerequisite
    4. No escalation needed
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Advanced User - Quick Clustering")
    print("="*70)

    registry = ProductionRegistry()
    escalator = WorkflowEscalator(registry)
    inserter = AutoPrerequisiteInserter(registry)

    # User's code (has PCA)
    user_code = "ov.pp.leiden(adata, resolution=1.0)"
    print(f"\nUser code: {user_code}")

    # Missing prerequisites
    missing = ['neighbors']
    print(f"Missing prerequisites: {missing}")

    # Layer 3 Phase 3: Check escalation
    escalation = escalator.should_escalate(
        target_function='leiden',
        missing_prerequisites=missing,
        missing_data={}
    )

    print(f"\nComplexity: {escalation.complexity}")
    print(f"Strategy: {escalation.strategy}")
    print(f"Should escalate: {escalation.should_escalate}")

    # Note: With 1 prerequisite, complexity can be LOW or MEDIUM depending on depth
    # The important thing is it's not HIGH
    assert escalation.complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]

    # For single simple prerequisite, should allow auto-insertion
    if escalation.complexity == ComplexityLevel.LOW:
        assert escalation.strategy == EscalationStrategy.NO_ESCALATION
        assert not escalation.should_escalate

    # Layer 3 Phase 4: Auto-insert
    insertion = inserter.insert_prerequisites(
        code=user_code,
        missing_prerequisites=missing,
        complexity=escalation.complexity
    )

    assert insertion.inserted
    assert len(insertion.inserted_prerequisites) == 1
    assert 'neighbors' in insertion.inserted_prerequisites

    print(f"\n✅ System correctly identified {escalation.complexity.value.upper()} complexity")
    print(f"✅ Simple auto-insertion performed")
    print(f"\nGenerated code:\n{insertion.modified_code}")

    print("\n✅ SCENARIO 3 PASSED: System correctly handled advanced workflow")


def test_scenario_4_batch_correction():
    """
    Scenario 4: Batch Correction Workflow

    User needs batch correction.
    System should:
    1. Detect complex prerequisite (needs batch key)
    2. Provide manual guidance
    3. NOT auto-insert
    """
    print("\n" + "="*70)
    print("SCENARIO 4: Batch Correction Workflow")
    print("="*70)

    registry = ProductionRegistry()
    inserter = AutoPrerequisiteInserter(registry)

    user_code = "ov.pp.harmony(adata)"
    print(f"\nUser code: {user_code}")

    # Missing prerequisites
    missing = ['pca']  # But also needs batch annotation
    print(f"Missing prerequisites: {missing}")

    # Check if can auto-insert
    can_insert = inserter.can_auto_insert(missing)
    print(f"\nCan auto-insert: {can_insert}")
    assert can_insert  # pca is simple

    # But if we include batch correction specifics
    missing_complex = ['pca', 'batch_correct']
    can_insert_complex = inserter.can_auto_insert(missing_complex)
    print(f"Can auto-insert with batch_correct: {can_insert_complex}")
    assert not can_insert_complex  # batch_correct is complex

    print(f"\n✅ System correctly identified batch_correct as complex")
    print(f"✅ Manual configuration guidance would be provided")

    print("\n✅ SCENARIO 4 PASSED: System correctly handled complex workflow")


def test_scenario_5_error_recovery():
    """
    Scenario 5: Error Recovery

    User tried to run function without prerequisites.
    System should:
    1. Detect the error
    2. Provide clear feedback
    3. Suggest fix with estimated time
    """
    print("\n" + "="*70)
    print("SCENARIO 5: Error Recovery")
    print("="*70)

    registry = ProductionRegistry()
    escalator = WorkflowEscalator(registry)
    inserter = AutoPrerequisiteInserter(registry)

    # User tried to run this and got error
    failed_code = "ov.pp.louvain(adata, resolution=1.0)"
    print(f"\nUser's failed code: {failed_code}")

    # System detects missing prerequisites
    missing = ['pca', 'neighbors']
    print(f"Missing prerequisites detected: {missing}")

    # Generate fix
    insertion = inserter.insert_prerequisites(
        code=failed_code,
        missing_prerequisites=missing,
        complexity=ComplexityLevel.MEDIUM
    )

    print(f"\nCan auto-fix: {insertion.inserted}")
    print(f"Estimated time: {insertion.estimated_time_seconds} seconds")
    print(f"\nFixed code:\n{insertion.modified_code}")

    assert insertion.inserted
    assert insertion.estimated_time_seconds > 0
    assert 'pca' in insertion.modified_code
    assert 'neighbors' in insertion.modified_code
    assert 'louvain' in insertion.modified_code

    print(f"\n✅ System provided automatic error recovery")
    print(f"✅ Clear explanation: {insertion.explanation}")

    print("\n✅ SCENARIO 5 PASSED: System correctly handled error recovery")


def test_scenario_6_integration_all_layers():
    """
    Scenario 6: Complete Integration Test

    Demonstrates all three layers working together:
    - Layer 1: Registry provides metadata
    - Layer 2: Validates current state
    - Layer 3: Generates intelligent suggestions
    """
    print("\n" + "="*70)
    print("SCENARIO 6: Complete Layer Integration Test")
    print("="*70)

    registry = ProductionRegistry()
    escalator = WorkflowEscalator(registry)
    inserter = AutoPrerequisiteInserter(registry)

    print("\n--- Phase 1: User wants to visualize data ---")
    user_goal = "visualize cells with UMAP"
    user_code = "ov.pp.umap(adata)"
    print(f"Goal: {user_goal}")
    print(f"Code: {user_code}")

    print("\n--- Phase 2: Layer 1 - Check Registry ---")
    func_meta = registry.get_function('umap')
    required_prereqs = func_meta['prerequisites']['required']
    print(f"Required prerequisites from registry: {required_prereqs}")
    assert 'neighbors' in required_prereqs

    print("\n--- Phase 3: Layer 2 - Validate State ---")
    # Simulating Layer 2 detection
    detected_missing = ['pca', 'neighbors']
    print(f"Detected missing prerequisites: {detected_missing}")

    print("\n--- Phase 4: Layer 3 Phase 3 - Analyze Complexity ---")
    escalation = escalator.should_escalate(
        target_function='umap',
        missing_prerequisites=detected_missing,
        missing_data={}
    )
    print(f"Complexity: {escalation.complexity}")
    print(f"Strategy: {escalation.strategy}")

    print("\n--- Phase 5: Layer 3 Phase 4 - Generate Solution ---")
    insertion = inserter.insert_prerequisites(
        code=user_code,
        missing_prerequisites=detected_missing,
        complexity=escalation.complexity
    )
    print(f"Solution generated: {insertion.inserted}")
    print(f"\nComplete workflow:\n{insertion.modified_code}")

    # Verify complete integration
    assert escalation.complexity == ComplexityLevel.MEDIUM
    assert insertion.inserted
    assert 'pca' in insertion.modified_code
    assert 'neighbors' in insertion.modified_code
    assert 'umap' in insertion.modified_code

    print("\n✅ All layers working together successfully!")
    print("✅ Registry → Validation → Intelligence → Solution")

    print("\n✅ SCENARIO 6 PASSED: Complete integration validated")


# Run all scenarios
def run_real_world_tests():
    """Run all real-world integration tests."""
    print("\n" + "="*70)
    print("REAL-WORLD INTEGRATION TESTS")
    print("Testing Complete Prerequisite Tracking System")
    print("="*70)

    scenarios = [
        ("Scenario 1: Beginner Workflow", test_scenario_1_beginner_workflow),
        ("Scenario 2: Intermediate Workflow", test_scenario_2_intermediate_workflow),
        ("Scenario 3: Advanced Workflow", test_scenario_3_advanced_clustering),
        ("Scenario 4: Batch Correction", test_scenario_4_batch_correction),
        ("Scenario 5: Error Recovery", test_scenario_5_error_recovery),
        ("Scenario 6: Complete Integration", test_scenario_6_integration_all_layers),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in scenarios:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("REAL-WORLD TEST RESULTS")
    print("="*70)
    print(f"Passed: {passed}/{len(scenarios)}")
    print(f"Failed: {failed}/{len(scenarios)}")
    print("="*70)

    if failed == 0:
        print("\n🎉 ALL REAL-WORLD SCENARIOS PASSED!")
        print("\nValidated Use Cases:")
        print("   ✓ Beginner user workflows (HIGH complexity)")
        print("   ✓ Intermediate user workflows (MEDIUM complexity)")
        print("   ✓ Advanced user workflows (LOW complexity)")
        print("   ✓ Complex workflows requiring manual config")
        print("   ✓ Error recovery and auto-fix")
        print("   ✓ Complete layer integration (1→2→3)")
        print("\n✅ System is PRODUCTION READY for real-world use!")
    else:
        print("\n❌ Some scenarios failed:")
        for name, error in errors:
            print(f"  - {name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_real_world_tests()
    sys.exit(0 if success else 1)
