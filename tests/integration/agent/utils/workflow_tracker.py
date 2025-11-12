"""
Workflow tracking and validation utilities.

Provides tools to track multi-step agent workflows and validate:
- Step completion
- State preservation between steps
- Expected outputs at each stage
- Workflow correctness
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""

    name: str
    request: str
    expected_outputs: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    completed: bool = False
    execution_time: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

    def validate_inputs(self, adata) -> tuple[bool, str]:
        """
        Check if required inputs are present.

        Args:
            adata: AnnData object to check

        Returns:
            tuple: (valid, message)
        """
        missing = []

        for req in self.required_inputs:
            if '.' in req:
                # Check nested attributes (e.g., "obsm.X_pca")
                parts = req.split('.')
                obj = adata

                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    elif isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        missing.append(req)
                        break
            else:
                # Check top-level attributes
                if not hasattr(adata, req):
                    missing.append(req)

        if missing:
            return False, f"Missing required inputs: {', '.join(missing)}"

        return True, "All required inputs present"

    def validate_outputs(self, adata) -> tuple[bool, List[str]]:
        """
        Check if expected outputs were produced.

        Args:
            adata: AnnData object to check

        Returns:
            tuple: (all_present, list of missing outputs)
        """
        missing = []

        for expected in self.expected_outputs:
            if '.' in expected:
                # Check nested (e.g., "obs.leiden")
                parts = expected.split('.')
                obj = adata

                found = True
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    elif isinstance(obj, dict) and part in obj:
                        obj = obj[part]
                    else:
                        found = False
                        break

                if not found:
                    missing.append(expected)
            else:
                if not hasattr(adata, expected):
                    missing.append(expected)

        return len(missing) == 0, missing


class WorkflowTracker:
    """Track and validate multi-step agent workflows."""

    def __init__(self, name: str = "workflow"):
        """
        Initialize workflow tracker.

        Args:
            name: Name of the workflow
        """
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_step_index: int = -1

    def add_step(
        self,
        name: str,
        request: str,
        expected_outputs: Optional[List[str]] = None,
        required_inputs: Optional[List[str]] = None
    ) -> WorkflowStep:
        """
        Add a step to the workflow.

        Args:
            name: Step name
            request: Agent request string
            expected_outputs: List of expected outputs (e.g., ["obs.leiden"])
            required_inputs: List of required inputs from previous steps

        Returns:
            WorkflowStep: The created step
        """
        step = WorkflowStep(
            name=name,
            request=request,
            expected_outputs=expected_outputs or [],
            required_inputs=required_inputs or []
        )
        self.steps.append(step)
        return step

    def start(self):
        """Start tracking workflow execution."""
        self.start_time = time.time()
        self.current_step_index = -1

    def execute_step(
        self,
        agent,
        adata,
        step_index: Optional[int] = None,
        validate_before: bool = True,
        validate_after: bool = True
    ):
        """
        Execute a workflow step with validation.

        Args:
            agent: OmicVerse agent instance
            adata: AnnData object
            step_index: Index of step to execute (default: next step)
            validate_before: Validate required inputs before execution
            validate_after: Validate expected outputs after execution

        Returns:
            tuple: (result_adata, step)

        Raises:
            ValueError: If validation fails
        """
        if step_index is None:
            step_index = self.current_step_index + 1

        if step_index >= len(self.steps):
            raise ValueError(f"Step index {step_index} out of range (0-{len(self.steps)-1})")

        step = self.steps[step_index]
        self.current_step_index = step_index

        # Validate inputs
        if validate_before and step.required_inputs:
            valid, message = step.validate_inputs(adata)
            if not valid:
                step.error = f"Input validation failed: {message}"
                raise ValueError(step.error)

        # Execute step
        step_start = time.time()

        try:
            result = agent.run(step.request, adata)

            # Extract adata from result
            if isinstance(result, dict):
                result_adata = result.get('adata', result.get('value', adata))
            else:
                result_adata = result

            step.result = result_adata
            step.execution_time = time.time() - step_start
            step.completed = True

            # Validate outputs
            if validate_after and step.expected_outputs:
                all_present, missing = step.validate_outputs(result_adata)

                if not all_present:
                    step.error = f"Output validation failed: missing {', '.join(missing)}"
                    print(f"Warning: {step.error}")
                    # Don't raise error - just warn

            return result_adata, step

        except Exception as e:
            step.execution_time = time.time() - step_start
            step.error = str(e)
            raise

    def finish(self):
        """Mark workflow as finished."""
        self.end_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get workflow execution summary.

        Returns:
            dict: Summary statistics
        """
        total_time = (self.end_time - self.start_time) if self.end_time else None

        completed = sum(1 for step in self.steps if step.completed)
        failed = sum(1 for step in self.steps if step.error)

        return {
            'workflow_name': self.name,
            'total_steps': len(self.steps),
            'completed_steps': completed,
            'failed_steps': failed,
            'total_time': total_time,
            'steps': [
                {
                    'name': step.name,
                    'completed': step.completed,
                    'execution_time': step.execution_time,
                    'error': step.error
                }
                for step in self.steps
            ]
        }

    def print_summary(self):
        """Print workflow execution summary."""
        summary = self.get_summary()

        print(f"\n{'='*70}")
        print(f"Workflow: {summary['workflow_name']}")
        print(f"{'='*70}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Completed: {summary['completed_steps']}")
        print(f"Failed: {summary['failed_steps']}")

        if summary['total_time']:
            print(f"Total time: {summary['total_time']:.2f}s")

        print(f"\nStep details:")
        for i, step_info in enumerate(summary['steps'], 1):
            status = "✅" if step_info['completed'] else "❌"
            time_str = f"{step_info['execution_time']:.2f}s" if step_info['execution_time'] else "N/A"

            print(f"  [{i}] {status} {step_info['name']} ({time_str})")

            if step_info['error']:
                print(f"      Error: {step_info['error']}")

        print(f"{'='*70}\n")

    def save_report(self, output_path: Path):
        """
        Save workflow report to JSON file.

        Args:
            output_path: Path to save report
        """
        summary = self.get_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Workflow report saved to {output_path}")

    def assert_all_completed(self):
        """
        Assert all steps completed successfully.

        Raises:
            AssertionError: If any step failed or incomplete
        """
        incomplete = [step.name for step in self.steps if not step.completed]
        failed = [step.name for step in self.steps if step.error]

        if incomplete:
            raise AssertionError(f"Incomplete steps: {', '.join(incomplete)}")

        if failed:
            raise AssertionError(f"Failed steps: {', '.join(failed)}")


def create_pbmc_preprocessing_workflow() -> WorkflowTracker:
    """
    Create a standard PBMC preprocessing workflow.

    Returns:
        WorkflowTracker: Configured workflow
    """
    workflow = WorkflowTracker("PBMC Preprocessing")

    workflow.add_step(
        name="Quality Control",
        request="perform quality control filtering with nUMI>500, mito<0.2",
        expected_outputs=["obs.n_counts", "obs.n_genes"]
    )

    workflow.add_step(
        name="Normalization & HVG",
        request="normalize and select 2000 highly variable genes",
        expected_outputs=["var.highly_variable"]
    )

    workflow.add_step(
        name="PCA",
        request="compute PCA with 50 components",
        expected_outputs=["obsm.X_pca"]
    )

    workflow.add_step(
        name="Clustering",
        request="compute neighbors and leiden clustering",
        required_inputs=["obsm.X_pca"],
        expected_outputs=["obs.leiden"]
    )

    workflow.add_step(
        name="UMAP",
        request="compute UMAP embedding",
        required_inputs=["obsm.X_pca"],
        expected_outputs=["obsm.X_umap"]
    )

    return workflow


def create_annotation_workflow() -> WorkflowTracker:
    """
    Create a cell type annotation workflow.

    Returns:
        WorkflowTracker: Configured workflow
    """
    workflow = WorkflowTracker("Cell Type Annotation")

    workflow.add_step(
        name="Preprocessing",
        request="preprocess data with QC, normalization, and HVG selection",
        expected_outputs=["var.highly_variable"]
    )

    workflow.add_step(
        name="Clustering",
        request="perform leiden clustering",
        expected_outputs=["obs.leiden"]
    )

    workflow.add_step(
        name="Annotation",
        request="annotate cell types using marker genes for PBMC data",
        required_inputs=["obs.leiden"],
        expected_outputs=[]  # Variable depending on method
    )

    return workflow


def create_deg_workflow() -> WorkflowTracker:
    """
    Create a DEG analysis workflow.

    Returns:
        WorkflowTracker: Configured workflow
    """
    workflow = WorkflowTracker("Differential Expression")

    workflow.add_step(
        name="DEG Analysis",
        request="perform differential expression analysis",
        expected_outputs=[]  # DEG results may be in various formats
    )

    workflow.add_step(
        name="Enrichment",
        request="perform pathway enrichment on significant genes",
        required_inputs=[],
        expected_outputs=[]
    )

    return workflow


if __name__ == '__main__':
    """Test workflow tracker."""
    print("Testing WorkflowTracker...")

    # Create test workflow
    workflow = create_pbmc_preprocessing_workflow()

    print(f"Created workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")

    for i, step in enumerate(workflow.steps, 1):
        print(f"  [{i}] {step.name}")
        print(f"      Request: {step.request[:60]}...")
        print(f"      Expected outputs: {step.expected_outputs}")
        print(f"      Required inputs: {step.required_inputs}")

    print("\n✅ Workflow tracker test complete")
