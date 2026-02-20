"""
Services Package - Business Logic Layer
========================================
Service modules for OmicVerse web application.
"""

from .kernel_service import (
    InProcessKernelExecutor,
    normalize_kernel_id,
    build_kernel_namespace,
    reset_kernel_namespace,
    get_kernel_context,
    get_execution_state,
    request_interrupt,
    execution_state,
    execution_state_lock,
)

from .agent_service import (
    get_agent_instance,
    run_agent_stream,
    run_agent_chat,
    agent_requires_adata,
)

__all__ = [
    # Kernel service
    'InProcessKernelExecutor',
    'normalize_kernel_id',
    'build_kernel_namespace',
    'reset_kernel_namespace',
    'get_kernel_context',
    'get_execution_state',
    'request_interrupt',
    'execution_state',
    'execution_state_lock',
    # Agent service
    'get_agent_instance',
    'run_agent_stream',
    'run_agent_chat',
    'agent_requires_adata',
]
