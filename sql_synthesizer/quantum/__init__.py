"""
Quantum-Inspired SQL Query Optimization

This package implements quantum computing concepts for SQL query optimization,
including superposition, entanglement, and quantum annealing algorithms.
"""

from .core import (
    QuantumQueryOptimizer,
    QuantumQueryPlanGenerator,
    QueryPlan,
    QuantumState,
    Qubit
)
from .scheduler import QuantumTaskScheduler
from .integration import QuantumSQLSynthesizer

__version__ = "1.0.0"
__all__ = [
    "QuantumQueryOptimizer",
    "QuantumQueryPlanGenerator", 
    "QuantumTaskScheduler",
    "QuantumSQLSynthesizer",
    "QueryPlan",
    "QuantumState",
    "Qubit"
]