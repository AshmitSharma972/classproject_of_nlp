"""
src/negation/__init__.py

Public surface of the negation / assertion module.

Exports
-------
AssertionEngine
    The main entry point for classifying entities with assertion status
    (PRESENT, NEGATED, POSSIBLE, HISTORICAL, FAMILY) and experiencer
    (PATIENT, OTHER).

Example usage::

    from src.negation import AssertionEngine

    engine = AssertionEngine()
    assertions = engine.run(entities, preprocessing_output)
"""

from src.negation.assertion_engine import AssertionEngine

__all__ = ["AssertionEngine"]
