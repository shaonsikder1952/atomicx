"""Causal Discovery Engine — discovers causal relationships between variables.

This package implements multiple causal discovery algorithms:
1. NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented
   lagrangian for Structure learning) — continuous DAG learning
2. PC Algorithm — constraint-based causal discovery
3. DoWhy — causal effect estimation and refutation
4. Time-varying Granger Causality — temporal causal detection
5. Causal Chain Storage — persistent causal graph with reasoning

The engine discovers WHICH variables CAUSE price movement (not just correlate)
and maintains a live causal DAG that updates as new data arrives.
"""
