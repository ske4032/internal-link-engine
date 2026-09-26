"""Graph persistence and graph algorithms, deliberately split.

:mod:`linking_engine.graph.repo` is the only module that speaks Bolt.
:mod:`linking_engine.graph.algorithms` holds pure functions and may never import
a database driver. The split is enforced by the ``algorithms-are-pure`` contract.
"""
