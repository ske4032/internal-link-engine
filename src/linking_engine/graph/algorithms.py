"""Pure graph algorithms - igraph, leidenalg, hdbscan - run in-process.

Functions here take edge lists and return arrays. They never open a connection,
never issue a query, and never import ``neo4j``, ``motor`` or ``pymongo``; the
``algorithms-are-pure`` import-linter contract makes that structural rather than
conventional. Running outside the database avoids the Community Edition
four-core cap, the three-projection limit, and the JVM cache misses that
dominate betweenness on a heap-object graph.

Implemented by issue #3.
"""
