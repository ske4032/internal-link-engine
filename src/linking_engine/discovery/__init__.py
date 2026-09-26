"""Stage 2: candidate retrieval, feature assembly, and ranking.

Eligibility is ``isIndexable AND source != target AND NOT already linked`` and
nothing else. Every other consideration - impressions, position, page age,
saturation - is a feature for the ranker, not a filter upstream of it.
"""
