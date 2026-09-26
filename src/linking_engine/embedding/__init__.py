"""Voyage client, token-aware batching, and the embedding write path.

Batching is by tokens, not list length. Vectors are 2048-dimensional; the
dimension is fixed by the Neo4j vector index and changing it is an ADR, not a PR.
"""
