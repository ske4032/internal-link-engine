"""Bolt queries - the only module in the codebase that talks to Neo4j.

Every Cypher statement and every use of the ``neo4j`` async driver lives here.
Writes are batched with ``UNWIND`` at roughly 500 pages per transaction, and the
``embeddedContentHash`` resume marker is set in the same transaction as the data
it marks, so an interrupted run resumes without re-embedding. Neo4j internal ids
are not stable across restarts, so the id mapping is rebuilt every run and never
cached.

Implemented by issue #3.
"""
