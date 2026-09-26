"""Internal Linking Intelligence Engine.

A multi-tenant SEO system that audits a site's existing internal links,
discovers missing ones, and derives anchor text from phrases already in the copy.

Module boundaries are enforced by import-linter contracts in ``.importlinter``,
not by convention. Pydantic models in :mod:`linking_engine.models` are the only
thing that crosses a module boundary.
"""
