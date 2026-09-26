"""Stage 1: scoring and verdicts for links that already exist.

Scores every stored edge across six dimensions and maps the result to an
:class:`~linking_engine.models.enums.ActionType` verdict. Runs independently of
discovery - ``auditEnabled`` and ``discoveryEnabled`` are separate tenant flags.
"""
