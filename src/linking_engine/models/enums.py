"""Closed vocabularies shared across every module boundary.

All of these are ``StrEnum``, so a member serialises to its own name and a stored
Neo4j or Mongo value validates straight back into the enum without a converter.
"""

from enum import StrEnum


class ActionType(StrEnum):
    """The verdict attached to a `SUGGESTED_ACTION` edge.

    Five action types, not six. There is deliberately no ``REPOSITION``: the
    crawler extracts body links only and discards nav, header, footer and
    sidebar links at extraction, so nothing downstream ever sees a footer link
    to move a link out of.
    """

    ADD_LINK = "ADD_LINK"
    REANCHOR = "REANCHOR"
    REMOVE = "REMOVE"
    FIX = "FIX"
    CONTENT_GAP = "CONTENT_GAP"


class AnchorType(StrEnum):
    """How closely an anchor phrase matches the target's resolved keyword.

    The tenant's type distribution is a preference, not a constraint: extraction
    runs first and the profile only chooses among phrases already in the copy.
    """

    EXACT = "EXACT"
    PARTIAL = "PARTIAL"
    NATURAL = "NATURAL"
    BRANDED = "BRANDED"


class IssueFlag(StrEnum):
    """A defect the audit found on an existing `LINKS_TO` edge.

    An edge can carry several at once, so these are collected into a
    ``frozenset`` rather than being mutually exclusive.
    """

    GENERIC = "GENERIC"
    MISALIGNED = "MISALIGNED"
    OFF_TOPIC = "OFF_TOPIC"
    OVER_OPTIMISED = "OVER_OPTIMISED"
    WASTED_EQUITY = "WASTED_EQUITY"
    BROKEN = "BROKEN"
    REDIRECTED = "REDIRECTED"
    NOFOLLOW = "NOFOLLOW"
    NOINDEX_TARGET = "NOINDEX_TARGET"


class LifecycleStage(StrEnum):
    """How far a page has travelled from publication to settled performance.

    Drives the lifecycle boost and the reserved new-page recommendation slots.
    """

    NEW = "NEW"
    EMERGING = "EMERGING"
    ESTABLISHED = "ESTABLISHED"
    MATURE = "MATURE"


class OrphanLabel(StrEnum):
    """What still links to a crawled page that no body link reaches.

    Menus and footer are the template blocks inside the crawled content: above or
    within the main text, and after it. Site-wide header and footer menus are not
    crawled.
    """

    MENUS_ONLY = "MENUS_ONLY"
    FOOTER_ONLY = "FOOTER_ONLY"
    MENUS_AND_FOOTER_ONLY = "MENUS_AND_FOOTER_ONLY"
    NOT_LINKED = "NOT_LINKED"


class PageType(StrEnum):
    """Structural role of a page in the site taxonomy.

    Vocabulary matches the corpus generator, which keys its impression and
    position distributions off exactly these four values.
    """

    PILLAR = "PILLAR"
    CATEGORY = "CATEGORY"
    PRODUCT = "PRODUCT"
    ARTICLE = "ARTICLE"


class KeywordSource(StrEnum):
    """Where a `TARGETS_KEYWORD` edge came from.

    This discriminator is what makes the keyword gap computable:
    ``keyword_gap = CLIENT_STRATEGIC targets - GSC_OBSERVED rankings``.
    """

    CLIENT_STRATEGIC = "CLIENT_STRATEGIC"
    GSC_OBSERVED = "GSC_OBSERVED"
    INFERRED = "INFERRED"


class ContentGapFinding(StrEnum):
    """Why the anchor ladder fell through to ``CONTENT_GAP``.

    Set on a `SUGGESTED_ACTION` edge for ``CONTENT_GAP`` actions only: either the
    source body never mentions the topic, or it does but every candidate phrase
    would read badly as a link.
    """

    NO_TOPICAL_MENTION = "NO_TOPICAL_MENTION"
    AWKWARD_PHRASING = "AWKWARD_PHRASING"


class RecommendationStatus(StrEnum):
    """Where a recommendation stands with the operator who reviewed it.

    ``MODIFIED`` is distinct from ``ACCEPTED`` because anchor feedback separates
    the anchor being usable from the anchor being used verbatim.
    """

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    MODIFIED = "MODIFIED"
    DISMISSED = "DISMISSED"
