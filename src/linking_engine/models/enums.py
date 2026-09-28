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
    """How far a page has travelled from publication to settled performance."""

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


class ClusterAgreement(StrEnum):
    """A pair's topic clustering against its link clustering; the disagreement is the signal.

    Two pages match on links only when both belong to the same link community, so a page
    in no link community (an orphan) never matches. The topic is unknown when either page
    has no topic community.
    """

    # Already well connected.
    SAME_TOPIC_SAME_LINKS = "SAME_TOPIC_SAME_LINKS"
    # The missing link.
    SAME_TOPIC_OTHER_LINKS = "SAME_TOPIC_OTHER_LINKS"
    # Linked neighbourhoods across topics: possibly a link to remove.
    OTHER_TOPIC_SAME_LINKS = "OTHER_TOPIC_SAME_LINKS"
    OTHER_TOPIC_OTHER_LINKS = "OTHER_TOPIC_OTHER_LINKS"
    UNKNOWN_TOPIC = "UNKNOWN_TOPIC"


class AnchorRung(StrEnum):
    """Which rung of the extraction ladder found an anchor phrase in the source copy, in order:
    the keyword verbatim, a stemmed variant, an overlapping set of stems, then a phrase whose
    meaning is close to the keyword's."""

    EXACT = "EXACT"
    STEMMED = "STEMMED"
    STEM_SET = "STEM_SET"
    SEMANTIC = "SEMANTIC"


class BridgeReason(StrEnum):
    """Why a hub pair is proposed a bridge."""

    # An edge of the maximum spanning tree over hub centroid similarity: keeps all hubs connected.
    SPANNING_TREE = "SPANNING_TREE"
    # One of a hub's nearest hubs by centroid similarity.
    NEAREST_HUB = "NEAREST_HUB"
    # Among the highest bridge-gap pairs: shared topic or demand, few links.
    BRIDGE_GAP = "BRIDGE_GAP"


class KeywordRung(StrEnum):
    """Which step of the keyword resolution chain chose a page's target keyword."""

    STRATEGIC = "STRATEGIC"
    GSC = "GSC"
    H1 = "H1"
    TITLE = "TITLE"


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


class UnanchoredReason(StrEnum):
    """Why a candidate pair got no anchor, named for the person who has to act on it. The first
    two are content gaps in the source page and the third a gap in the target page; the last two
    say the pair was not fully searched in this run, which is no finding about either page."""

    SOURCE_DOES_NOT_MENTION_TOPIC = "SOURCE_DOES_NOT_MENTION_TOPIC"
    TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE = "TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE"
    TARGET_PAGE_HAS_NO_KEYWORD = "TARGET_PAGE_HAS_NO_KEYWORD"
    SOURCE_PAGE_TEXT_UNAVAILABLE = "SOURCE_PAGE_TEXT_UNAVAILABLE"
    MEANING_SEARCH_NOT_RUN = "MEANING_SEARCH_NOT_RUN"


class ScorerName(StrEnum):
    """What ordered a source page's candidate pairs. The values name MLflow metrics."""

    LEARNED = "learned"
    # The same model trained without the columns that hiding a link moves by itself.
    LEARNED_EXCL_LINK_COUNTS = "learned_excl_link_counts"
    # The same model trained without the anchor placement columns, which a hidden link keeps.
    LEARNED_EXCL_PLACEMENT = "learned_excl_placement"
    # The hand-weighted scorer of #17.
    BASELINE = "baseline"
    # The registered model currently holding the production alias.
    HOLDER = "holder"


class RecommendationStatus(StrEnum):
    """Where a recommendation stands with the operator who reviewed it.

    ``MODIFIED`` is distinct from ``ACCEPTED`` because anchor feedback separates
    the anchor being usable from the anchor being used verbatim.
    """

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    MODIFIED = "MODIFIED"
    DISMISSED = "DISMISSED"
