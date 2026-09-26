"""
Taxonomy and planted ground truth.

Everything the corpus asserts against is declared here. If you change a topic,
a spread, or a bridge gap, the assertion suite changes with it — so treat this
file as the specification, not as configuration.
"""

from __future__ import annotations

DIM = 2048  # matches voyage-4-large output_dimension and the Neo4j vector index


# ── topics ──────────────────────────────────────────────────────────────────
# `spread` controls cluster density in the synthetic embedding space. It is the
# variable Leiden's global resolution parameter cannot adapt to and HDBSCAN can,
# so the range here is what makes that comparison meaningful.

TOPICS: dict[str, dict] = {
    "hydraulic": {
        "head": "industrial hydraulic press",
        "spread": 0.22,
        "subtopics": {
            "tonnage": ["press tonnage", "force rating", "cylinder bore"],
            "tooling": ["die cushion", "press ram", "tool holder"],
            "sizing": ["press bed size", "daylight opening", "stroke length"],
        },
        "verbs": ["forming", "stamping", "deep drawing"],
    },
    "pressbrake": {
        "head": "press brake machine",
        "spread": 0.18,  # tightest: product pages
        "subtopics": {
            "bending": ["bending force", "bend allowance", "air bending"],
            "gauging": ["back gauge", "crowning system", "angle measurement"],
        },
        "verbs": ["bending", "folding", "bottoming"],
    },
    "maintenance": {
        "head": "press maintenance schedule",
        "spread": 0.55,  # most diffuse: long-form blog
        "subtopics": {
            "fluids": ["hydraulic fluid", "oil analysis", "filtration"],
            "wear": ["seal replacement", "wear inspection", "bearing play"],
            "scheduling": ["servicing interval", "downtime reduction", "planned outage"],
        },
        "verbs": ["servicing", "inspecting", "lubricating"],
    },
    "safety": {
        "head": "machine guarding standards",
        "spread": 0.48,
        "subtopics": {
            "guarding": ["light curtain", "safety interlock", "fixed guard"],
            "compliance": ["risk assessment", "guarding standard", "conformity"],
        },
        "verbs": ["guarding", "isolating", "certifying"],
    },
    "materials": {
        "head": "sheet metal grades",
        "spread": 0.30,
        "subtopics": {
            "grades": ["stainless grade", "aluminium alloy", "mild steel"],
            "properties": ["tensile strength", "springback", "material thickness"],
        },
        "verbs": ["cutting", "shearing", "specifying"],
    },
}


# ── planted structure ───────────────────────────────────────────────────────

# Topic pairs that SHOULD be linked: they share query vocabulary but have
# near-zero link density. Hub-to-hub bridge scoring must rank these top.
BRIDGE_GAPS: list[tuple[str, str]] = [
    ("pressbrake", "materials"),
    ("hydraulic", "safety"),
]

# Densely bridged already, for contrast.
WELL_CONNECTED: list[tuple[str, str]] = [("hydraulic", "maintenance")]

# Topics whose declared pillar is deliberately displaced from the cluster
# centroid. The centroid-nearest page should NOT be the declared pillar.
PILLAR_MISMATCH: frozenset[str] = frozenset({"maintenance", "safety"})

# Fraction of pages whose body deliberately omits their own head term. Drives
# the extraction ladder past rung 1 into variants, Jaccard, semantic and
# eventually CONTENT_GAP.
NO_HEAD_TERM_RATE = 0.22

# Cross-topic link probabilities.
P_CROSS_BRIDGE_GAP = 0.01
P_CROSS_WELL_CONNECTED = 0.55
P_CROSS_BASELINE = 0.28


# ── anchors ─────────────────────────────────────────────────────────────────
# Sourced from accessibility link-text guidance (WCAG 2.4.4 / 2.4.9), which is
# the same defect measured for a different reason.

GENERIC_ANCHORS: list[str] = [
    "click here", "here", "read more", "learn more", "this page",
    "find out more", "see more", "more info", "more information",
    "continue reading", "view more", "check it out", "details",
]


# ── utility pages ───────────────────────────────────────────────────────────
# These exist on a real site and are reachable through global navigation, but
# nothing in body copy links to them. The crawler extracts body links only, so
# they appear as orphans — which is correct. Template links are an SEO concern
# audited per template, not per page.

NAV_TARGETS: list[tuple[str, str, str]] = [
    ("blog", "Blog", "Index of every article we publish."),
    ("docs", "Documentation", "Product manuals and specification sheets."),
    ("contact", "Contact", "Reach the sales and support teams."),
    ("about", "About", "Who we are and how we got here."),
    ("support", "Support", "Warranty claims and service requests."),
    ("news", "News", "Company announcements and press coverage."),
]

FOOTER_TARGETS: list[tuple[str, str]] = [
    ("terms", "Terms of Service"), ("privacy", "Privacy Policy"),
    ("cookies", "Cookie Notice"), ("careers", "Careers"),
    ("sitemap", "Sitemap"), ("accessibility", "Accessibility"),
    ("returns", "Returns Policy"), ("warranty", "Warranty"),
]

# Pages that belong to no coherent topic but DO receive body links. Leiden must
# place them in a community; HDBSCAN should label them -1.
NOISE_SUBJECTS: list[tuple[str, str]] = [
    ("company-history", "Our company was founded in 1974 by two engineers."),
    ("factory-tour", "A walkthrough of our manufacturing floor."),
    ("trade-show-2025", "Highlights from our stand at the spring exhibition."),
    ("charity-partnership", "Supporting engineering apprenticeships locally."),
    ("office-relocation", "Our head office has moved to a new site."),
    ("staff-profiles", "Meet the people behind the workshop."),
]


# ── GSC ─────────────────────────────────────────────────────────────────────
# CTR by position. Synthetic corpus ships one; real tenants derive their own,
# because curves differ substantially by vertical and SERP feature mix.

CTR_CURVE: dict[int, float] = {
    1: 0.284, 2: 0.152, 3: 0.096, 4: 0.068, 5: 0.050,
    6: 0.039, 7: 0.031, 8: 0.025, 9: 0.021, 10: 0.018,
}

PAGE_TYPE_IMPRESSION_BASE: dict[str, float] = {
    "PILLAR": 9.0, "CATEGORY": 7.6, "PRODUCT": 6.8, "ARTICLE": 6.5,
}

PAGE_TYPE_POSITION_SCALE: dict[str, float] = {
    "PILLAR": 3.5, "CATEGORY": 5.5, "PRODUCT": 7.0, "ARTICLE": 8.5,
}


def ctr_at(position: float) -> float:
    """Power-law decay beyond position 10, table lookup within it."""
    p = int(round(position))
    if p in CTR_CURVE:
        return CTR_CURVE[p]
    return max(0.0005, 0.33 / (position ** 1.15))
