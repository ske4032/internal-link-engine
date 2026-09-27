"""Markdown cleaning: every noise kind found in the scraped corpus, plus links.

The patterns covered are the ones measured in the real crawl (setext headings,
linked SVG images, breadcrumbs, escaped registry paths, tables), not a guessed
list.
"""

import pytest

from linking_engine.ingest.markdown_clean import (
    body_hash,
    clean_meta,
    clean_page,
    find_boilerplate,
    is_navigation,
    line_shares,
    template_key,
)

PAGE = "https://www.example.com/blog/post/"


def clean(markdown: str, **kwargs):
    return clean_page(markdown, PAGE, **kwargs)


def removed(page) -> dict[str, int]:
    return dict(page.removed)


# ── structure ────────────────────────────────────────────────────────────────


def test_setext_underlines_are_dropped_and_the_equals_heading_becomes_h1() -> None:
    page = clean("Survey Report\n=============\n\nWhat is inside\n--------------\n\nBody.")
    assert page.body_text == "Survey Report\n\nWhat is inside\n\nBody."
    assert page.h1 == "Survey Report"
    assert removed(page)["rule_or_underline"] == 2


@pytest.mark.parametrize("rule", ["---", "***", "* * *", "___", "=========="])
def test_horizontal_rules_are_dropped(rule: str) -> None:
    assert clean(f"Before.\n\n{rule}\n\nAfter.").body_text == "Before.\n\nAfter."


def test_atx_heading_markers_are_dropped_and_the_first_h1_is_kept() -> None:
    page = clean("# Main Title\n\n### Sub section ###\n\ntext")
    assert page.body_text == "Main Title\n\nSub section\n\ntext"
    assert page.h1 == "Main Title"


def test_nested_list_and_blockquote_markers_are_stripped() -> None:
    page = clean("*   *   What do sysadmins read\n> “Quoted line”\n1. First\n- second")
    assert page.body_text == "What do sysadmins read\n“Quoted line”\nFirst\nsecond"


def test_table_rows_are_flattened_and_rules_and_empty_rows_dropped() -> None:
    page = clean("|     |     |\n| --- | --- |\n| Windows | Patch |\n| macOS | Update |")
    assert page.body_text == "Windows; Patch\nmacOS; Update"


def test_paragraphs_keep_one_blank_line() -> None:
    assert clean("one\n\n\n\ntwo\nthree\n\n").body_text == "one\n\ntwo\nthree"


# ── inline noise ─────────────────────────────────────────────────────────────


def test_images_including_linked_svg_data_images_are_removed() -> None:
    svg = "data:image/svg+xml,%3Csvg%20viewBox='0%200%20150%20150'%3E%3C/svg%3E"
    page = clean(
        f"Intro ![Report cover](https://cdn.example.com/a.png) text.\n"
        f"[![install icon]({svg})](https://www.example.com/install/)"
    )
    assert page.body_text == "Intro text."
    assert page.links == ()
    assert removed(page)["image"] == 1
    assert removed(page)["linked_image"] == 1


def test_bare_and_angle_bracket_urls_are_removed() -> None:
    page = clean("See https://www.example.com/x?y=1 or <https://example.org> or www.foo.com today.")
    assert page.body_text == "See or or today."


def test_emphasis_code_html_entities_and_escapes_are_cleaned() -> None:
    page = clean(
        "**Explore**: the *key* __parts__ and _notes_, run `%windir%\\System32\\x.exe`"
        "<br>Set &amp; Forget in HKEY\\_LOCAL\\_MACHINE"
    )
    assert page.body_text == (
        "Explore: the key parts and notes, run %windir%\\System32\\x.exe "
        "Set & Forget in HKEY_LOCAL_MACHINE"
    )


def test_underscores_inside_words_survive() -> None:
    assert clean("the snake_case_name stays").body_text == "the snake_case_name stays"


def test_zero_width_characters_are_removed() -> None:
    assert clean("pat\u200bch\ufeff now").body_text == "patch now"


# ── links ────────────────────────────────────────────────────────────────────


def test_link_keeps_its_anchor_in_the_body_and_is_extracted() -> None:
    page = clean("Read our [patch guide](/guides/patching/#step-2) before you start. Then relax.")
    assert page.body_text == "Read our patch guide before you start. Then relax."
    (link,) = page.links
    assert str(link.target_url) == "https://www.example.com/guides/patching/"
    assert link.anchor_text == "patch guide"
    assert link.surrounding_text == "Read our patch guide before you start."
    assert link.is_internal is True


def test_external_links_are_marked_external() -> None:
    (link,) = clean("See [the NVD entry](https://nvd.nist.gov/vuln/detail/CVE-1).").links
    assert link.is_internal is False


def test_www_and_bare_host_count_as_the_same_site() -> None:
    (link,) = clean("[home](https://example.com/)").links
    assert link.is_internal is True


def test_anchor_markup_is_cleaned_before_extraction() -> None:
    (link,) = clean("[**Bold** anchor](/x/)").links
    assert link.anchor_text == "Bold anchor"


def test_a_url_used_as_its_own_anchor_is_extracted_but_not_kept_in_the_body() -> None:
    page = clean("Source: [https://example.com/r/](https://example.com/r/) end")
    assert page.body_text == "Source: end"
    assert page.links[0].anchor_text == "https://example.com/r/"


@pytest.mark.parametrize("dest", ["mailto:a@example.com", "tel:+123", "javascript:void(0)"])
def test_non_web_links_keep_their_text_but_are_not_extracted(dest: str) -> None:
    page = clean(f"Contact [our team]({dest}) now")
    assert page.body_text == "Contact our team now"
    assert page.links == ()


def test_same_page_fragment_links_are_not_extracted() -> None:
    page = clean("Jump to [the summary](#summary).")
    assert page.body_text == "Jump to the summary."
    assert page.links == ()


def test_link_with_parentheses_and_title_in_destination() -> None:
    (link,) = clean('[term](https://en.wikipedia.org/wiki/Patch_(computing) "Wiki")').links
    assert str(link.target_url) == "https://en.wikipedia.org/wiki/Patch_(computing)"


# ── boilerplate ──────────────────────────────────────────────────────────────


def test_boilerplate_and_breadcrumb_lines_are_removed_with_their_links() -> None:
    banner = "**Patch this CVE on all your endpoints in under 5 minutes.** First 200 free"
    page = clean(
        f"[Example](https://www.example.com/)\n\u203a [Blog](/blog/)\n{banner}\n\nReal [content](/c/).",
        boilerplate=frozenset({"[Example](https://www.example.com/)", banner}),
    )
    assert page.body_text == "Real content."
    assert [link.anchor_text for link in page.links] == ["content"]
    assert removed(page)["boilerplate_line"] == 2
    assert removed(page)["breadcrumb_line"] == 1


def test_find_boilerplate_returns_lines_at_or_above_the_share() -> None:
    template = "Shared footer line that repeats everywhere"
    docs = [f"{template}\nunique content number {i} here" for i in range(9)] + ["only this one"]
    assert find_boilerplate(docs, min_share=0.9) == frozenset({template})
    assert find_boilerplate(docs, min_share=0.95) == frozenset()


def test_line_shares_ignores_short_lines_and_counts_each_document_once() -> None:
    shares = dict(line_shares(["long enough line here!!\nlong enough line here!!\nshort", "x"]))
    assert shares == {"long enough line here!!": 0.5}


@pytest.mark.parametrize("share", [0, -0.1, 1.5])
def test_find_boilerplate_rejects_an_invalid_share(share: float) -> None:
    with pytest.raises(ValueError, match="min_share"):
        find_boilerplate(["a"], min_share=share)


# ── page level ───────────────────────────────────────────────────────────────


def test_cleaning_is_idempotent_on_its_own_output() -> None:
    once = clean("# T\n\n**A** [b](/b/) ![i](/i.png) https://x.com\n\n| c | d |").body_text
    assert clean(once).body_text == once


def test_title_is_stripped_and_blank_title_becomes_none() -> None:
    assert clean("x", title="  A title ").title == "A title"
    assert clean("x", title="   ").title is None


def test_page_url_is_normalised_and_must_be_absolute() -> None:
    assert (
        str(clean_page("x", "HTTPS://WWW.Example.com/a/#frag").url) == "https://www.example.com/a/"
    )
    with pytest.raises(ValueError, match="absolute"):
        clean_page("x", "/relative/")


# ── refinements measured on the real crawl ───────────────────────────────────


def test_link_text_wrapped_across_lines_is_joined_and_extracted() -> None:
    page = clean("Not a trial. [Start\npatching](/signup/) today.")
    assert page.body_text == "Not a trial. Start patching today."
    assert page.links[0].anchor_text == "Start patching"


def test_orphan_markup_is_removed() -> None:
    page = clean("**Learn more about security and compliance\nstray tail]( ) here")
    assert page.body_text == "Learn more about security and compliance\nstray tail here"


def test_a_line_starting_with_an_angle_separator_is_a_breadcrumb() -> None:
    assert clean("\u203a CVE-2026-16861\n\nBody.").body_text == "Body."


def test_glyph_continuation_is_dropped_only_after_a_breadcrumb_or_template_line() -> None:
    root = "[Homepage](https://www.example.com/)"
    page = clean(
        f"{root}\n 5 [Blog](/blog/)\n 5 2024 AI Impact Report\n\n5 steps to patch faster.",
        boilerplate=frozenset({root}),
    )
    assert page.body_text == "5 steps to patch faster."
    assert page.links == ()


def test_navigation_lines_need_only_the_lower_share_but_sentences_need_the_higher() -> None:
    nav = "*   [Alerts](https://www.example.com/documentation/alerts/)"
    sentence = "IBM i 7.3 through 7.6 are affected by a broad set of flaws."
    docs = [f"{nav}\n{sentence}\nunique body {i}" for i in range(5)]
    docs += [f"other page number {i} with its own text" for i in range(95)]
    found = find_boilerplate(docs, min_share=0.2, nav_min_share=0.02)
    assert template_key(nav) in found
    assert template_key(sentence) not in found


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("*   [Alerts](https://x.com/a/)", True),
        ("5 [Blog](https://x.com/blog/)", True),
        ("### By Peter Barnett", True),
        ("- Security Concerns", True),
        ("A sentence with a [link](https://x.com/) inside it.", False),
        ("Plain repeated sentence of content.", False),
    ],
)
def test_is_navigation(line: str, expected: bool) -> None:
    assert is_navigation(line) is expected


# ── outline ──────────────────────────────────────────────────────────────────


def test_headings_outline_covers_setext_and_atx_levels_in_order() -> None:
    page = clean(
        "Report\n======\n\nIntro text.\n\nWhat is inside\n--------------\n\n### Detail\n\nBody."
    )
    assert page.headings == ((1, "Report"), (2, "What is inside"), (3, "Detail"))
    assert page.h1 == "Report"


def test_a_rule_after_a_blank_line_is_not_a_heading() -> None:
    page = clean("Paragraph text.\n\n---\n\nMore text.")
    assert page.headings == ()
    assert page.body_text == "Paragraph text.\n\nMore text."


# ── measured after the first write ──────────────────────────────────────────


def test_glyph_bullets_are_stripped() -> None:
    page = clean(
        "^ Automate patching\n\u2022 Enhanced security\n\u2610 Role count ok\n\u2013 Real-time view"
    )
    assert page.body_text == "Automate patching\nEnhanced security\nRole count ok\nReal-time view"


def test_replacement_characters_are_removed() -> None:
    assert clean("\ufffd 10 hours saved each month").body_text == "10 hours saved each month"


def test_a_punctuation_fragment_is_joined_to_the_line_it_continues() -> None:
    page = clean("See the [2024 report](/r/)\n, identifying a 61% increase.")
    assert page.body_text == "See the 2024 report, identifying a 61% increase."


def test_template_lines_that_differ_only_in_urls_are_detected_and_removed() -> None:
    docs = [f"[Previous Post](/p/{i}/) [Next Post](/n/{i}/)\nbody {i}" for i in range(10)]
    found = find_boilerplate(docs, min_share=0.5)
    assert template_key("[Previous Post](/p/1/) [Next Post](/n/1/)") in found
    page = clean("[Previous Post](/p/7/) [Next Post](/n/7/)\nReal text.", boilerplate=found)
    assert page.body_text == "Real text."
    assert page.links == ()


def test_meta_fields_are_cleaned() -> None:
    assert clean_meta("\u200bTitle &amp; More  ") == "Title & More"
    assert clean_meta("  ") is None
    assert clean_meta(None) is None
    assert clean("x", title="\u200bTitle").title == "Title"


def test_a_glyph_behind_wrapped_emphasis_is_stripped() -> None:
    page = clean("*   **IT Asset\n    ** \u2013 Real-time visibility")
    assert page.body_text.splitlines()[-1] == "Real-time visibility"


def test_punctuation_only_lines_are_dropped() -> None:
    assert clean("Heading text\n;\n.\nBody.").body_text == "Heading text\nBody."


def test_a_leading_dot_word_is_not_a_continuation() -> None:
    assert clean("Patched in the test environment.\n.NET Framework fix").body_text == (
        "Patched in the test environment.\n.NET Framework fix"
    )


# ── template links ───────────────────────────────────────────────────────────

ROOT = "[Home](https://www.example.com/)"
SIDEBAR = ("*   [Alerts](/docs/alerts/)", "*   [Reports](/docs/reports/)")
BANNER = "**Try it free for 30 days on every endpoint.** [Sign up](/signup/)"
RELATED = "[Recent posts](/blog/) [Older posts](/blog/?page=2) [Partner](https://other.test/)"
UNRECORDED = (
    "[![badge](/b.png)](/awards/) [Contact](mailto:team@example.com) "
    "[Top](#top) [](/empty/) ![logo](/logo.png)"
)
TEMPLATE = frozenset({ROOT, *SIDEBAR, BANNER, RELATED, UNRECORDED})
TEMPLATED_PAGE = "\n".join(
    [
        ROOT,
        " 5 [Blog](/blog/)",
        "\u203a [Guides](/blog/guides/) \u203a Patch guide",
        *SIDEBAR,
        "",
        "# Patch guide",
        "",
        "Read our [setup notes](/docs/setup/) first.",
        SIDEBAR[0],
        "More text with [an external link](https://other.test/x).",
        "",
        "Setup",
        "-----",
        "",
        "Final words on [reports](/docs/reports/).",
        "",
        BANNER,
        RELATED,
        UNRECORDED,
        "",
    ]
)


def test_recording_template_links_leaves_body_and_body_hash_as_before() -> None:
    # Expected values are the cleaner's output on this page before template links were recorded.
    page = clean(TEMPLATED_PAGE, boilerplate=TEMPLATE)
    assert page.body_text == (
        "Patch guide\n\nRead our setup notes first.\nMore text with an external link."
        "\n\nSetup\n\nFinal words on reports."
    )
    assert body_hash(page.body_text) == (
        "61716174c0d867f9856079fdfe89b975e0a71189df74ebe02912b319a9e91e59"
    )
    assert (page.h1, page.headings) == ("Patch guide", ((1, "Patch guide"), (2, "Setup")))
    assert [
        (str(link.target_url), link.anchor_text, link.surrounding_text, link.is_internal)
        for link in page.links
    ] == [
        ("https://www.example.com/docs/setup/", "setup notes", "Read our setup notes first.", True),
        ("https://other.test/x", "an external link", "More text with an external link.", False),
        ("https://www.example.com/docs/reports/", "reports", "Final words on reports.", True),
    ]
    assert page.removed == (
        ("boilerplate_line", 7),
        ("breadcrumb_line", 2),
        ("heading_marker", 1),
        ("link_markup", 3),
        ("rule_or_underline", 1),
    )


def test_template_links_are_menu_above_or_within_the_body_and_footer_after_it() -> None:
    page = clean(TEMPLATED_PAGE, boilerplate=TEMPLATE)
    assert [(link.zone, str(link.target_url)) for link in page.template_links] == [
        ("menu", "https://www.example.com/"),
        ("menu", "https://www.example.com/blog/"),
        ("menu", "https://www.example.com/blog/guides/"),
        ("menu", "https://www.example.com/docs/alerts/"),
        ("menu", "https://www.example.com/docs/reports/"),
        ("menu", "https://www.example.com/docs/alerts/"),
        ("footer", "https://www.example.com/signup/"),
        ("footer", "https://www.example.com/blog/"),
        ("footer", "https://www.example.com/blog/?page=2"),
    ]


def test_external_same_page_non_web_empty_and_image_links_are_not_template_links() -> None:
    page = clean(f"Body.\n{UNRECORDED}\n{RELATED}", boilerplate=frozenset({UNRECORDED, RELATED}))
    assert [str(link.target_url) for link in page.template_links] == [
        "https://www.example.com/blog/",
        "https://www.example.com/blog/?page=2",
    ]


def test_an_invalid_template_link_is_skipped_without_counting_as_removed() -> None:
    too_long = "[Search](/search?q=" + "a" * 2100 + ")"  # over HttpUrl's 2083 characters
    page = clean(f"Body.\n{too_long}", boilerplate=frozenset({too_long}))
    assert page.template_links == ()
    assert removed(page) == {"boilerplate_line": 1}


def test_a_breadcrumb_after_the_last_body_line_is_footer() -> None:
    page = clean("Body text.\n\n---\n\n\u203a [Back to the blog](/blog/)")
    assert [(link.zone, str(link.target_url)) for link in page.template_links] == [
        ("footer", "https://www.example.com/blog/")
    ]


def test_without_body_text_every_template_link_is_menu() -> None:
    page = clean(f"{ROOT}\n\n{BANNER}", boilerplate=frozenset({ROOT, BANNER}))
    assert page.body_text == ""
    assert {link.zone for link in page.template_links} == {"menu"}
    assert len(page.template_links) == 2


# ── body hash ────────────────────────────────────────────────────────────────

# Expected digests come from `shasum -a 256` over the UTF-8 bytes, not from hashlib.
MIXED_SCRIPT = "Caf\u00e9 na\u00efve \u65e5\u672c\u8a9e \U0001f680"
MIXED_SCRIPT_SHA256 = "a7d1bfbd648a86abefea7aaebdb9662662231c44e8bdd8780427363ce75552a2"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
COMPOSED_SHA256 = "73473dcc12b763085904a5279d048c4d5b3b008c46f1f32443b99de04aa83a14"
DECOMPOSED_SHA256 = "c42cc7a1ca08364b6fd859fa50d2454730a8236290a423373cc630da77c6d711"


def test_body_hash_is_sha256_of_utf8_for_non_ascii_text() -> None:
    assert body_hash(MIXED_SCRIPT) == MIXED_SCRIPT_SHA256


def test_body_hash_of_empty_text_is_the_empty_sha256() -> None:
    assert body_hash("") == EMPTY_SHA256


def test_body_hash_changes_when_one_character_changes() -> None:
    edited = MIXED_SCRIPT.replace("na\u00efve", "naive")
    assert len(edited) == len(MIXED_SCRIPT)
    assert body_hash(edited) != body_hash(MIXED_SCRIPT)
    assert body_hash("Body text.") != body_hash("Body text!")


def test_body_hash_is_over_exact_code_points_not_a_normal_form() -> None:
    assert body_hash("Caf\u00e9") == COMPOSED_SHA256
    assert body_hash("Cafe\u0301") == DECOMPOSED_SHA256


def test_body_hash_is_lowercase_hex_of_64_chars() -> None:
    digest = body_hash(MIXED_SCRIPT)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef"), digest
