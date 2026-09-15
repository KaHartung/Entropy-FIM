"""
Semantic pre-filter: cheap regex/unicode heuristics to reject non-UML and
non-Latin-script content before the (much slower) real-engine classification.
"""

import re
import unicodedata


# ---------------------- SEMANTIC FILTER (non-UML / sprite removal) ----------------------

# --- Strong non-UML indicators (compiled once) ---
RX_C4_MACROS = re.compile(r"(?i)\bC4_(Context|Container|Component|Deployment)\b")
RX_C4_CALLS  = re.compile(r"(?i)\b(Person|System|Container|Component|Rel)\s*\(")
RX_TYPES     = re.compile(r"(?im)^\s*(mindmap|wbs|gantt|chronology|json|yaml|salt)\b")

# Same non-UML diagram families as RX_TYPES, but anchored to the actual @start<type>
# tag rather than a bare word -- more precise (no risk of matching the word in prose),
# and catches cases the bare-word check misses (confirmed: a @startditaa ASCII-art box
# diagram has no bare "ditaa"/etc. text, but its box-drawing characters happen to
# contain literal "-->", which has_uml_content's arrow signal picks up by coincidence).
# PlantUML's own SourceStringReader.getBlocks() doesn't even recognize @startditaa as a
# block, so without this it falls through to the jar's num_blocks==0 wrap-and-retry
# path, which then wrongly nests it inside a synthetic @startuml and reports 'invalid'.
# `dot` included for the same reason: @startdot/digraph content parses fine via the
# jar, but as a raw Java class name (PSystemDot) rather than a core.DiagramType enum
# value, so extract_insights_jar's _NON_UML_JAR_TYPES (built from enum names) can't
# recognize it either -- catching it here, before the jar ever runs, sidesteps that.
RX_NON_UML_START_TAG = re.compile(r"(?im)^\s*@start(mindmap|wbs|gantt|chronology|json|yaml|salt|ditaa|dot)\b")

# C4 theme/config fingerprints
RX_C4_THEMEVAR = re.compile(r'(?im)^\s*!\s*\$THEME\s*=\s*"\s*C4_[^"]+"\s*$')
RX_C4_THEMESTR = re.compile(r'(?i)\bC4_[A-Za-z0-9_]+\b')
RX_C4_VARS     = re.compile(r'(?im)^\s*!\s*\$(PERSON|SYSTEM|CONTAINER|COMPONENT|BOUNDARY|ARROW)_')

# C4-PlantUML's macros, once expanded (or hand-authored in the same style), leave
# plain rectangle/folder/frame elements behind instead of the macro calls themselves --
# but they keep two fingerprints: bracketed stereotype labels like "[Person]"/
# "[Software System]" (C4-PlantUML's literal label template), and numbered guillemet
# stereotypes paired with a matching skinparam block (C4-PlantUML's per-element
# styling idiom, e.g. `skinparam rectangle<<26>> { ... }` alongside a `<<26>>` element).
RX_C4_BRACKET_STEREOTYPE = re.compile(
    r"\[(External\s+)?(Person|Software\s+System|System|Container|Component|Database)\]", re.IGNORECASE
)
RX_C4_NUMBERED_SKINPARAM = re.compile(
    r"(?im)^\s*skinparam\s+(rectangle|folder|frame|node|database|cloud)<<\d+>>\s*\{"
)

# !includeurl is PlantUML's directive specifically for including from a remote URL
# (unlike !include, which can also point to a local file). In every example seen so
# far, it's exclusively used to pull in an icon/macro library (Azure, AWS, GOV icons,
# etc.) and then call macro-style functions to build boxes -- not organic hand-authored
# UML content. Real declarations/relationships can still coexist in such a file (e.g.
# `-->` arrows connecting the macro-built boxes), which is exactly why has_uml_content's
# arrow signal alone isn't enough to catch this pattern without this fingerprint.
RX_INCLUDEURL = re.compile(r"(?im)^\s*!\s*includeurl\b")

# A literal generator-tool attribution: e.g. helm-chartmap's PlantUmlChartMapPrinter
# stamps its fully-qualified class name into a `footer` line of every diagram it
# generates. These are auto-generated "spec sheet" boxes (artifact/usecase used purely
# as generic labeled containers for dependency-graph data, e.g. Helm chart/Docker image
# metadata) rather than organic UML modeling, but they still contain real relationship
# arrows, so has_uml_content's arrow signal alone can't distinguish them. Narrow and
# low-risk by design: exact tool name match, not a general "looks auto-generated"
# heuristic -- extend with more tool names here if further examples turn up.
RX_GENERATOR_ATTRIBUTION = re.compile(r"PlantUmlChartMapPrinter")

# UML content signal (keep list conservative). Only feeds has_uml_content, which in
# turn is now only consulted for the narrow "contains a C4_* identifier but nothing
# else recognizable" fallback below -- has_uml_content used to also be is_non_uml's
# general catch-all ("no recognizable UML content at all -> non-uml"), but that was
# retired: this hand-maintained keyword/arrow list is inherently incomplete and was
# confirmed false-positiving on real content (implicit-lifeline sequence diagrams
# using only bare `->`, and diagrams using only the `component` keyword or `[Name]`
# bracket shorthand -- neither is in this list at all). extract_insights_jar now
# handles that "is there really nothing here" question post-jar instead, grounded in
# what the engine actually parsed rather than a pre-jar guess.
RX_UML_SIGNAL = re.compile(
    r"(?im)^\s*(class|interface|enum|abstract|package|object|state|usecase|actor|participant)\b"
    r"|--\|>|..\|>|\*--|o--|<\|--|-->|<--"
    r"|^\s*(start|stop)\s*$"
    r"|\bif\s*\(.+\)\s*then\b"
    r"|\brepeat\b|\bendwhile\b"
    r"|^\s*:[^:\n]+;\s*$"
)

# Remove preprocessing / styling lines + comment-only lines to check "real content"
RX_PREPROC_LINE = re.compile(r"(?m)^\s*(!|skinparam)\b.*$")
RX_COMMENT_LINE = re.compile(r"(?m)^\s*('.*|//.*)$")


def has_uml_content(s: str) -> bool:
    if not s:
        return False
    # strip comments + preprocessing boilerplate
    core = RX_COMMENT_LINE.sub("", s)
    core = RX_PREPROC_LINE.sub("", core)
    core = core.strip()
    if not core:
        return False
    return bool(RX_UML_SIGNAL.search(core))


def is_non_uml(s: str) -> bool:
    if not s:
        return False

    # strong non-UML indicators
    if RX_C4_MACROS.search(s) or RX_C4_CALLS.search(s) or RX_TYPES.search(s) or RX_NON_UML_START_TAG.search(s):
        return True

    # icon-library usage (see RX_INCLUDEURL) -- checked before has_uml_content's arrow
    # signal below, since those same arrows routinely connect the macro-built boxes
    if RX_INCLUDEURL.search(s):
        return True

    # known auto-generated "spec sheet diagram" tool attribution (see RX_GENERATOR_ATTRIBUTION)
    if RX_GENERATOR_ATTRIBUTION.search(s):
        return True

    # C4 theme/config-only indicators (your snippet triggers these)
    if RX_C4_THEMEVAR.search(s) or RX_C4_VARS.search(s):
        return True

    # C4-PlantUML's expanded/hand-authored form: no macro calls, but the same
    # bracketed stereotypes and numbered-skinparam styling idiom give it away
    if RX_C4_BRACKET_STEREOTYPE.search(s) or RX_C4_NUMBERED_SKINPARAM.search(s):
        return True

    # fallback: if it contains C4_* and has no UML content, treat as non-UML
    if RX_C4_THEMESTR.search(s) and not has_uml_content(s):
        return True

    return False


def contains_non_ascii(s: str) -> bool:
    """True if s contains at least one non-ASCII character."""
    if not s:
        return False
    return any(ord(ch) > 127 for ch in s)


def contains_non_latin_script(s: str) -> bool:
    """
    True if s contains any LETTER character that is not Latin.
    - Latin letters (including accented) => not flagged
    - Digits/punctuation/whitespace => ignored
    - Greek/Cyrillic/Han/Arabic/etc letters => flagged
    """
    if not s:
        return False

    for ch in s:
        # Only consider letters; ignore digits/punctuation/whitespace/symbols
        if unicodedata.category(ch).startswith("L"):
            name = unicodedata.name(ch, "")
            # Latin letters contain 'LATIN' in the Unicode name
            if "LATIN" not in name:
                return True
    return False


def passes_semantic_filter(code: str) -> tuple[bool, str | None]:
    """
    Apply the semantic filter (non-UML/C4/sprite removal + Latin-only content).

    Returns:
        tuple: (passes, reason) where reason is None if passes is True
    """
    if is_non_uml(code):
        return False, "non_uml"
    if contains_non_latin_script(code):
        return False, "non_latin_script"
    if contains_non_ascii(code):
        return False, "non_ascii"
    return True, None
