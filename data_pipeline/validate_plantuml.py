#!/usr/bin/env python3
"""
PlantUML diagram-type classifier and repair helpers.

Classifies PlantUML source using extract_insights_jar (real PlantUML engine ground
truth, with a regex fallback only where the engine can't subdivide further -- see
that function's docstring for the full resolution order), and provides the shared
cleaning steps (dedup, footer/tag repair, multi-block splitting) every data-pipeline
script composes.
"""

import atexit
import os
import re
import select
import subprocess
import threading
from pathlib import Path

from datasets import disable_caching

from filter_data import RX_COMMENT_LINE, is_non_uml

# datasets caches .map()/.filter() results on disk keyed by a fingerprint of the
# mapped function -- but that fingerprint is only ever derived from the top-level
# mapped function's own bytecode, not from anything it calls. A row already
# classified by an older run of this module can get served that stale cached result
# instead of being reclassified, completely silently, if this module's own
# classification logic changes but the top-level mapped function's bytecode doesn't.
# Disabling caching here, in the module nearly every other script in this pipeline
# imports from, protects all of them from this failure mode. Process-local and
# idempotent -- harmless if a caller also disables it itself, and doesn't touch any
# on-disk cache files, so it's safe even while other dataset scripts are running.
disable_caching()


# Structural signals per UML sub-type. Used both as the primary classifier in
# extract_insights, and as a fallback in extract_insights_jar for diagrams the
# real PlantUML engine reports as DESCRIPTION (its catch-all for use_case/
# component/deployment diagrams, which it doesn't distinguish internally).
#
# 'deployment' is deliberately not in here -- see _DEPLOYMENT_FALLBACK_PATTERN below.
# Bare "actor" is deliberately not in 'use_case' either -- see _ACTOR_KEYWORD_PATTERN.
#
# Every bare-keyword declaration (class/state/object/component/usecase/actor/deployment)
# is anchored to line start (^\s*), matching the same discipline the num_X
# element-counting regexes below already use. Real PlantUML declarations are always
# the first token on their line -- a bare \b check without that anchor matches the
# keyword as a substring of an unrelated identifier too (e.g. `SceneManagement::Component`
# at the end of a relationship line, or `MainActivity`/`Deployment Team` elsewhere),
# since \s+ is satisfied by a trailing newline and \b only requires a non-word char
# before it.
#
# They're also followed by (?=[\w"]): a real declaration is followed by an identifier
# or a quoted name (`object Foo`, `class "My Class" as X`), never by an arrow. Without
# this, a line like `object -> OIDC` -- where "object" is just an identifier used as a
# relationship endpoint (declared earlier via some other means), not the object-diagram
# keyword -- would still match, since it's the first token on its line too.
_TYPE_PATTERNS = {
    # interface/enum are class-diagram declarations too (num_classes below already
    # counts them alongside class/abstract class; this was missing that parity)
    'class': r'^\s*(abstract\s+class|class|interface|enum)\s+(?=[\w"])',
    # usecase keyword, `:Name:` actor shorthand, `(text) as alias` ellipse shorthand,
    # and a bare `(text)` ellipse used directly as a relationship endpoint (e.g.
    # `user1 --> (Usecase 1)`, no "as alias" at all -- arguably the most common form)
    # are all use-case-diagram-exclusive syntax. The last one requires the paren to be
    # immediately adjacent to an arrow (only whitespace between) to avoid matching
    # class-diagram method signatures (`Foo(a, b)`) or activity conditions
    # (`if (x) then (y)`), which also use non-empty parens but never right next to one.
    'use_case': (
        r'^\s*usecase\s+(?=[\w"])'
        r'|^\s*:[^:\n]+:\s*(as\s+\w+)?\s*$'
        r'|\)\s+as\s+\(?\w+\)?'
        r'|(-+>|\.+>|<-+|<\.+)\s*\([^()\n]+\)'
        r'|\([^()\n]+\)\s*(-+>|\.+>|<-+|<\.+)'
    ),
    # participant is the explicit form; activate/deactivate/autonumber/hide footbox
    # are sequence-only directives that catch diagrams using implicit lifelines
    'sequence': r'\bparticipant\s+|\bactivate\s+\w+|\bdeactivate\b|\bautonumber\b|\bhide\s+footbox\b',
    'state': r'^\s*state\s+(?=[\w"])',
    'object': r'^\s*object\s+(?=[\w"])',
    # the explicit `component Name` keyword, or the much more common `[Name]` bracket
    # shorthand -- but not `[[url]]` hyperlink syntax (doubled brackets) or `[#color]`
    # (PlantUML's arrow/element color-tint syntax, e.g. `A --[#green]-|> B`)
    'component': r'^\s*component\s+(?=[\w"])|(?<!\[)\[(?!\[)(?!#)[^\[\]\n]+\](?!\])',
    # the bare word "activity" is unreliable (matches identifiers/notes that just
    # mention it); real activity diagrams use these structural constructs instead
    'activity': r'^\s*start\s*$|^\s*stop\s*$|\bif\s*\(.+\)\s*then\b|\brepeat\b|\bendwhile\b|^\s*:[^:\n]+;\s*$',
}

# A bare "deployment" match is weak on its own -- it's matched team names, step
# labels, and placeholder titles in practice, not real deployment-diagram syntax.
# `node` is genuine deployment-diagram vocabulary though, and since this whole pattern
# is only consulted as a last resort (once component/use_case/etc. have all had first
# shot -- see _classify_by_patterns), the earlier concern about it colliding with
# generic grouping shapes in component diagrams doesn't apply here: if a diagram had
# real component signals too, those would already have matched and this fallback would
# never even run. `package` is deliberately NOT included -- it's used as a generic
# grouping/namespace construct across nearly every diagram type (including plain class
# diagrams), not deployment-specific like `node` is, and can even appear as an ordinary
# element name (confirmed: a real example uses `agent package` as an agent's name).
# Anchored to line start (plus the identifier lookahead) for the same reasons as the
# keywords above.
_DEPLOYMENT_FALLBACK_PATTERN = r'^\s*(deployment|node)\s+(?=[\w"])'

# A bare "actor" is the weakest, most ambiguous signal we have: it's the use-case
# keyword, but it's also routinely used to declare an external participant in
# otherwise ordinary sequence diagrams (`actor User` then `User -> System: ...`) or
# component/architecture diagrams (`actor "App User"` alongside `component ...`).
# So it's only treated as a use_case signal when nothing else matched at all --
# otherwise it's just an external participant, not evidence this is a use-case diagram.
_ACTOR_KEYWORD_PATTERN = r'^\s*actor\s+(?=[\w"])'

# Same as _TYPE_PATTERNS, but with 'class' narrowed to exclude a bare `interface`
# declaration -- used only for _classify_by_patterns's "which type(s) does this match"
# check (which both drives regex-only classification and decides 'mixed' when >1 type
# matches). interface is just as legitimately a component diagram's ball-and-socket
# provided/required interface declaration as it is a class-diagram interface, and real
# component diagrams routinely declare one alongside their [Name]/component elements
# -- confirmed real examples where treating bare interface as unconditional class-
# diagram evidence made an otherwise-clean component diagram match both 'class' and
# 'component' and get wrongly classified 'mixed'.
#
# _TYPE_PATTERNS['class'] itself is deliberately left untouched (still includes
# interface): num_classes counting and the CLASS-vs-OBJECT disambiguation override in
# extract_insights_jar both still need interface treated as real class-family content
# there -- only the mixed-detection/classification role is narrowed.
_TYPE_PATTERNS_FOR_CLASSIFICATION = {
    **_TYPE_PATTERNS,
    'class': r'^\s*(abstract\s+class|class|enum)\s+(?=[\w"])',
}


def _match_type_patterns(plantuml_code: str) -> list[str]:
    return [
        name for name, pattern in _TYPE_PATTERNS_FOR_CLASSIFICATION.items()
        if re.search(pattern, plantuml_code, re.IGNORECASE | re.MULTILINE)
    ]


def _classify_by_patterns(plantuml_code: str) -> str | None:
    """
    Classify plantuml_code using _TYPE_PATTERNS (plus the actor/deployment
    special cases above). Returns None (caller should use 'unknown') if nothing
    matches at all, including the deployment fallback.
    """
    matched_types = set(_match_type_patterns(plantuml_code))

    if not matched_types and re.search(_ACTOR_KEYWORD_PATTERN, plantuml_code,
                                        re.IGNORECASE | re.MULTILINE):
        matched_types.add('use_case')

    if len(matched_types) > 1:
        return 'mixed'
    if len(matched_types) == 1:
        return next(iter(matched_types))
    if re.search(_DEPLOYMENT_FALLBACK_PATTERN, plantuml_code, re.IGNORECASE | re.MULTILINE):
        return 'deployment'
    return None


# interface/enum declarations share the same "keyword, then identifier-or-quoted-name"
# shape as class (see _TYPE_PATTERNS['class']'s docstring above), so they get the same
# (?=[\w"]) lookahead treatment instead of a bare \w+ -- otherwise a quoted declaration
# like `interface "My Interface" as X` would be classified correctly (via
# _TYPE_PATTERNS['class'], which already covers interface/enum) but silently
# undercounted here.
_RX_NUM_INTERFACES = r'^\s*interface\s+(?=[\w"])'
_RX_NUM_ENUMS = r'^\s*enum\s+(?=[\w"])'


def _count_elements(plantuml_code: str, diagram_type: str) -> dict:
    """
    Count structural elements, gated to the diagram type each element actually
    belongs to -- e.g. num_classes is only counted when diagram_type == 'class', so a
    sequence diagram that happens to mention the word "class" in a note or comment
    doesn't pollute class-diagram statistics. Uses the exact same patterns as
    _TYPE_PATTERNS/_ACTOR_KEYWORD_PATTERN wherever one already exists, instead of a
    separately maintained count-only regex -- previously these had drifted out of
    sync (missing the component `[Name]` bracket shorthand, the use_case colon/paren
    shorthand, and the class/object quoted-name lookahead), silently undercounting
    exactly the diagrams whose type depended on that shorthand syntax.

    Relationships/notes/packages/namespaces/title/legend/skinparam are cross-cutting
    features that don't belong to one specific diagram type, so those stay
    unconditional.
    """
    counts = {
        'num_classes': 0,
        'num_interfaces': 0,
        'num_enums': 0,
        'num_components': 0,
        'num_actors': 0,
        'num_usecases': 0,
        'num_relationships': 0,
        'num_notes': 0,
        'num_packages': 0,
        'num_namespaces': 0,
        'num_objects': 0,
        'has_title': False,
        'has_legend': False,
        'has_skinparam': False,
    }

    if diagram_type == 'class':
        counts['num_classes'] = len(re.findall(_TYPE_PATTERNS['class'], plantuml_code, re.MULTILINE | re.IGNORECASE))
        counts['num_interfaces'] = len(re.findall(_RX_NUM_INTERFACES, plantuml_code, re.MULTILINE | re.IGNORECASE))
        counts['num_enums'] = len(re.findall(_RX_NUM_ENUMS, plantuml_code, re.MULTILINE | re.IGNORECASE))

    if diagram_type == 'component':
        counts['num_components'] = len(re.findall(_TYPE_PATTERNS['component'], plantuml_code, re.MULTILINE | re.IGNORECASE))

    if diagram_type == 'use_case':
        counts['num_actors'] = len(re.findall(_ACTOR_KEYWORD_PATTERN, plantuml_code, re.MULTILINE | re.IGNORECASE))
        counts['num_usecases'] = len(re.findall(_TYPE_PATTERNS['use_case'], plantuml_code, re.MULTILINE | re.IGNORECASE))

    if diagram_type == 'object':
        counts['num_objects'] = len(re.findall(_TYPE_PATTERNS['object'], plantuml_code, re.MULTILINE | re.IGNORECASE))

    # Count relationships (arrows) -- every diagram type can have these
    relationship_patterns = [
        r'--', r'<--', r'-->','<->', r'\*--', r'--\*', r'o--', r'--o',
        r'<\|--', r'--\|>', r'<\|x', r'x\|>', r'\[\[', r'\]\]'
    ]
    for pattern in relationship_patterns:
        counts['num_relationships'] += len(re.findall(pattern, plantuml_code))

    counts['num_packages'] = len(re.findall(r'^\s*package\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))
    counts['num_namespaces'] = len(re.findall(r'^\s*namespace\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))
    counts['num_notes'] = len(re.findall(r'^\s*note\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))
    counts['has_title'] = bool(re.search(r'^\s*title\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))
    counts['has_legend'] = bool(re.search(r'^\s*legend\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))
    counts['has_skinparam'] = bool(re.search(r'^\s*skinparam\s+', plantuml_code, re.MULTILINE | re.IGNORECASE))

    return counts


def extract_insights(plantuml_code: str) -> dict:
    """Extract insights from valid PlantUML code, classifying via regex only."""
    diagram_type = 'unknown'
    if is_non_uml(plantuml_code):
        diagram_type = 'non-uml'
    else:
        classified = _classify_by_patterns(plantuml_code)
        if classified is not None:
            diagram_type = classified

    insights = {'diagram_type': diagram_type}
    insights.update(_count_elements(plantuml_code, diagram_type))
    return insights


JAVA_SRC_DIR = Path(__file__).resolve().parent / "java"
DIAGRAM_TYPE_DETECTOR_CLASS = "DiagramTypeDetector"

# UmlDiagramType values that map 1:1 onto our categories. PlantUML has no separate
# USECASE/COMPONENT/DEPLOYMENT values -- all three come back as DESCRIPTION, which
# extract_insights_jar falls back to regex to further split.
_JAR_TYPE_MAP = {
    'SEQUENCE': 'sequence',
    'CLASS': 'class',
    'STATE': 'state',
    'OBJECT': 'object',
    'ACTIVITY': 'activity',
    'TIMING': 'timing',
    # Composite Structure Diagram is real, official UML -- unlike BPM below. Note this
    # value is currently vestigial in the jar itself: nothing in PlantUML's own
    # sources constructs a diagram of this type (confirmed no other class references
    # DiagramType.COMPOSITE), so this mapping is correct but presently unreachable --
    # kept for forward compatibility in case a future PlantUML version wires it up.
    'COMPOSITE': 'composite',
}

# core.DiagramType values (TitledDiagram.getDiagramType(), PlantUML 1.2026+) for
# diagram families that are not UML at all -- alternate syntaxes PlantUML also renders
# (mindmap, gantt charts, entity-relation diagrams, JSON/YAML pretty-printers, business
# process diagrams, ...). is_non_uml's @start<type> regex (RX_NON_UML_START_TAG/
# RX_TYPES in filter_data.py) only enumerates a subset of these (mindmap/wbs/gantt/
# chronology/json/yaml/salt/ditaa); this is used as a post-jar safety net in
# extract_insights_jar to catch the rest by real engine ground truth instead of having
# to keep extending that regex for every obscure tag PlantUML supports (confirmed
# real, regex-missed examples in the dataset: @startboard, @startwire).
#
# Deliberately excludes UNKNOWN: that's PlantUML's own "couldn't determine" bucket,
# semantically the same as our 'unknown' (which is where it already falls through to
# below) -- not evidence either way of non-UML content.
_NON_UML_JAR_TYPES = {
    'DITAA', 'DOT', 'JCCKIT', 'SALT', 'FLOW', 'CREOLE', 'MATH', 'LATEX', 'DEFINITION',
    'GANTT', 'CHRONOLOGY', 'NWDIAG', 'MINDMAP', 'WBS', 'WIRE', 'JSON', 'GIT', 'BOARD',
    'YAML', 'HCL', 'EBNF', 'REGEX', 'FILES', 'CHEN_EER', 'CHART', 'PACKET', 'SPRITES',
    'BPM',
    'CRASH', 'HELP',
}

# Diagram types that represent an actual classified UML diagram, as opposed to the
# process-level outcomes ('non-uml', 'invalid', 'unknown') extract_insights_jar can
# also produce. Includes 'mixed': as of the resolution logic above, that label is
# reserved for a real, single, jar-valid diagram that legitimately combines more than
# one UML sub-type (allow_mixing, or a jar-confirmed multi-pattern match) -- what
# would otherwise look structurally "mixed" (more than one @start.../@end... block, or
# non-trivial content left dangling outside the recognized block) resolves to
# 'invalid' instead, precisely so it's excluded from here. Element-count insights are
# only meaningful averaged over these; also used to scope the allow_mixing check below
# to diagrams already confirmed real.
REAL_DIAGRAM_TYPES = {'sequence', 'class', 'state', 'object', 'activity', 'timing', 'composite',
                       'use_case', 'component', 'deployment', 'mixed'}

# PlantUML's own `allowmixing`/`allow_mixing` command (real syntax confirmed via the
# jar's CommandAllowMixing: matches the literal regex fragments "allow", "_?", "mixing"
# concatenated -- i.e. "allowmixing" or "allow_mixing", underscore optional). It's a
# class-diagram-only pragma that tells the engine to intentionally allow element types
# that would otherwise conflict to coexist in one diagram -- the author's own explicit
# signal that this diagram deliberately mixes content, which extract_insights_jar
# treats the same as our 'mixed' bucket (see the check right after type resolution).
RX_ALLOW_MIXING = re.compile(r'(?im)^\s*allow_?mixing\s*$')


def ensure_diagram_type_detector_compiled(jar_path: str, java_home: str) -> Path:
    """
    Compile DiagramTypeDetector.java against jar_path if missing, stale, or last
    compiled against a different jar.

    The jar-identity check matters because plantuml.jar's internal API is not stable
    across versions -- confirmed PlantUML 1.2026 removed skin.UmlDiagramType (and
    TitledDiagram.getUmlDiagramType()) entirely, folding its values into
    core.DiagramType instead. A stale .class file compiled against a different-major
    jar version doesn't necessarily fail to load; if the jar still happens to have a
    class of the same name, a now-incompatible method call inside classify() throws at
    runtime instead, which is caught there as a per-diagram error (blocks=0) --
    silently degrading every row to the regex fallback instead of raising anything a
    caller would notice. Recompiling whenever jar_path changes turns that into a loud,
    immediate compile failure instead.

    Call this once from a driver script's main process before fanning out with
    num_proc workers, to avoid concurrent javac invocations racing on the same
    output file.
    """
    src = JAVA_SRC_DIR / f"{DIAGRAM_TYPE_DETECTOR_CLASS}.java"
    compiled = JAVA_SRC_DIR / f"{DIAGRAM_TYPE_DETECTOR_CLASS}.class"
    jar_marker = JAVA_SRC_DIR / f"{DIAGRAM_TYPE_DETECTOR_CLASS}.compiled_against_jar"

    jar_path_abs = str(Path(jar_path).resolve())
    stale = (
        not compiled.is_file()
        or compiled.stat().st_mtime < src.stat().st_mtime
        or not jar_marker.is_file()
        or jar_marker.read_text().strip() != jar_path_abs
    )

    if stale:
        javac_bin = os.path.join(java_home, "bin", "javac")
        if not os.path.isfile(javac_bin):
            raise FileNotFoundError(f"javac not found at {javac_bin}")
        result = subprocess.run(
            [javac_bin, "-cp", jar_path, "-d", str(JAVA_SRC_DIR), str(src)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile {src}: {result.stderr}")
        jar_marker.write_text(jar_path_abs)

    return JAVA_SRC_DIR


_RESULT_RE = re.compile(
    r"<valid>(true|false)</valid><type>(\w+)</type><blocks>(\d+)</blocks>", re.IGNORECASE
)


class _PersistentDiagramTypeDetector:
    """
    Keeps one long-lived DiagramTypeDetector JVM alive and streams diagrams to it
    one at a time (length-prefixed, so embedded newlines are never ambiguous with
    message framing), instead of paying a full JVM boot per diagram.
    """

    def __init__(self, jar_path: str, java_home: str):
        java_bin = os.path.join(java_home, "bin", "java")
        if not os.path.isfile(java_bin):
            raise FileNotFoundError(f"java executable not found at {java_bin}")
        if not os.path.isfile(jar_path):
            raise FileNotFoundError(f"PlantUML jar not found: {jar_path}")

        classes_dir = ensure_diagram_type_detector_compiled(jar_path, java_home=java_home)
        # Scraped PlantUML source can contain !include/!includeurl pointing at
        # arbitrary external URLs; without this, PlantUML will actually issue those
        # HTTP requests while parsing (confirmed: a dead/moved GitHub raw-content URL
        # triggered a fetch, a 404, and the worker dying). SANDBOX disables all
        # network/local-file access during parsing, which we want unconditionally
        # here regardless of the exact crash mechanism -- untrusted scraped content
        # should never be able to make this pipeline issue outbound requests.
        env = {**os.environ, "PLANTUML_SECURITY_PROFILE": "SANDBOX"}
        self.process = subprocess.Popen(
            [java_bin, "-Djava.awt.headless=true", "-cp", f"{jar_path}{os.pathsep}{classes_dir}",
             DIAGRAM_TYPE_DETECTOR_CLASS, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        atexit.register(self.close)

    def query(self, code: str, timeout: int = 30) -> tuple[int, str, bool]:
        if self.process.poll() is not None:
            raise RuntimeError("DiagramTypeDetector server process is no longer running")

        payload = code.encode("utf-8")
        self.process.stdin.write(f"{len(payload)}\n".encode("ascii"))
        self.process.stdin.write(payload)
        self.process.stdin.flush()

        # Only checks that *some* output is ready, not a full line -- fine here since
        # each response is one short line the server writes+flushes atomically.
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close(force=True)
            return 0, "TIMEOUT", False

        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read().decode(errors="replace")
            self.close(force=True)
            raise RuntimeError(f"DiagramTypeDetector server closed unexpectedly: {stderr}")

        match = _RESULT_RE.search(line.decode("utf-8", errors="replace"))
        if not match:
            return 0, "NONE", False

        is_valid = match.group(1).lower() == "true"
        diagram_type_name = match.group(2)
        num_blocks = int(match.group(3))
        return num_blocks, diagram_type_name, is_valid

    def close(self, force: bool = False) -> None:
        if self.process.poll() is not None:
            return
        try:
            if force:
                self.process.kill()
            else:
                self.process.stdin.close()
                self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


_PERSISTENT_DETECTOR_LOCK = threading.Lock()
_PERSISTENT_DETECTORS: dict[tuple[str, str], _PersistentDiagramTypeDetector] = {}


def _get_persistent_detector(jar_path: str, java_home: str) -> _PersistentDiagramTypeDetector:
    """
    Process-local detector cache. Each dataset.map(num_proc=N) worker is a separate
    OS process, so this naturally gives each worker its own persistent JVM without
    any cross-process sharing/locking concerns; the lock only guards this one process.
    """
    key = (jar_path, java_home)
    with _PERSISTENT_DETECTOR_LOCK:
        detector = _PERSISTENT_DETECTORS.get(key)
        if detector is None or detector.process.poll() is not None:
            detector = _PersistentDiagramTypeDetector(jar_path, java_home)
            _PERSISTENT_DETECTORS[key] = detector
    return detector


def get_diagram_type_jar(code: str, jar_path: str, java_home: str, timeout: int = 30) -> tuple[int, str, bool]:
    """
    Ask the real PlantUML engine for a diagram's UmlDiagramType, via a persistent
    DiagramTypeDetector JVM (one per (jar_path, java_home) per process, reused
    across calls) instead of spawning a fresh JVM per call.

    Args:
        code: PlantUML source to parse
        jar_path: path to a plantuml.jar file
        java_home: path to a JDK installation; its bin/java (and bin/javac, if a
            (re)compile is needed) is used
        timeout: seconds to wait for a response before giving up (kills and restarts
            the persistent worker, since its output stream is now out of sync)

    Returns:
        tuple: (num_blocks, diagram_type_name, is_valid). diagram_type_name is an
        UmlDiagramType enum name (e.g. "SEQUENCE", "DESCRIPTION") for UML diagrams,
        the Diagram implementation's class name for non-UML diagram families PlantUML
        also renders (mindmap, gantt, ...), or "NONE"/"TIMEOUT"/"SERVER_ERROR" on failure.
    """
    detector = _get_persistent_detector(jar_path, java_home)
    try:
        return detector.query(code, timeout=timeout)
    except RuntimeError:
        # The persistent worker died mid-query. A single row's failure must never
        # take down an entire multi-hour dataset.map() run -- retry once against a
        # freshly spawned worker (query() already killed the dead one, so
        # _get_persistent_detector will detect that and start a new one) before
        # giving up on just this one row.
        detector = _get_persistent_detector(jar_path, java_home)
        try:
            return detector.query(code, timeout=timeout)
        except RuntimeError:
            return 0, "SERVER_ERROR", False


# @end (with no type suffix) is valid PlantUML syntax -- it closes whichever
# @start... block is currently open, not just a literal "@enduml" -- so the end
# side needs \w* (optional suffix), unlike @start which always needs a real type.
# Both @startuml and @enduml also optionally take a trailing diagram name on the
# same line (`@startuml name ... @enduml name`); [^\n]* consumes that so it isn't
# left dangling as phantom "outside" content.
_BLOCK_SPAN_RE = re.compile(r'@start\w+.*?@end\w*[^\n]*', re.IGNORECASE | re.DOTALL)
# Used to recover from PSystemErrorPreprocessor failures (see extract_insights_jar):
# strips !include/!includeurl/etc. directive lines so the rest of the diagram can be
# retried without whatever unresolvable include caused the preprocessing failure.
RX_INCLUDE_LINE = re.compile(r'(?im)^\s*!\s*include\w*\b.*$')
# Used to recover from PSystemWelcome results (see extract_insights_jar): strips just
# the !definelong(...)/!enddefinelong delimiter lines, keeping the macro body, so an
# unused (never-invoked) macro definition becomes directly-executable diagram content.
RX_DEFINELONG_LINE = re.compile(r'(?im)^\s*!\s*(definelong\b.*|enddefinelong)\s*$')
# Markdown code-fence lines (```, ```plantuml, ```puml, ...) are a common, benign
# wrapper for samples scraped from README/docs files -- not evidence of a second
# diagram or dangling content, so they're tolerated the same as comment lines.
RX_MD_FENCE_LINE = re.compile(r'(?m)^\s*```\w*\s*$')
# Some scraped files use a leading backtick as an ad hoc "comment" marker for
# license/copyright banners (not real PlantUML syntax -- that's a leading apostrophe,
# already handled by RX_COMMENT_LINE) but it's the same kind of boilerplate.
RX_BACKTICK_COMMENT_LINE = re.compile(r'(?m)^\s*`.*$')
# PlantUML's real multi-line block comment: /' ... '/ (like /* ... */ but with ').
# Unlike RX_COMMENT_LINE, the lines inside don't need their own leading marker.
RX_BLOCK_COMMENT = re.compile(r"/'.*?'/", re.DOTALL)
# C/Java-style /* ... */ (often Javadoc) isn't real PlantUML syntax either -- like the
# backtick banners, it's boilerplate left over from wherever the diagram was extracted
# from (e.g. a .java file), not evidence of real content or a second diagram.
RX_C_STYLE_BLOCK_COMMENT = re.compile(r'/\*.*?\*/', re.DOTALL)
# A Markdown heading (# Title, ## Subtitle, ...) is another common README artifact --
# PlantUML has no standalone-line syntax starting with #, so this is unambiguous.
RX_MD_HEADING_LINE = re.compile(r'(?m)^#{1,6}\s+.*$')


def _has_content_outside_blocks(plantuml_code: str) -> bool:
    """
    True if there's non-trivial text (not just blank lines/comments/markdown code
    fences/BOM) outside any @start.../@end... block -- e.g. content left dangling
    after a stray @enduml, which the real PlantUML engine silently ignores rather
    than erroring on, but which our regex fallback would otherwise still see and
    misclassify from.
    """
    outside = _BLOCK_SPAN_RE.sub('', plantuml_code)
    outside = RX_BLOCK_COMMENT.sub('', outside)
    outside = RX_C_STYLE_BLOCK_COMMENT.sub('', outside)
    outside = RX_COMMENT_LINE.sub('', outside)
    outside = RX_MD_FENCE_LINE.sub('', outside)
    outside = RX_MD_HEADING_LINE.sub('', outside)
    outside = RX_BACKTICK_COMMENT_LINE.sub('', outside)
    # A leading UTF-8 BOM is common in files saved by Windows editors -- it's not
    # whitespace per str.isspace()/strip(), so it would otherwise survive as
    # phantom "outside content" on an otherwise-single-block file.
    outside = outside.replace('﻿', '')
    return bool(outside.strip())


def split_into_blocks(code: str) -> list[str]:
    """
    Split code into its individual @start.../@end... block texts, using the same
    span regex extract_insights_jar's 'mixed' detection relies on (_BLOCK_SPAN_RE).

    Returns [code] unchanged -- as a single-element list, not the matched span -- if
    there's zero or one block: a zero-block file (no @start.../@end... tags at all)
    is a different situation, already handled by extract_insights_jar's num_blocks==0
    wrap-and-retry, not something to split; a one-block file has nothing to split
    either, and passing through the original text (rather than just the matched span)
    preserves any surrounding boilerplate exactly as before this function existed.

    This is a splitter, not a lossless round-trip: whatever content sits before,
    between, or after the recognized blocks in a genuinely multi-block file is
    discarded. Use _has_content_outside_blocks separately first if you need to know
    whether that discarded content was non-trivial.
    """
    blocks = _BLOCK_SPAN_RE.findall(code)
    if len(blocks) <= 1:
        return [code]
    return blocks


def extract_insights_jar(plantuml_code: str, jar_path: str, java_home: str, timeout: int = 30) -> dict:
    """
    Like extract_insights, but determines diagram_type using the real PlantUML engine
    (get_diagram_type_jar) instead of regex heuristics alone.

    This does not require the code to already have passed any pre-filter -- every
    sample is run through the real engine, and the engine's own validity signal (not
    a Python-side heuristic) decides 'invalid'.

    diagram_type resolution:
    - 'non-uml': fails the same semantic content filter as extract_insights (unchanged)
    - no @start.../@end... at all (e.g. a diagram body missing its wrapper tags): retried
      once wrapped in synthetic @startuml/@enduml, since is_non_uml already filtered out
      non-UML text, so this is far more often a diagram missing its wrapper tags than
      genuinely ambiguous content -- real ground truth is available once wrapped, no
      need to guess via regex. Falls back to the regex patterns only if even the wrapped
      version still has no recognizable block at all.
    - 'mixed': reserved for a real, single, jar-valid diagram that legitimately
      combines more than one UML sub-type -- an explicit allow_mixing/allow mixing
      command (see RX_ALLOW_MIXING), or more than one of the fallback regex patterns
      matching (either for a jar-confirmed DESCRIPTION-family diagram, e.g.
      use_case/component/deployment commonly coexisting in one architecture-style
      diagram, or for the num_blocks==0 fallback classifying content the engine
      couldn't parse at all). This is real, single-diagram, single-block content --
      part of REAL_DIAGRAM_TYPES, not excluded the way the cases below are.
    - 'invalid': the real PlantUML engine reports a parse/syntax error for the
      (single) diagram block -- including a wrapped body that depends on
      !include/!includeurl content SANDBOX mode blocks it from fetching. Also covers
      what would otherwise look structurally "mixed" but isn't a real diagram at all:
      more than one @start.../@end... block (PlantUML viewers typically only render
      the last one), or non-trivial content left dangling outside the one recognized
      block -- neither represents one real diagram, so neither belongs in 'mixed'.
    - sequence/class/state/object/activity/timing/composite: taken directly from UmlDiagramType
    - DESCRIPTION (PlantUML's shared bucket for use_case/component/deployment):
      classified further with the same regex patterns extract_insights uses;
      'unknown' if none of them match
    - anything else (other non-UML diagram families the engine parsed successfully): 'unknown'

    Args:
        plantuml_code: PlantUML source to analyze
        jar_path: path to a plantuml.jar file
        java_home: path to a JDK installation used to run (and, if needed, compile) the jar
        timeout: seconds to allow the JVM to run before giving up
    """
    insights = extract_insights(plantuml_code)

    if insights['diagram_type'] == 'non-uml':
        return insights

    num_blocks, jar_type, jar_valid = get_diagram_type_jar(plantuml_code, jar_path, java_home=java_home,
                                                             timeout=timeout)

    # No @start.../@end... at all: PlantUML can't classify anything without them, so
    # retry once with synthetic tags added, to get real ground truth instead of
    # guessing via regex on content that's usually just missing its wrapper tags.
    # _has_content_outside_blocks doesn't apply to this retry -- the whole original
    # text is inside the single synthetic block by construction, so there's nothing
    # meaningful "outside" of it to check.
    # Content is always .strip()'d before being placed between synthetic @startuml/
    # @enduml tags -- confirmed PlantUML's !definelong/!enddefinelong handling is
    # sensitive to a blank line immediately adjacent to it (either right after
    # @startuml or right before @enduml). Since real content almost always already
    # ends with its own trailing newline, wrapping with an unconditional "\n{code}\n"
    # reliably introduces exactly that blank line before @enduml, turning what should
    # recover as PSystemWelcome into a genuine PSystemErrorV2 parse failure instead.
    used_synthetic_wrapper = False
    if num_blocks == 0:
        wrapped = f"@startuml\n{plantuml_code.strip()}\n@enduml"
        num_blocks, jar_type, jar_valid = get_diagram_type_jar(wrapped, jar_path, java_home=java_home,
                                                                 timeout=timeout)
        used_synthetic_wrapper = True

    if not jar_valid and jar_type == 'PSystemErrorPreprocessor':
        # A failure during preprocessing (as opposed to a genuine syntax error in the
        # diagram body, e.g. PSystemErrorV2) reliably means an !include/!includeurl
        # couldn't be resolved -- blocked by SANDBOX, a dead URL, or a missing local
        # file -- not that the diagram itself is broken. Retry once with those
        # directives stripped, since the rest of the diagram is often still perfectly
        # valid and classifiable without them (confirmed: stripping the include from a
        # real example turned a false 'invalid' into a correct 'sequence').
        stripped = RX_INCLUDE_LINE.sub('', plantuml_code)
        if stripped != plantuml_code:
            retry_source = f"@startuml\n{stripped.strip()}\n@enduml" if used_synthetic_wrapper else stripped
            num_blocks, jar_type, jar_valid = get_diagram_type_jar(retry_source, jar_path, java_home=java_home,
                                                                     timeout=timeout)

    if jar_valid and jar_type == 'PSystemWelcome':
        # !definelong(...)/!enddefinelong defines a reusable macro; if it's never
        # actually invoked, PlantUML preprocesses away all real content and falls back
        # to its placeholder "welcome" diagram instead of erroring. Retry with just the
        # delimiter lines stripped (keeping the body), turning the unused macro body
        # into directly-executable content (confirmed: recovers 'activity' on a real
        # example that only defined, but never called, a DOWNLOAD_ITEMS() macro).
        stripped = RX_DEFINELONG_LINE.sub('', plantuml_code)
        if stripped != plantuml_code:
            retry_source = f"@startuml\n{stripped.strip()}\n@enduml" if used_synthetic_wrapper else stripped
            num_blocks, jar_type, jar_valid = get_diagram_type_jar(retry_source, jar_path, java_home=java_home,
                                                                     timeout=timeout)

    if num_blocks == 0:
        # Still nothing, even wrapped -- no ground truth available at all.
        classified = _classify_by_patterns(plantuml_code)
        insights['diagram_type'] = classified if classified is not None else 'unknown'
    elif num_blocks > 1 or (not used_synthetic_wrapper and _has_content_outside_blocks(plantuml_code)):
        # Structurally not one real diagram -- multiple @start.../@end... blocks, or
        # non-trivial content left dangling outside the one recognized block. Not
        # 'mixed': that's reserved for a real, single, jar-valid diagram that
        # legitimately combines UML sub-types, which this isn't.
        insights['diagram_type'] = 'invalid'
    elif not jar_valid:
        insights['diagram_type'] = 'invalid'
    elif jar_type in _JAR_TYPE_MAP:
        mapped = _JAR_TYPE_MAP[jar_type]
        if mapped == 'class' and not re.search(_TYPE_PATTERNS['class'], plantuml_code, re.IGNORECASE | re.MULTILINE) \
                and re.search(_TYPE_PATTERNS['object'], plantuml_code, re.IGNORECASE | re.MULTILINE):
            # PlantUML's real UmlDiagramType never distinguishes object diagrams from
            # class diagrams -- confirmed even a minimal, package-free object diagram
            # (object A/object B, no class/interface/enum at all) reports CLASS, since
            # object diagrams share the same underlying grammar/factory as class
            # diagrams internally. Refine using the same regex used for the DESCRIPTION
            # fallback: only override to 'object' when there's no actual class/
            # interface/enum declaration alongside the object declarations, so a class
            # diagram that also happens to show a couple of illustrative object
            # instances still correctly stays 'class'.
            mapped = 'object'
        insights['diagram_type'] = mapped
    elif jar_type == 'DESCRIPTION':
        classified = _classify_by_patterns(plantuml_code)
        insights['diagram_type'] = classified if classified is not None else 'unknown'
    elif jar_type in _NON_UML_JAR_TYPES:
        # A real, distinct non-UML diagram family the regex @start<type> tag list
        # (RX_NON_UML_START_TAG/RX_TYPES) doesn't happen to enumerate -- see
        # _NON_UML_JAR_TYPES.
        insights['diagram_type'] = 'non-uml'
    else:
        insights['diagram_type'] = 'unknown'

    # An explicit author signal (PlantUML's own allowmixing/allow_mixing command --
    # see RX_ALLOW_MIXING) means the diagram deliberately combines element types that
    # would otherwise conflict; treat that the same as our own 'mixed' bucket (already
    # in REAL_DIAGRAM_TYPES, so this is a no-op if a DESCRIPTION-bucket multi-pattern
    # match already landed on 'mixed' above). Only checked once the diagram is already
    # confirmed real -- non-uml/invalid/unknown outcomes don't represent "confirmed
    # UML" to begin with, so the keyword's presence there wouldn't mean anything.
    if insights['diagram_type'] in REAL_DIAGRAM_TYPES and RX_ALLOW_MIXING.search(plantuml_code):
        insights['diagram_type'] = 'mixed'

    # Recompute element counts against the FINAL type resolved above -- the counts
    # already in `insights` are stale, from extract_insights()'s regex-only guess at
    # the very top of this function, which this whole function exists to correct.
    insights.update(_count_elements(plantuml_code, insights['diagram_type']))

    # Jar-confirmed-valid content that has zero real declarations of its own resolved
    # type (e.g. a title/skinparam-only stub the engine still happily parses as a
    # perfectly valid, empty CLASS diagram) is exactly what has_uml_content used to
    # try to catch pre-jar, via a hand-maintained keyword/arrow regex -- confirmed
    # unreliable (false-positived on real sequence and component diagrams; see
    # is_non_uml in filter_data.py, which no longer calls it for this purpose). This
    # is the same check grounded in what the engine actually resolved instead of a
    # pre-jar guess, so it can't misfire the same way. Only defined for the four types
    # with an existing, reliable type-specific declaration count -- sequence/state/
    # activity/timing/composite/deployment have no such count to check against, so are
    # deliberately left alone here.
    diagram_type = insights['diagram_type']
    if diagram_type == 'class' and insights['num_classes'] == 0:
        insights['diagram_type'] = 'non-uml'
    elif diagram_type == 'component' and insights['num_components'] == 0:
        insights['diagram_type'] = 'non-uml'
    elif diagram_type == 'object' and insights['num_objects'] == 0:
        insights['diagram_type'] = 'non-uml'
    elif diagram_type == 'use_case' and insights['num_usecases'] == 0 and insights['num_actors'] == 0:
        insights['diagram_type'] = 'non-uml'

    return insights


# A `footer`/`endfooter` block (optionally prefixed with left/right/center alignment)
# is real PlantUML syntax that can legitimately hold diagram content, so this only
# matches -- and strip_sketchit_footer only removes -- a block that actually contains
# the "SketchIt!" mention, not any footer block in general. Narrow and low-risk by
# design, same philosophy as the generator-attribution check in filter_data.py: exact
# tool-attribution match, not "this looks like boilerplate."
RX_FOOTER_BLOCK = re.compile(
    r'(?ims)^[ \t]*(?:(?:left|right|center)[ \t]+)?footer[ \t]*\n.*?^[ \t]*endfooter[ \t]*$\n?'
)


def strip_sketchit_footer(code: str) -> tuple[str, bool]:
    """
    Remove SketchIt!'s (https://bitbucket.org/pmesmeur/sketch.it) generated
    attribution footer, e.g.:
        right footer


        PlantUML diagram generated by SketchIt! (https://bitbucket.org/pmesmeur/sketch.it)
        For more information about this tool, please contact philippe.mesmeur@gmail.com
        endfooter
    It's pure tool attribution, not diagram content -- the rest of a
    SketchIt-generated diagram is ordinary, legitimate content -- only this footer
    block needs removing.

    Returns:
        tuple: (code_with_footer_removed, was_removed)
    """
    def _maybe_strip(match: re.Match) -> str:
        return '' if 'SketchIt' in match.group(0) else match.group(0)

    stripped = RX_FOOTER_BLOCK.sub(_maybe_strip, code)
    return stripped, stripped != code


def repair_unbalanced_tags(code: str) -> tuple[str, bool]:
    """
    Fix a mismatched @startuml/@enduml count -- the most common real-world cause is a
    scraped/truncated sample missing just its closing (or, less often, opening) tag,
    not a genuinely different diagram. Appends the missing @enduml line(s) at the end
    when there are more starts than ends, or prepends the missing @startuml line(s) at
    the beginning when there are more ends than starts.

    This is a best-effort text patch, not a structural fix: if more than one tag is
    missing, the added lines are all clustered at one end rather than interleaved at
    wherever each one actually belongs, so multi-block files with more than one
    missing tag may still not come out semantically correct. Only fires when
    @startuml appears at all -- content with neither tag is a different situation,
    already handled by extract_insights_jar's num_blocks==0 wrap-and-retry.

    Returns:
        tuple: (repaired_code, was_repaired). repaired_code is the original code
        unchanged if no repair was made (including when none was needed).
    """
    if '@startuml' not in code.lower():
        return code, False

    start_count = len(re.findall(r'@startuml', code, re.IGNORECASE))
    end_count = len(re.findall(r'@enduml', code, re.IGNORECASE))
    if start_count == end_count:
        return code, False

    if start_count > end_count:
        missing = start_count - end_count
        repaired = code.rstrip('\n') + ('\n@enduml' * missing) + '\n'
    else:
        missing = end_count - start_count
        repaired = ('@startuml\n' * missing) + code

    return repaired, True


def strip_outer_whitespace(code: str) -> tuple[str, bool]:
    """
    Strip leading/trailing whitespace (blank lines, trailing spaces) from the sample
    as a whole. Purely cosmetic and outer-only -- doesn't touch indentation or
    whitespace anywhere in the middle of the diagram, just the two ends.

    Returns:
        tuple: (stripped_code, was_changed)
    """
    stripped = code.strip()
    return stripped, stripped != code


# ---------------------- Shared dataset-level pipeline steps ----------------------
# The three functions below operate on a datasets.Dataset (or one of its example
# dicts/batches), not just a bare code string -- unlike the atomic functions above,
# which every caller composes for itself. Kept here, shared, so every script that
# builds a dataset from raw PlantUML source runs the exact same cleaning pipeline in
# the exact same order, instead of each maintaining its own copy that can drift out
# of sync.


def split_multi_block_rows(batch):
    """
    Expand any row containing more than one @start.../@end... block into one row per
    block (see split_into_blocks); rows with 0 or 1 blocks pass through unchanged.

    Should run as close to the very start of a pipeline as possible -- before dedup,
    repair, or any filter -- so each sub-diagram gets the full cleaning/filtering
    treatment as its own independent sample, rather than one syntax error in a single
    sub-diagram dragging the other, otherwise-valid sub-diagrams in the same file down
    into 'mixed' or 'invalid' too.

    Use via ds.map(split_multi_block_rows, batched=True, ...) -- batched because this
    can change the total row count, which a plain (non-batched) .map() can't do.
    """
    out = {key: [] for key in batch}
    n = len(batch["code"])
    for i in range(n):
        for block in split_into_blocks(batch["code"][i]):
            for key in batch:
                out[key].append(block if key == "code" else batch[key][i])
    return out


def dedup_by_code(ds, logger):
    """Drop rows whose `code` string exactly matches an earlier row's, keeping the first occurrence."""
    seen = set()

    def is_first_occurrence(example):
        # Stripped so purely-cosmetic leading/trailing whitespace differences
        # between otherwise-identical samples don't defeat the match.
        code = example["code"].strip()
        if code in seen:
            return False
        seen.add(code)
        return True

    before = len(ds)
    # Must run single-threaded: `seen` is shared mutable state that has to be
    # consistent across every row in order, which num_proc workers (separate
    # processes, separate `seen` copies) can't provide.
    ds = ds.filter(is_first_occurrence)
    after = len(ds)
    logger.info(f" > removed {before - after} exact-duplicate rows by code content ({before} -> {after})")
    return ds


def repair_row(example):
    """
    Shared preprocessing step: strip SketchIt's attribution footer, fix a mismatched
    @startuml/@enduml count, then trim outer whitespace -- all before any filter runs,
    so every downstream filter/classifier (and whatever ends up saved) sees the same
    cleaned-up code, not just whichever filter happened to check for it. Whitespace
    trimming runs last since the two repair steps can themselves introduce or remove
    a trailing newline (see their docstrings), and the final code should be trimmed
    either way.
    """
    code, footer_stripped = strip_sketchit_footer(example["code"])
    code, tags_repaired = repair_unbalanced_tags(code)
    code, whitespace_stripped = strip_outer_whitespace(code)
    return {
        "code": code,
        "footer_stripped": footer_stripped,
        "tags_repaired": tags_repaired,
        "whitespace_stripped": whitespace_stripped,
    }
