"""Render display formulas used by the Markdown documentation."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
import tomllib
from io import BytesIO
from pathlib import Path
from typing import TypedDict


class FormulaEntry(TypedDict):
    """Describe one generated documentation formula."""

    name: str
    latex: str
    alt: str
    references: list[str]
    placement: str


SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SOURCE_DIR.parent
DOCS_DIR = OUTPUT_DIR.parent.parent
REPOSITORY_ROOT = DOCS_DIR.parent
MANIFEST_PATH = SOURCE_DIR / "formulas.toml"
SVG_METADATA_PATTERN = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)
SVG_HASH_PATTERN = re.compile(r'data-formula-sha256="([0-9a-f]{64})"')
SVG_DEPTH_PATTERN = re.compile(r'data-formula-depth-pt="([0-9.]+)"')
DISPLAY_BLOCK_PATTERN = re.compile(r'<p align="center">\s*(.*?)\s*</p>', re.DOTALL)
RENDERER_VERSION = "3"


def _load_manifest() -> list[FormulaEntry]:
    """Load and validate the formula manifest."""
    data = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    formulas = data.get("formula")
    if not isinstance(formulas, list) or not formulas:
        raise ValueError("formulas.toml must contain at least one [[formula]] entry")

    entries: list[FormulaEntry] = []
    names: set[str] = set()
    for raw_entry in formulas:
        if not isinstance(raw_entry, dict):
            raise TypeError("each formula entry must be a table")
        entry = FormulaEntry(
            name=str(raw_entry["name"]),
            latex=str(raw_entry["latex"]),
            alt=str(raw_entry["alt"]),
            references=[str(value) for value in raw_entry["references"]],
            placement=str(raw_entry.get("placement", "display")),
        )
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry["name"]):
            raise ValueError(
                "formula names must contain lowercase letters, digits, and hyphens"
            )
        if entry["name"] in names:
            raise ValueError("duplicate formula name: " + entry["name"])
        if not entry["latex"].strip() or not entry["alt"].strip():
            raise ValueError("formula source and alt text must be non-empty")
        if not entry["references"]:
            raise ValueError("formula references must be non-empty")
        if entry["placement"] not in {"display", "inline"}:
            raise ValueError("formula placement must be 'display' or 'inline'")
        if entry["placement"] == "inline":
            digest = hashlib.sha256(entry["latex"].encode("utf-8")).hexdigest()[:16]
            if entry["name"] != "inline-" + digest:
                raise ValueError(
                    "inline formula name must match its source hash: " + entry["name"]
                )
        names.add(entry["name"])
        entries.append(entry)
    return entries


def _mathtext_source(source: str) -> str:
    """Return a mathematically equivalent Mathtext-compatible expression."""
    if source.startswith(r"S_{F,k}|x\rangle="):
        return (
            r"S_{F,k}|x\rangle=(-1)^{\mathrm{parity}"
            r"(\mathrm{bitand}(k,F_{\mathrm{int}}(x)))}|x\rangle"
        )
    if source.startswith("g(y,x,l)="):
        return (
            r"g(y,x,l)=\mathrm{bitand}"
            r"(\mathtt{pixel}[y,x]\gg l,1)"
        )
    if source.startswith(r"(-1)^{\operatorname{popcount}"):
        return (
            r"(-1)^{\mathrm{popcount}(\mathrm{bitand}"
            r"(\mathrm{int}(\mathtt{bitstring},2),\mathtt{mask}))}"
        )
    if source.startswith(r"P|j\rangle="):
        return (
            r"P|j\rangle=\mathtt{phase}\,\mathtt{y\_phase}"
            r"(-1)^{\mathrm{popcount}(\mathrm{bitand}"
            r"(j,\mathtt{z\_mask}))}"
            r"|\mathrm{xor}(j,\mathtt{x\_mask})\rangle"
        )

    value = source.replace(r"\tfrac12", r"\frac{1}{2}")
    value = value.replace(r"\tfrac", r"\frac")
    value = value.replace(r"\mathcal S", r"\mathcal{S}")
    value = value.replace(r"\operatorname{parity}", r"\mathrm{parity}")
    value = value.replace(r"\mathbin{\&}", r"\&")
    value = value.replace(r"\mathbin{|}", "|")
    value = value.replace(r"\bmod", r"\;\mathrm{mod}\;")
    return re.sub(
        r"\\pmod\s*(\{[^}]+\}|[^\s,.]+)",
        lambda match: r"\;(\mathrm{mod}\ " + match.group(1) + ")",
        value,
    )


def _render_svg(entry: FormulaEntry) -> tuple[bytes, float]:
    """Return deterministic SVG bytes and baseline depth for one formula."""
    try:
        import matplotlib as mpl
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
    except ImportError as exc:
        raise RuntimeError(
            "formula rendering requires Matplotlib; install it only in the "
            "documentation-maintenance environment"
        ) from exc

    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["savefig.transparent"] = True
    mpl.rcParams["svg.fonttype"] = "path"
    mpl.rcParams["svg.hashsalt"] = "cqlib-algorithms-doc-formulas-v2"
    buffer = BytesIO()
    depth = math_to_image(
        "$" + _mathtext_source(entry["latex"]) + "$",
        buffer,
        format="svg",
        dpi=180,
        color="#111827",
        prop=FontProperties(size=14),
    )
    svg = buffer.getvalue().decode("utf-8")
    svg = SVG_METADATA_PATTERN.sub("", svg)

    source_hash = _source_hash(entry)
    title_id = "formula-" + entry["name"]
    svg = svg.replace(
        "<svg ",
        (
            '<svg role="img" aria-labelledby="'
            + title_id
            + '" data-formula-sha256="'
            + source_hash
            + '" data-formula-depth-pt="'
            + f"{float(depth):.3f}"
            + '" '
        ),
        1,
    )
    title = '<title id="' + title_id + '">' + html.escape(entry["alt"]) + "</title>"
    svg = re.sub(
        r"(<svg\b[^>]*>)",
        lambda match: match.group(1) + "\n " + title,
        svg,
        count=1,
    )
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    return svg.encode("utf-8"), float(depth)


def _source_hash(entry: FormulaEntry) -> str:
    """Return the content hash stored in a generated formula asset."""
    source = "\0".join(
        (
            RENDERER_VERSION,
            entry["placement"],
            entry["latex"],
            entry["alt"],
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _asset_hash(path: Path) -> str | None:
    """Return the recorded source hash from an existing SVG asset."""
    if not path.is_file():
        return None
    match = SVG_HASH_PATTERN.search(path.read_text(encoding="utf-8"))
    return match.group(1) if match is not None else None


def _asset_depth(path: Path) -> float | None:
    """Return the recorded baseline depth from an existing SVG asset."""
    if not path.is_file():
        return None
    match = SVG_DEPTH_PATTERN.search(path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match is not None else None


def _asset_path(entry: FormulaEntry) -> Path:
    """Return the generated asset path for a formula entry."""
    directory = OUTPUT_DIR / "inline" if entry["placement"] == "inline" else OUTPUT_DIR
    return directory / (entry["name"] + ".svg")


def _reference_path(document: Path, entry: FormulaEntry) -> str:
    """Return a POSIX relative path from a document to a generated formula."""
    target = _asset_path(entry)
    return Path(os.path.relpath(target, document.parent)).as_posix()


def _check_references(entries: list[FormulaEntry]) -> list[str]:
    """Return documentation-reference errors."""
    errors: list[str] = []
    for entry in entries:
        for reference in entry["references"]:
            document = REPOSITORY_ROOT / reference
            if not document.is_file():
                errors.append("missing reference document: " + reference)
                continue
            source_path = _reference_path(document, entry)
            text = document.read_text(encoding="utf-8")
            if entry["placement"] == "inline":
                depth = _asset_depth(_asset_path(entry))
                if depth is None:
                    errors.append("missing baseline depth for " + entry["name"])
                    continue
                expected = (
                    '<img src="'
                    + source_path
                    + '" alt="'
                    + html.escape(entry["alt"], quote=True)
                    + '" style="vertical-align: -'
                    + f"{depth:.3f}"
                    + 'pt;">'
                )
            else:
                expected = 'src="' + source_path + '"'
            if expected not in text:
                errors.append(
                    reference
                    + " does not reference "
                    + entry["name"]
                    + " using "
                    + expected
                )
    return errors


def _check_markdown_formula_style() -> list[str]:
    """Return errors for display formulas that bypass the generated assets."""
    errors: list[str] = []
    for document in sorted(DOCS_DIR.rglob("*.md")):
        if document.name == "documentation-style.md":
            continue
        text = document.read_text(encoding="utf-8")
        for block in DISPLAY_BLOCK_PATTERN.findall(text):
            if "<img " not in block or "assets/formulas/" not in block:
                errors.append(
                    document.relative_to(REPOSITORY_ROOT).as_posix()
                    + " contains a non-generated display formula"
                )
        if "$$" in text or "\\(" in text or "\\[" in text:
            errors.append(
                document.relative_to(REPOSITORY_ROOT).as_posix()
                + " contains an unsupported math delimiter"
            )
        if "<sub>" in text or "<sup>" in text:
            errors.append(
                document.relative_to(REPOSITORY_ROOT).as_posix()
                + " contains HTML math instead of a generated inline formula"
            )
    return errors


def _render(entries: list[FormulaEntry], *, check: bool, force: bool) -> int:
    """Render changed formulas or verify that generated assets are current."""
    errors: list[str] = []
    expected_names = {
        _asset_path(entry).relative_to(OUTPUT_DIR).as_posix() for entry in entries
    }
    existing_names = (
        {path.relative_to(OUTPUT_DIR).as_posix() for path in OUTPUT_DIR.rglob("*.svg")}
        if OUTPUT_DIR.is_dir()
        else set()
    )
    for orphan in sorted(existing_names - expected_names):
        errors.append("orphan generated formula: " + orphan)

    if not check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    changed = 0
    for entry in entries:
        target = _asset_path(entry)
        current_hash = _asset_hash(target)
        expected_hash = _source_hash(entry)
        if current_hash == expected_hash and not force:
            continue
        if check:
            errors.append("stale or missing generated formula: " + target.name)
            continue
        rendered, _depth = _render_svg(entry)
        current = target.read_bytes() if target.is_file() else None
        if current == rendered:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rendered)
        changed += 1
        print("rendered " + target.relative_to(REPOSITORY_ROOT).as_posix())

    errors.extend(_check_references(entries))
    errors.extend(_check_markdown_formula_style())
    if errors:
        for error in errors:
            print("error: " + error, file=sys.stderr)
        return 1

    if check:
        print(
            f"verified {len(entries)} generated formulas and their Markdown references"
        )
    else:
        print(f"updated {changed} of {len(entries)} generated formulas")
    return 0


def main() -> int:
    """Render documentation formulas or check committed assets."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if an asset or Markdown reference is missing or stale",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate every formula after an intentional renderer change",
    )
    args = parser.parse_args()
    if args.check and args.force:
        parser.error("--check and --force cannot be combined")
    return _render(_load_manifest(), check=args.check, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
