#!/usr/bin/env python3
"""Build the CubeFit PDF manual from the MkDocs navigation tree.

The PDF source order is taken directly from ``mkdocs.yml``. A temporary
Pandoc Lua filter assigns bounded, proportional widths to Markdown tables and
renders all inline-code tokens with breakable monospace LaTeX. This prevents
long paths, environment-variable names, function names, and API identifiers
from running through the right margin. Fenced code blocks are left unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "mkdocs.yml"
DEFAULT_OUTPUT = "docs/CubeFit.pdf"


LATEX_HEADER = r"""
% CubeFit PDF table/layout support.
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{url}
\usepackage{xurl}
\usepackage{etoolbox}
\usepackage{microtype}

% Keep tables compact enough for technical identifiers without making the
% surrounding document small. Widths themselves are assigned by the Lua
% filter below and always sum to <= \textwidth.
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\AtBeginEnvironment{longtable}{\small}

% Permit sensible line breaking in long paths and identifiers.  Inline code
% is emitted as \nolinkurl{...} by the Lua filter, so xurl supplies legal
% break points without inserting visible spaces or changing the monospace
% appearance.
\Urlmuskip=0mu plus 2mu\relax

% Give TeX a little extra flexibility before it resorts to an overfull box.
% This affects paragraph line fitting only; it does not shrink the document.
\setlength{\emergencystretch}{2em}
""".strip()


LUA_FILTER = r"""
-- PDF layout filter for CubeFit.
--
-- 1. Constrain Pandoc table columns to proportional fractions of the
--    printable width.
-- 2. Render *all* inline Code nodes with \\nolinkurl{...}.  With xurl loaded,
--    this preserves monospace styling while allowing long Python identifiers,
--    HDF5 paths, filenames, and environment variables to wrap at punctuation.
-- 3. Leave CodeBlock nodes alone: source-code blocks retain Pandoc's normal
--    verbatim/highlighted rendering.

local function widths_for(n)
  if n == 1 then
    return {1.00}
  elseif n == 2 then
    return {0.30, 0.70}
  elseif n == 3 then
    return {0.34, 0.18, 0.48}
  elseif n == 4 then
    return {0.24, 0.16, 0.20, 0.40}
  end

  local widths = {}
  for i = 1, n do
    widths[i] = 1.0 / n
  end
  return widths
end

local function latex_url_escape(value)
  -- Arguments to \\nolinkurl are read with url-package semantics rather than
  -- normal LaTeX text semantics.  Braces and percent signs still need
  -- protection because they can terminate the command argument or begin a
  -- comment before url processing sees the complete token.
  value = value:gsub("\\\\", "/")
  value = value:gsub("%%", "\\%%")
  value = value:gsub("{", "\\{")
  value = value:gsub("}", "\\}")
  return value
end

function Code(code)
  if FORMAT:match("latex") then
    local value = latex_url_escape(code.text)
    return pandoc.RawInline(
      "latex",
      "\\nolinkurl{" .. value .. "}"
    )
  end
  return code
end

function Table(tbl)
  local n = #tbl.colspecs
  local widths = widths_for(n)

  for i = 1, n do
    local align = tbl.colspecs[i][1]
    tbl.colspecs[i] = {align, widths[i]}
  end

  return tbl
end
""".strip()


def _collect_nav_files(nav: Any) -> list[str]:
    """Return Markdown paths from an MkDocs ``nav`` tree in nav order."""
    files: list[str] = []

    if isinstance(nav, str):
        if nav.lower().endswith((".md", ".markdown")):
            files.append(nav)
        return files

    if isinstance(nav, list):
        for item in nav:
            files.extend(_collect_nav_files(item))
        return files

    if isinstance(nav, dict):
        for value in nav.values():
            files.extend(_collect_nav_files(value))
        return files

    if nav is None:
        return files

    raise TypeError(
        "Unsupported MkDocs navigation entry: "
        f"{type(nav).__name__}: {nav!r}"
    )


def _load_mkdocs_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the MkDocs configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"MkDocs configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid MkDocs configuration in {config_path}: "
            "expected a YAML mapping."
        )

    if "nav" not in config:
        raise ValueError(
            f"No 'nav' section found in {config_path}."
        )

    return config


def _resolve_sources(
    config: dict[str, Any],
    config_path: Path,
) -> list[Path]:
    """Resolve MkDocs navigation entries to Markdown source files."""
    config_dir = config_path.parent.resolve()
    docs_dir = (config_dir / config.get("docs_dir", "docs")).resolve()
    nav_files = _collect_nav_files(config["nav"])

    if not nav_files:
        raise ValueError(
            "No Markdown documents were found in the MkDocs navigation."
        )

    sources: list[Path] = []
    seen: set[Path] = set()

    for relative_path in nav_files:
        source = (docs_dir / relative_path).resolve()
        if source in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(
                "MkDocs navigation references a missing document: "
                f"{relative_path}\nExpected: {source}"
            )
        sources.append(source)
        seen.add(source)

    return sources


def _check_executable(name: str) -> None:
    """Raise an informative error if an executable is unavailable."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH."
        )


def _pandoc_command(
    sources: list[Path],
    output: Path,
    header_path: Path,
    filter_path: Path,
    title: str,
    subtitle: str,
    mainfont: str,
    monofont: str,
) -> list[str]:
    """Construct the Pandoc command for the consolidated PDF."""
    command = [
        "pandoc",
        "--standalone",
        "--from=gfm",
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--metadata",
        f"title={title}",
        "--metadata",
        f"subtitle={subtitle}",
        "--include-in-header",
        str(header_path),
        "--lua-filter",
        str(filter_path),
        "-V",
        "geometry:margin=1in",
        "-V",
        f"mainfont={mainfont}",
        "-V",
        f"monofont={monofont}",
    ]
    command.extend(str(source) for source in sources)
    command.extend(["-o", str(output)])
    return command


def build_pdf(
    config_path: Path,
    output: Path,
    title: str,
    subtitle: str,
    mainfont: str,
    monofont: str,
    dry_run: bool = False,
) -> None:
    """Build the consolidated CubeFit PDF documentation."""
    config_path = config_path.resolve()
    output = output.resolve()
    config = _load_mkdocs_config(config_path)
    sources = _resolve_sources(config, config_path)

    if not dry_run:
        _check_executable("pandoc")
        _check_executable("xelatex")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cubefit-docs-") as temp_dir:
        temp_path = Path(temp_dir)
        header_path = temp_path / "cubefit-pdf-header.tex"
        filter_path = temp_path / "cubefit-tables.lua"
        header_path.write_text(LATEX_HEADER + "\n", encoding="utf-8")
        filter_path.write_text(LUA_FILTER + "\n", encoding="utf-8")

        # RawInline LaTeX must contain one command backslash.  A doubled
        # prefix would emit ``\\nolinkurl`` into the .tex file, leaving
        # underscores outside a verbatim-like command and causing XeLaTeX
        # errors such as "Missing $ inserted".
        if r"\\\\nolinkurl" in LUA_FILTER:
            raise RuntimeError(
                "Internal PDF filter error: doubled LaTeX command prefix "
                "for nolinkurl."
            )

        command = _pandoc_command(
            sources=sources,
            output=output,
            header_path=header_path,
            filter_path=filter_path,
            title=title,
            subtitle=subtitle,
            mainfont=mainfont,
            monofont=monofont,
        )

        print(f"Documentation sources: {len(sources)}")
        for index, source in enumerate(sources, start=1):
            print(f"  {index:2d}. {source}")
        print(f"\nOutput: {output}\n")

        if dry_run:
            import shlex
            print(" ".join(shlex.quote(item) for item in command))
            return

        subprocess.run(command, check=True)

    if not output.is_file():
        raise RuntimeError(
            f"Pandoc completed but did not create {output}."
        )

    size_mb = output.stat().st_size / (1024.0 * 1024.0)
    print(f"Built {output} ({size_mb:.2f} MiB)")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the CubeFit PDF manual from the navigation order "
            "defined in mkdocs.yml."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="CubeFit")
    parser.add_argument(
        "--subtitle",
        default="Documentation and Reference Manual",
    )
    parser.add_argument("--mainfont", default="Open Sans")
    parser.add_argument(
        "--monofont",
        default="IntoneMono Nerd Font Mono",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sources",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the documentation builder."""
    args = _parse_args()

    try:
        config_path = Path(args.config).resolve()
        config = _load_mkdocs_config(config_path)
        sources = _resolve_sources(config, config_path)

        if args.list_sources:
            for source in sources:
                print(source)
            return 0

        build_pdf(
            config_path=config_path,
            output=Path(args.output),
            title=args.title,
            subtitle=args.subtitle,
            mainfont=args.mainfont,
            monofont=args.monofont,
            dry_run=args.dry_run,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
