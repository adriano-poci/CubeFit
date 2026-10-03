#!/usr/bin/env python3
"""Build CubeFit HTML and the frozen-style PDF from docs/CubeFit.md."""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

from pdf_style import (
    BLACK, BODY, BOTTOM_MARGIN, FONT_MONO, FONT_SANS, FONT_SANS_BOLD,
    HEADER_TEXT, HEADING, LEFT_MARGIN, MUTED, PAGE_HEIGHT, PAGE_SIZE,
    PAGE_WIDTH, RIGHT_MARGIN, SUBHEADING, TABLE_HEAD, TABLE_HEAD_RULE,
    TABLE_RULE, TOP_MARGIN,
    paragraph_styles,
)


def _font_path(family: str, style: str = "Regular") -> str:
    """Resolve a system font without bundling or redistributing font files."""
    query = family if style == "Regular" else f"{family}:style={style}"
    try:
        out = subprocess.check_output(
            ["fc-match", "-f", "%{file}", query], text=True).strip()
        if out and Path(out).is_file():
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    candidates = {
        ("FreeSans", "Regular"): [
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"],
        ("FreeSans", "Bold"): [
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"],
        ("FreeMono", "Regular"): [
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf"],
    }
    for path in candidates.get((family, style), []):
        if Path(path).is_file():
            return path
    raise RuntimeError(
        f"Required font {family} {style} was not found. Install the GNU "
        "FreeFont package to reproduce the frozen CubeFit PDF style.")


def register_fonts() -> None:
    """Register the exact font families used by the frozen reference PDF."""
    pdfmetrics.registerFont(TTFont(
        FONT_SANS, _font_path("FreeSans", "Regular")))
    pdfmetrics.registerFont(TTFont(
        FONT_SANS_BOLD, _font_path("FreeSans", "Bold")))
    pdfmetrics.registerFont(TTFont(
        FONT_MONO, _font_path("FreeMono", "Regular")))


class CubeFitDocTemplate(BaseDocTemplate):
    """A4 document template with the frozen CubeFit header/footer geometry."""

    def __init__(self, filename: str, **kwargs):
        super().__init__(
            filename, pagesize=PAGE_SIZE, leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN, topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN, title="CubeFit Documentation",
            author="CubeFit documentation overhaul", **kwargs)
        frame = Frame(
            LEFT_MARGIN, BOTTOM_MARGIN,
            PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN,
            PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
            id="normal")
        self.addPageTemplates([
            PageTemplate(id="normal", frames=frame,
                         onPage=self._header_footer),
        ])

    def _header_footer(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont(FONT_SANS, 8.0)
        canvas.setFillColor(MUTED)
        canvas.drawString(56.7, PAGE_HEIGHT - 35.6, HEADER_TEXT)
        canvas.drawRightString(PAGE_WIDTH - 62.7, 30.0, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        level = getattr(flowable, "_cubefit_toc_level", None)
        if level is None:
            return
        text = flowable.getPlainText()
        key = getattr(flowable, "_cubefit_key", None)
        if key is None:
            return
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def _inline_markup(token) -> str:
    """Convert markdown-it inline children to ReportLab Paragraph markup."""
    if token.type != "inline" or not token.children:
        return html.escape(token.content)
    out = []
    link_stack = []
    for child in token.children:
        typ = child.type
        if typ == "text":
            out.append(html.escape(child.content))
        elif typ == "code_inline":
            out.append(
                f'<font name="{FONT_MONO}">{html.escape(child.content)}</font>')
        elif typ == "strong_open":
            out.append("<b>")
        elif typ == "strong_close":
            out.append("</b>")
        elif typ == "em_open":
            out.append("<i>")
        elif typ == "em_close":
            out.append("</i>")
        elif typ == "softbreak":
            out.append(" ")
        elif typ == "hardbreak":
            out.append("<br/>")
        elif typ == "link_open":
            href = child.attrGet("href") or ""
            link_stack.append(href)
            out.append(f'<a href="{html.escape(href)}" color="#245B7C">')
        elif typ == "link_close":
            if link_stack:
                link_stack.pop()
            out.append("</a>")
        else:
            out.append(html.escape(child.content or ""))
    return "".join(out)


def _plain_inline(token) -> str:
    if token.type == "inline":
        return token.content
    return ""


def _extract_front_matter(tokens):
    """Extract the title-page fields and return the first body token index."""
    title = "CubeFit"
    subtitle = (
        "Architecture, mathematics, operation, diagnostics, and validation")
    revision = ""
    source = ""
    i = 0
    headings = []
    while i < len(tokens):
        if tokens[i].type == "hr":
            return title, subtitle, revision, source, i + 1
        if tokens[i].type == "heading_open" and i + 1 < len(tokens):
            level = int(tokens[i].tag[1])
            text = tokens[i + 1].content
            headings.append((level, text))
            if level == 1 and len(headings) == 1:
                title = text
            elif level == 2 and len(headings) == 2:
                subtitle = text
            i += 3
            continue
        if tokens[i].type == "paragraph_open" and i + 1 < len(tokens):
            raw = tokens[i + 1].content
            m = re.search(r"Documentation revision:\*\*\s*(.+?)(?:\n|$)", raw)
            if not m:
                m = re.search(r"Documentation revision:\s*(.+?)(?:\n|$)", raw)
            if m:
                revision = m.group(1).strip()
            m = re.search(r"Source basis:\*\*\s*(.+)", raw, re.S)
            if not m:
                m = re.search(r"Source basis:\s*(.+)", raw, re.S)
            if m:
                source = re.sub(r"\s+", " ", m.group(1)).strip()
            i += 3
            continue
        i += 1
    return title, subtitle, revision, source, i


def _cover_story(styles, title, subtitle, revision, source):
    source_line = source
    if source:
        source_line = (
            "Grounded in the supplied complete Repomix repository snapshot.")
    return [
        Spacer(1, 99.2),
        Paragraph(html.escape(title), styles["cover_title"]),
        Spacer(1, 22.7),
        Paragraph(html.escape(subtitle), styles["cover_text"]),
        Spacer(1, 33.7),
        Paragraph(
            html.escape(f"Documentation revision: {revision}"),
            styles["cover_text"]),
        Spacer(1, 11.3),
        Paragraph(html.escape(source_line), styles["cover_text"]),
        Spacer(1, 68.1),
        Paragraph("Current production pathway", styles["cover_text"]),
        Spacer(1, 11.3),
        Paragraph(
            "genCubeFit -&gt; PipelineRunner.solve_all_mp_batched<br/>"
            "            -&gt; solve_streaming_nnls -&gt; streamActiveSetNNLS",
            styles["cover_code"]),
        PageBreak(),
    ]


def _toc_story(styles):
    toc = TableOfContents()
    toc.levelStyles = [styles["toc0"], styles["toc1"]]
    toc.dotsMinLevel = 0
    self_row = Table(
        [[Paragraph("Contents", styles["toc_self"]),
          Paragraph("2", styles["toc_self"])]],
        colWidths=[
            PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - 24.0, 24.0],
        rowHeights=[18.0], hAlign="LEFT")
    self_row.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    return [Paragraph("Contents", styles["toc_title"]),
            self_row, toc, PageBreak()]


def _table_from_tokens(tokens, start, styles, avail_width):
    rows = []
    header_rows = 0
    row = []
    i = start + 1
    in_head = False
    while i < len(tokens) and tokens[i].type != "table_close":
        typ = tokens[i].type
        if typ == "thead_open":
            in_head = True
        elif typ == "thead_close":
            in_head = False
        elif typ == "tr_open":
            row = []
        elif typ in ("th_open", "td_open"):
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                style = (styles["table_head"] if in_head
                         else styles["table_body"])
                row.append(Paragraph(_inline_markup(tokens[i + 1]), style))
                i += 1
        elif typ == "tr_close":
            rows.append(row)
            if in_head:
                header_rows += 1
        i += 1
    ncols = max((len(r) for r in rows), default=1)
    if ncols == 3:
        widths = [0.17 * avail_width, 0.48 * avail_width, 0.35 * avail_width]
    elif ncols == 2:
        widths = [0.34 * avail_width, 0.66 * avail_width]
    else:
        widths = [avail_width / ncols] * ncols
    table = Table(rows, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6.0),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, TABLE_RULE),
    ]
    if header_rows:
        commands.append((
            "BACKGROUND", (0, 0), (-1, header_rows - 1), TABLE_HEAD))
        commands.append((
            "LINEBELOW", (0, header_rows - 1),
            (-1, header_rows - 1), 0.6, TABLE_HEAD_RULE))
    table.setStyle(TableStyle(commands))
    return table, i + 1


def _body_story(tokens, start, styles):
    story = []
    width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    i = start
    heading_index = 0
    list_stack = []
    quote_depth = 0
    while i < len(tokens):
        tok = tokens[i]
        typ = tok.type
        if typ == "heading_open":
            level = int(tok.tag[1])
            if i + 1 < len(tokens):
                text = _inline_markup(tokens[i + 1])
                style_name = {2: "h1", 3: "h2", 4: "h3"}.get(level, "h3")
                p = Paragraph(text, styles[style_name])
                if level in (2, 3):
                    p._cubefit_toc_level = level - 2
                    p._cubefit_key = f"heading-{heading_index}"
                    heading_index += 1
                story.append(p)
            i += 3
            continue
        if typ == "paragraph_open":
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                style = styles["quote"] if quote_depth else styles["body"]
                markup = _inline_markup(tokens[i + 1])
                if list_stack:
                    ordered, counter = list_stack[-1]
                    if ordered:
                        bullet = f"{counter}."
                        list_stack[-1] = (ordered, counter + 1)
                        story.append(Paragraph(
                            markup, styles["number"], bulletText=bullet))
                    else:
                        story.append(Paragraph(
                            markup, styles["bullet"], bulletText="•"))
                else:
                    story.append(Paragraph(markup, style))
            i += 3
            continue
        if typ == "fence" or typ == "code_block":
            story.append(XPreformatted(
                html.escape(tok.content.rstrip("\n")), styles["code"]))
            i += 1
            continue
        if typ == "table_open":
            table, i = _table_from_tokens(tokens, i, styles, width)
            story.append(table)
            story.append(Spacer(1, 5.0))
            continue
        if typ == "bullet_list_open":
            list_stack.append((False, 1))
        elif typ == "ordered_list_open":
            start_num = int(tok.attrGet("start") or 1)
            list_stack.append((True, start_num))
        elif typ in ("bullet_list_close", "ordered_list_close"):
            if list_stack:
                list_stack.pop()
            story.append(Spacer(1, 2.0))
        elif typ == "blockquote_open":
            quote_depth += 1
        elif typ == "blockquote_close":
            quote_depth = max(0, quote_depth - 1)
        elif typ == "hr":
            # The frozen reference treats section rules as whitespace only.
            story.append(Spacer(1, 14.0))
        i += 1
    return story


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    register_fonts()
    styles = paragraph_styles()
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(md_path.read_text(encoding="utf-8"))
    title, subtitle, revision, source, start = _extract_front_matter(tokens)
    story = []
    story.extend(_cover_story(styles, title, subtitle, revision, source))
    story.extend(_toc_story(styles))
    story.extend(_body_story(tokens, start, styles))
    doc = CubeFitDocTemplate(str(pdf_path))
    doc.multiBuild(story)


HTML_CSS = """
:root { --navy:#12395a; --head:#174b70; --sub:#245b7c; --body:#202a33;
        --muted:#65717d; --table:#e8eff4; --rule:#b8c7d2; }
html { background:#f3f5f7; }
body { max-width:900px; margin:0 auto; padding:48px 64px 80px; background:white;
       color:var(--body);
       font-family:"FreeSans","Liberation Sans",Arial,sans-serif;
       font-size:16px; line-height:1.45; }
h1 { color:var(--navy); font-size:2.25rem; }
h2 { color:var(--head); margin-top:1.8em; }
h3 { color:var(--sub); margin-top:1.5em; }
h4 { color:var(--body); }
code, pre { font-family:"FreeMono","Liberation Mono",monospace; }
code { color:#435563; }
pre { padding:0.35rem 1.25rem; overflow:auto; background:#fbfcfd; }
table { width:100%; border-collapse:collapse; margin:1rem 0 1.4rem; }
th { background:var(--table); color:#536575; font-weight:400; text-align:left; }
th, td { padding:0.45rem 0.65rem; border-bottom:1px solid var(--rule); }
blockquote { color:#536575; border-left:3px solid var(--rule); margin-left:0;
             padding-left:1rem; }
a { color:var(--sub); }
"""


def build_html(md_path: Path, html_path: Path) -> None:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    rendered = md.render(md_path.read_text(encoding="utf-8"))
    title = "CubeFit Documentation"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{HTML_CSS}</style></head><body>{rendered}</body></html>"""
    html_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs/CubeFit.md")
    parser.add_argument("--pdf", default="docs/CubeFit.pdf")
    parser.add_argument("--html", default="docs/CubeFit.html")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not args.html_only:
        build_pdf(source, Path(args.pdf))
        print(f"wrote {args.pdf}")
    if not args.pdf_only:
        build_html(source, Path(args.html))
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
