# Building the frozen-style CubeFit documentation

`CubeFit.md` is the canonical documentation source. Do not edit the generated
PDF or HTML by hand.

The PDF renderer is intentionally local to this repository. Its typography,
page geometry, colours, heading ladder, tables, code blocks, title page,
contents pages, header, and footer are frozen in `pdf_style.py`.

## Requirements

On Debian/Ubuntu-like systems:

```bash
sudo apt install fonts-freefont-ttf fontconfig poppler-utils
python -m pip install reportlab markdown-it-py
```

The renderer does **not** bundle or redistribute fonts. It resolves the GNU
FreeFont installation on the local system. The reference PDF uses FreeSans,
FreeSans Bold, and FreeMono.

## Build

From the repository root:

```bash
make docs
```

or directly:

```bash
python docs/build_docs.py
```

This reads:

```text
docs/CubeFit.md
```

and writes:

```text
docs/CubeFit.pdf
docs/CubeFit.html
```

## Update workflow

1. Edit only `docs/CubeFit.md` for documentation content.
2. Run `make docs`.
3. Open the PDF and check changed sections.
4. For renderer/style changes, run `make docs-verify` and compare against the
   frozen `docs/CubeFit.reference.pdf`.
5. Do not replace the reference PDF merely because content changed. It is a
   visual-style reference, not a content baseline.

Content changes will naturally change pagination. The renderer preserves the
same style system; it does not force old line/page breaks onto new text.

## When to edit `pdf_style.py`

Normally: never. Edit it only when intentionally changing the documentation
visual identity. A scientific or algorithmic documentation update should only
change `CubeFit.md`.
