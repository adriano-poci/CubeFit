"""Frozen visual specification for the CubeFit reference manual."""

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm

PAGE_SIZE = A4
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
LEFT_MARGIN = 56.7
RIGHT_MARGIN = 56.7
TOP_MARGIN = 56.7
BOTTOM_MARGIN = 56.7

NAVY = HexColor("#12395A")
HEADING = HexColor("#174B70")
SUBHEADING = HexColor("#245B7C")
BODY = HexColor("#202A33")
MUTED = HexColor("#65717D")
MUTED_BLUE = HexColor("#52606D")
TABLE_HEAD = HexColor("#E8EFF4")
TABLE_RULE = HexColor("#D7E0E6")
TABLE_HEAD_RULE = HexColor("#9AABB8")
WHITE = HexColor("#FFFFFF")
BLACK = HexColor("#000000")

FONT_SANS = "CubeFitSans"
FONT_SANS_BOLD = "CubeFitSansBold"
FONT_MONO = "CubeFitMono"

HEADER_TEXT = "CubeFit - current architecture and solver manual"


def paragraph_styles():
    """Return the immutable ParagraphStyle set used by the PDF renderer."""
    return {
        "body": ParagraphStyle(
            "body", fontName=FONT_SANS, fontSize=9.4, leading=12.7,
            textColor=BODY, spaceAfter=6.0, alignment=TA_LEFT,
            allowWidows=0, allowOrphans=0,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=FONT_SANS_BOLD, fontSize=14.5, leading=17.4,
            textColor=HEADING, spaceBefore=13.5, spaceAfter=6.0,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=FONT_SANS_BOLD, fontSize=11.5, leading=14.0,
            textColor=SUBHEADING, spaceBefore=9.0, spaceAfter=4.0,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3", fontName=FONT_SANS_BOLD, fontSize=10.0, leading=12.3,
            textColor=BODY, spaceBefore=7.0, spaceAfter=3.5,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "code", fontName=FONT_MONO, fontSize=7.5, leading=10.0,
            textColor=BLACK, leftIndent=14.2, rightIndent=4.0,
            spaceBefore=1.5, spaceAfter=6.0,
        ),
        "quote": ParagraphStyle(
            "quote", fontName=FONT_SANS, fontSize=9.4, leading=12.7,
            textColor=MUTED_BLUE, leftIndent=12.0, rightIndent=8.0,
            borderWidth=0, spaceBefore=3.0, spaceAfter=6.0,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=FONT_SANS, fontSize=9.4, leading=12.7,
            textColor=BODY, leftIndent=14.0, firstLineIndent=-8.0,
            bulletIndent=2.0, spaceAfter=2.0,
        ),
        "number": ParagraphStyle(
            "number", fontName=FONT_SANS, fontSize=9.4, leading=12.7,
            textColor=BODY, leftIndent=18.0, firstLineIndent=-12.0,
            bulletIndent=1.0, spaceAfter=2.0,
        ),
        "table_head": ParagraphStyle(
            "table_head", fontName=FONT_SANS, fontSize=8.0, leading=9.5,
            textColor=MUTED_BLUE, spaceAfter=0,
        ),
        "table_body": ParagraphStyle(
            "table_body", fontName=FONT_SANS, fontSize=8.0, leading=9.5,
            textColor=MUTED_BLUE, spaceAfter=0,
        ),
        "toc_title": ParagraphStyle(
            "toc_title", fontName=FONT_SANS_BOLD, fontSize=19.0,
            leading=23.0, textColor=NAVY, spaceAfter=20.0,
        ),
        "toc_self": ParagraphStyle(
            "toc_self", fontName=FONT_SANS_BOLD, fontSize=10.0,
            leading=12.0, textColor=BLACK,
        ),
        "toc0": ParagraphStyle(
            "toc0", fontName=FONT_SANS, fontSize=9.0, leading=12.0,
            textColor=BLACK, leftIndent=10.0, firstLineIndent=0,
        ),
        "toc1": ParagraphStyle(
            "toc1", fontName=FONT_SANS, fontSize=8.3, leading=11.2,
            textColor=MUTED_BLUE, leftIndent=22.0, firstLineIndent=0,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", fontName=FONT_SANS_BOLD, fontSize=28.0,
            leading=33.0, textColor=NAVY, alignment=TA_CENTER,
            leftIndent=-6.0, rightIndent=6.0,
        ),
        "cover_text": ParagraphStyle(
            "cover_text", fontName=FONT_SANS, fontSize=14.0,
            leading=19.0, textColor=HexColor("#3E5568"),
            alignment=TA_CENTER, leftIndent=-6.0, rightIndent=6.0,
        ),
        "cover_code": ParagraphStyle(
            "cover_code", fontName=FONT_MONO, fontSize=7.5,
            leading=10.0, textColor=BLACK, alignment=TA_LEFT,
            leftIndent=14.2,
        ),
    }
