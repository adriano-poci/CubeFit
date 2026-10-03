DOC=docs/CubeFit.md
HTML=docs/CubeFit.html
PDF=docs/CubeFit.pdf
REFERENCE=docs/CubeFit.reference.pdf
PYTHON?=python

.PHONY: all docs html pdf verify clean

all: docs

docs: html pdf

html:
	$(PYTHON) docs/build_docs.py --source $(DOC) --html $(HTML) --html-only

pdf:
	$(PYTHON) docs/build_docs.py --source $(DOC) --pdf $(PDF) --pdf-only

verify: pdf
	$(PYTHON) docs/verify_pdf_style.py $(PDF) --reference $(REFERENCE)

clean:
	rm -f $(HTML) $(PDF)
	rm -rf docs/_pdf_style_diff
