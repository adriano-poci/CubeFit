DOC=docs/CubeFit.md
HTML=docs/CubeFit.html
PDF=docs/CubeFit.pdf

all: html pdf

html:
	pandoc -s -f gfm -t html5 $(DOC) -o $(HTML) --metadata title="CubeFit" --toc

pdf:
	pandoc -s -f gfm $(DOC) -o $(PDF) --pdf-engine=xelatex \
	  -V geometry:margin=1in -V mainfont="Latin Modern Roman" \
	  -V monofont="Latin Modern Mono" --toc
