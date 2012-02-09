docs = pragmatic_slides.pdf

%.pdf: %.tex
	pdflatex -shell-escape $<
	if fgrep "Rerun to" $*.log; then $(MAKE) --assume-new $< $@;fi
	if fgrep "There were undefined"  $*.log;\
	then bibtex $*; \
	$(MAKE) NOBIBTEX=true --assume-new $< $@;fi

pragmatic_slides.pdf: pragmatic_slides.py
	./pragmatic_slides.py
	pdflatex pragmatic_slides
	pdflatex pragmatic_slides

clean:
	rm -rf *.log  *.bbl *.blg *.div *.out *.vrb *.aux *.nav *.snm *.toc pragmatic_slides.tex $(docs)
