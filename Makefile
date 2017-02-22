# PHYS 242 - Final project
CC = /usr/local/cuda-5.5/bin/nvcc
#CC = nvcc
FLAGS = -arch=sm_35
CFLAGS = -O3 $(FLAGS) -D_FORCE_INLINES

LIBRARIES = -lm

D_BAIDU = -DINITIAL_PRICE=207.330002 -DEXPECTED_RETURN=-0.000419601957843 -DEXPECTED_VOLUME=0.0214357830351
D_FACEBOOK = -DINITIAL_PRICE=83.300003 -DEXPECTED_RETURN=0.00135277973892 -DEXPECTED_VOLUME=0.00401862250701
D_YANDEX = -DINITIAL_PRICE=15.45 -DEXPECTED_RETURN=-7.9067189593e-05 -DEXPECTED_VOLUME=0.0307840480595

.PHONY: clean docs data

all: compile docs clean

# Compile the sources
compile: src/asian-options-main.cu src/asian-options.cu src/asian-options.h
	$(CC) $(CFLAGS) -o bin/asian-options-baidu $(D_BAIDU) src/asian-options-main.cu src/asian-options.cu $(LIBRARIES)
	$(CC) $(CFLAGS) -o bin/asian-options-facebook $(D_FACEBOOK) src/asian-options-main.cu src/asian-options.cu $(LIBRARIES)
	$(CC) $(CFLAGS) -o bin/asian-options-yandex $(D_YANDEX) src/asian-options-main.cu src/asian-options.cu $(LIBRARIES)

# Create simulation data
data: compile
	./bin/asian-options-baidu | gzip -9c > ./data/output/Baidu.gz
	./bin/asian-options-facebook | gzip -9c > ./data/output/Facebook.gz
	./bin/asian-options-yandex | gzip -9c > ./data/output/Yandex.gz

# Compile the docs
docs: docs/report/final-report.tex docs/report/references.bib docs/individual_reports/xiaojian.tex docs/individual_reports/l1kong.tex docs/individual_reports/jsidrach.tex src/estimate-parameters.ipynb src/simulations-cuda.ipynb src/toy-model-asian-options.ipynb
	cd "src/"; ipython nbconvert --to html *.ipynb; mv *.html "../docs/notebooks/";
	cd "docs/report/"; pdflatex -shell-escape "final-report.tex"; bibtex "final-report"; pdflatex -shell-escape "final-report.tex"; pdflatex -shell-escape "final-report.tex"; cd "../../"; mv "docs/report/final-report.pdf" "Report - Final Project.pdf"
	cd "docs/individual_reports/"; pdflatex -shell-escape "xiaojian.tex"; pdflatex -shell-escape "xiaojian.tex"; cd "../../"; mv "docs/individual_reports/xiaojian.pdf" "Individual Report - Xiaojian Jin.pdf"
	cd "docs/individual_reports/"; pdflatex -shell-escape "l1kong.tex"; pdflatex -shell-escape "l1kong.tex"; cd "../../"; mv "docs/individual_reports/l1kong.pdf" "Individual Report - Lingyi Kong.pdf"
	cd "docs/individual_reports/"; pdflatex -shell-escape "jsidrach.tex"; pdflatex -shell-escape "jsidrach.tex"; cd "../../"; mv "docs/individual_reports/jsidrach.pdf" "Individual Report - J. Sidrach.pdf"

# Delete binaries / auxiliary files
clean:
	cd "docs/report/"; rm -f *.o *.alg *.nav *.vrb latexmkrc *.snm *.acr *.acn *.lof *.log *.lot *.out *.bak *.toc *.xdy *.ist *.gls *.glo *.blg *.aux *.bbl *.glg *.glsdefs *.pyg  *.synctex.gz *Notes.bib
	cd "docs/individual_reports/"; rm -f *.o *.alg *.nav *.vrb latexmkrc *.snm *.acr *.acn *.lof *.log *.lot *.out *.bak *.toc *.xdy *.ist *.gls *.glo *.blg *.aux *.bbl *.glg *.glsdefs *.pyg  *.synctex.gz
	rm -f bin/*
