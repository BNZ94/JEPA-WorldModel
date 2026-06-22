.ONESHELL:
.PHONY: help paper smoke clean-paper
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

paper: ## Compile paper/main.tex -> paper/main.pdf (tries tectonic / latexmk / pdflatex)
	@cd paper && \
	if command -v tectonic >/dev/null 2>&1; then \
		tectonic main.tex ; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode main.tex ; \
	elif command -v pdflatex >/dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex ; \
	else \
		echo "No LaTeX engine found. Install one of:" ; \
		echo "  tectonic   (single binary):  https://tectonic-typesetting.github.io" ; \
		echo "  texlive    (Debian/Ubuntu):  sudo apt-get install texlive-latex-extra" ; \
		exit 1 ; \
	fi

smoke: ## Run the microbiome CPU contract smoke tests (needs a torch env)
	python eb_jepa/datasets/microbiome/_smoke_glv.py
	python eb_jepa/datasets/microbiome/_smoke_data.py
	python -m examples.microbiome_jepa._smoke_plan
	python -m examples.microbiome_jepa._smoke_probe

clean-paper: ## Remove LaTeX build artifacts
	cd paper && rm -f main.aux main.log main.out main.toc main.fdb_latexmk main.fls
