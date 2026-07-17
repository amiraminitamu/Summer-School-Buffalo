.PHONY: check report

check:
	python src/00_validate_geometry.py
	python -m compileall -q src

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error Project_Report.tex
