.PHONY: check report

check:
	python 01_geometry/validate_geometry.py
	python -m compileall -q 01_geometry 03_aimd/scripts

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error Project_Report.tex
