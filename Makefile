.PHONY: check geometry syntax

check: geometry syntax

geometry:
	python 01_geometry/validate_geometry.py

syntax:
	python -m compileall -q 01_geometry 02_pyscf_static 03_aimd/scripts
