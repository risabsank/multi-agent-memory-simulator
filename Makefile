test:
	pytest tests/
local:
	pip install -e .
eval:
	python scenarios/eval_pipeline.py