PY ?= python
MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
PORT ?= 8000
POLICY ?= lpm_aging
BLOCKS ?= 4096

.PHONY: install test test-gpu smoke serve sweep e1 e2 e3 kernel analyze clean

install:
	$(PY) -m pip install -e ".[bench,hf,dev]"

test:
	$(PY) -m pytest tests -q

test-gpu:
	$(PY) -m pytest tests -q -m gpu

smoke:
	$(PY) -m bench.run_matrix --grid bench/grids/smoke.yaml --out results/smoke
	$(PY) -m bench.analyze --results results/smoke --out analysis/smoke

serve:
	$(PY) -m mercury.server --model $(MODEL) --dtype float16 --policy $(POLICY) \
		--num-blocks $(BLOCKS) --port $(PORT)

sweep:
	$(PY) -m bench.run_matrix --grid bench/grids/e1.yaml --sweep 2 4 8 12 16 24

e1:
	$(PY) -m bench.run_matrix --grid bench/grids/e1.yaml --out results/e1

e2:
	$(PY) -m bench.run_matrix --grid bench/grids/e2.yaml --out results/e2

e3:
	$(PY) -m bench.run_matrix --grid bench/grids/e3.yaml --out results/e3

kernel:
	$(PY) -m bench.kernel_bench --out analysis --model $(MODEL)

analyze:
	$(PY) -m bench.analyze --results results/e1 --out analysis/e1

clean:
	rm -rf .pytest_cache **/__pycache__
