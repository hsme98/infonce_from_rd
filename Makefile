PYTHON ?= python
PROFILE ?= quick
DEVICE ?= auto
OUTPUT_ROOT ?= results/reproduced
DATA_ROOT ?= data/cifar100

.PHONY: install install-tabular test smoke quick tabular cifar all ablation figures validate clean

install:
	$(PYTHON) -m pip install --no-build-isolation -e ".[cifar,dev]"

install-tabular:
	$(PYTHON) -m pip install --no-build-isolation -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_test.py --output-dir results/smoke/cifar_synthetic

quick:
	$(PYTHON) scripts/run_all.py --profile quick --output-root results/quick --device $(DEVICE)

tabular:
	$(PYTHON) scripts/run_tabular.py --profile $(PROFILE) --output-root $(OUTPUT_ROOT)/tabular

cifar:
	$(PYTHON) scripts/run_cifar.py --config configs/$(PROFILE)/cifar.json --output-root $(OUTPUT_ROOT)/cifar --data-root $(DATA_ROOT) --device $(DEVICE)

all:
	$(PYTHON) scripts/run_all.py --profile $(PROFILE) --output-root $(OUTPUT_ROOT) --data-root $(DATA_ROOT) --device $(DEVICE)

ablation:
	$(PYTHON) scripts/run_optimizer_ablation.py --config configs/$(PROFILE)/optimizer_ablation.json --output-root $(OUTPUT_ROOT)/optimizer_ablation

figures:
	$(PYTHON) scripts/rebuild_bundled_figures.py

validate:
	$(PYTHON) scripts/validate_results.py --root $(OUTPUT_ROOT)

clean:
	rm -rf results/reproduced results/quick results/pilot results/smoke
