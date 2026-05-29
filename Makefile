# Regime-Aware GBWM — common tasks
.PHONY: install install-all test lint fmt app train evaluate backtest calibrate clean

install:          ## core install (numpy/pandas/yaml/matplotlib)
	pip install -e .

install-all:      ## full install (RL + data + app + dev)
	pip install -e ".[all,dev]"

test:             ## run the test suite (pytest, or the stdlib runner)
	pytest -q || python run_tests.py

lint:
	ruff check src tests && black --check src tests

fmt:
	ruff check --fix src tests && black src tests

app:              ## launch the Streamlit demo
	streamlit run app/streamlit_app.py

train:            ## solve/train all agents into the checkpoint registry
	gbwm train --agent all

evaluate:         ## Monte-Carlo comparison table
	gbwm evaluate

backtest:         ## single-path rollout + explanation
	gbwm backtest --agent regime_aware_g_learner --seed 7

calibrate:        ## estimate regimes from market data (use --offline for no network)
	gbwm calibrate

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ artifacts/runs
