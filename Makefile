.PHONY: help pyversion install install-ml api console schema seed test doctor verify data seed-disputes disputes triage relink packet submit train metrics version lint fmt stripe trigger live-dispute dispute

PY := python

help:
	@echo "make install     core API deps (fast, always works)"
	@echo "make install-ml  model + agent + PDF deps (day 2 onward)"
	@echo "make api         run the API on :8000"
	@echo "make console     open the merchant console (the API serves it)"
	@echo "make test        run tests"
	@echo "make doctor      diagnose the live pipeline end to end"
	@echo "make verify      check every claim this project makes, against your data"
	@echo "make data        build the training table (synthetic)"
	@echo "make train       train the propensity model + metrics"
	@echo "make seed-disputes  generate disputes locally, load into Postgres"
	@echo "make triage      score the open worklist (API must be running)"
	@echo "make relink      link disputes that arrived before their order did"
	@echo "make packet ID=… draft and verify an evidence packet"
	@echo "make submit ID=… file it (MODE=dry_run|stage|submit, default dry_run)"
	@echo "make metrics     print the metrics report"
	@echo "make version     which build is this"
	@echo "make lint        ruff check"
	@echo "make stripe      forward Stripe test webhooks to localhost"
	@echo "make trigger     ask Stripe for a REAL test dispute (needs make stripe)"
	@echo "make live-dispute  the same, against an order we hold evidence for"

# Guards against the classic macOS trap: `python3` outside conda is 3.9,
# which cannot import enum.StrEnum and fails much further downstream.
pyversion:
	@$(PY) -c "import sys; v=sys.version_info; \
	print(f'Python {v.major}.{v.minor}.{v.micro} at {sys.executable}'); \
	sys.exit(0) if v >= (3,11) else sys.exit( \
	print('\n  ERROR: Python 3.11+ required.\n' \
	      '  Create the environment from a newer interpreter:\n\n' \
	      '    conda create -n cbg python=3.12 -y && conda activate cbg\n' \
	      '    # or\n' \
	      '    /opt/anaconda3/bin/python3 -m venv .venv && source .venv/bin/activate\n') or 1)"

install: pyversion
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r api/requirements.txt
	@echo "\nCore installed. Run: make api"

install-ml: pyversion
	$(PY) -m pip install -r api/requirements-ml.txt

api:
	cd api && uvicorn app.main:app --reload --port 8000

# The console is one file served by the API that feeds it. There is no build
# step and no second process: a reviewer clones this, runs `make api`, and has
# the screen. A toolchain between them and the demo is a toolchain that can
# break on their machine rather than mine.
console:
	@echo "http://localhost:8000/console"
	@command -v open >/dev/null && open http://localhost:8000/console || true

schema:
	cd api && $(PY) -m app.cli schema

seed:
	cd api && $(PY) -m app.cli seed

test:
	cd api && $(PY) -m pytest -q

doctor:
	$(PY) scripts/selftest.py

# `make test` proves the logic in isolation; this proves the behaviour against
# your database, your key and your running API — which is where every defect
# this project has actually had was found.
verify:
	$(PY) scripts/verify.py

data:
	$(PY) data/build_dataset.py --rows 120000

# Named apart from `make trigger` on purpose. These used to be `disputes` and
# `dispute`, one letter apart, doing entirely unrelated things: this one
# writes ten thousand rows to Postgres and never contacts Stripe.
# Safe to re-run: it preserves every processor-origin dispute and the
# order, evidence and packets behind it.
seed-disputes: data/processed/orders.parquet
	$(PY) data/synthesize_disputes.py --orders 10000 --load

disputes: seed-disputes

triage:
	@curl -s -X POST localhost:8000/disputes/triage | $(PY) -m json.tool

# Disputes recorded without an order, matched again now that the order exists.
relink:
	@curl -s -X POST localhost:8000/disputes/relink | $(PY) -m json.tool

# Filing. dry_run is the default on purpose: `make submit` should never be
# the command that sends a representment to a bank by accident.
MODE ?= dry_run

packet:
	@test -n "$(ID)" || (echo "usage: make packet ID=dsp_..." && exit 1)
	@curl -s -X POST "localhost:8000/disputes/$(ID)/packet" | $(PY) -m json.tool

submit:
	@test -n "$(ID)" || (echo "usage: make submit ID=dsp_... [MODE=stage]" && exit 1)
	@curl -s -X POST "localhost:8000/disputes/$(ID)/submit?mode=$(MODE)$(if $(filter submit,$(MODE)),&confirm=true,)" \
	  | $(PY) -m json.tool

train: data/processed/orders.parquet
	cd api && $(PY) -m app.ml.train
	@$(PY) scripts/sync_readme.py

# Depends on the generator, not just on existing. Otherwise a changed
# generator leaves a stale table on disk and `make train` silently retrains on
# the old data — which looks exactly like the new code having no effect.
data/processed/orders.parquet: data/build_dataset.py
	@echo "Training table is missing or older than the generator — rebuilding."
	@$(MAKE) data

metrics:
	@cat api/app/ml/artifacts/METRICS.md

version:
	@echo "commits: $$(git rev-list --count HEAD)"
	@echo "head:    $$(git log --oneline -1)"
	@$(PY) -c "import sys;print('python: ',sys.version.split()[0])"

lint:
	ruff check api/

fmt:
	ruff format api/

stripe:
	stripe listen --forward-to localhost:8000/webhooks/stripe

# Asks Stripe to create a real test dispute, which arrives over the webhook.
# `make stripe` must already be running in another tab to forward it.
#
# Stripe's dispute-generating test cards, if you would rather make one by hand:
#   4000000000000259  fraudulent
#   4000000000002685  product not received
#   4000000000001976  inquiry (retrieval phase)
trigger:
	@echo "Triggering a real test dispute -- 'make stripe' must be running elsewhere."
	stripe trigger charge.dispute.created

# `make trigger` disputes a charge with no order behind it, so retrieval finds
# nothing and the system correctly refuses to draft. This pays for a real order
# with the card Stripe disputes on purpose, so what arrives has evidence.
live-dispute:
	$(PY) scripts/live_dispute.py $(if $(ORDER),--order $(ORDER),) $(if $(CATEGORY),--category $(CATEGORY),)

dispute: trigger
