# Interpreter-mode tests (CPU, run anywhere; the CI gate)
test:
	TRITON_INTERPRET=1 pytest -q

# Same suite against the real compiler on a CUDA box
test-gpu:
	TRITON_INTERPRET=0 pytest -q

lint:
	ruff check .
	pyright

# Step-0 validation on a new device
smoke:
	python scripts/jetson_smoke.py

# Full sweep on the Jetson, wrapped in the pause/restore session script
bench:
	scripts/bench_guss.sh python -m bench.run --kernel all

plot:
	python -m bench.plot --results $(RESULTS)

.PHONY: test test-gpu lint smoke bench plot
