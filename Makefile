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

# Raw CUDA C++ baseline (Jetson/CUDA box only; self-checking, prints CSV)
CUDA_ARCH ?= sm_87
cuda-bench:
	mkdir -p out
	nvcc -O3 -arch=$(CUDA_ARCH) -o out/cuda_elementwise cuda/elementwise.cu
	./out/cuda_elementwise

plot:
	python -m bench.plot --results $(RESULTS)

.PHONY: test test-gpu lint smoke bench cuda-bench plot
