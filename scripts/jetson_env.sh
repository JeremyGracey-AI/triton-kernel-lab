# Source me on a Jetson before running anything that imports torch.
#
# Newer Jetson torch wheels (e.g. 2.11.0 from pypi.jetson-ai-lab.io/jp6/cu126)
# link NVIDIA libraries that ship as pip packages rather than with JetPack —
# libcudss is the one that bites first (pip install nvidia-cudss-cu12). pip
# puts them under site-packages/nvidia/<name>/lib, which is not on the wheel's
# RPATH, so export those dirs.
#
# WARNING — this glob exports EVERY pip-installed NVIDIA lib dir, so the venv
# must contain ONLY companion libs JetPack lacks. SBSA/x86-lineage builds of
# libraries JetPack already provides (cuBLAS, cuDNN, NCCL, ...) — e.g. pulled
# in as deps of a PyPI torch you later replaced — will shadow the Jetson
# builds through this path and fail at first use: cuBLAS dies with
# CUBLAS_STATUS_ALLOC_FAILED at cublasCreate, and only for the first op that
# touches it (matmul), long after import and elementwise kernels succeeded.
# Keep the venv's nvidia-* set minimal (`pip list | grep nvidia`).
_venv="${VIRTUAL_ENV:-.venv}"
_libs=$(ls -d "$_venv"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_libs}${LD_LIBRARY_PATH:-}"
