all: ./lib/data.cuh ./lib/kernels.cuh ./lib/pam.cuh ./tests/test_build.cu
	nvcc -I./lib  -o test1 tests/test_build.cu --expt-relaxed-constexpr  -g