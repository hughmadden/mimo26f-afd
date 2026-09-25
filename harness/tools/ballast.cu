// GPU ballast for top-of-memory tests: holds N GiB (cudaMalloc + memset) until killed.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    const double gib = argc > 1 ? atof(argv[1]) : 1.0;
    const size_t bytes = size_t(gib * (1ull << 30));
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) { fprintf(stderr, "ballast: cudaMalloc %.2f GiB: %s\n", gib, cudaGetErrorString(e)); return 1; }
    cudaMemset(p, 0, bytes);
    cudaDeviceSynchronize();
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    printf("ballast: holding %.2f GiB; device free %.2f GiB\n", gib, f / double(1ull << 30));
    fflush(stdout);
    for (;;) pause();
}
