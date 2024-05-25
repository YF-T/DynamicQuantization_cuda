from torch.utils.cpp_extension import load
cuda_module = load(name="kv_cache_cuda",
                   sources=["K_cache.cu", "V_cache.cu", "GEMM.cu", "KV_cache_ops.cpp"],
                   extra_cuda_cflags=["-O3", "-ccbin=/usr/bin/gcc-10"],
                   verbose=True)