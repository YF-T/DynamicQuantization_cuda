#include <torch/extension.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

// A(bsz, n_heads, l, m) * B(bsz, n_heads, m, n) = C(bsz, n_heads, l, n)
// grid(bsz, n_head, l)
// block(n // 2, 1, 1)
__global__ void gemm_kernel(half *A,
                     half *B,
                     half *C,
                     const int stride_batch_A,
                     const int stride_n_heads_A,
                     const int stride_l_A,
                     const int stride_m_A,
                     const int stride_batch_B,
                     const int stride_n_heads_B,
                     const int stride_m_B,
                     const int stride_n_B,
                     const int stride_batch_C,
                     const int stride_n_heads_C,
                     const int stride_l_C,
                     const int stride_n_C,
                     const int m,
                     const int n) {
    half2 result = __half2half2(__short_as_half(0x0000));
    for(int i = 0; i < m; i += 1) {
//        half data_a = A[stride_batch_A * blockIdx.x + stride_n_heads_A * blockIdx.y + stride_l_A * blockIdx.z + stride_m_A * i];
//        half data_b_1 = B[stride_batch_B * blockIdx.x + stride_n_heads_B * blockIdx.y + stride_m_B * i + stride_n_B * threadIdx.x * 2];
//        half data_b_2 = B[stride_batch_B * blockIdx.x + stride_n_heads_B * blockIdx.y + stride_m_B * i + stride_n_B * (threadIdx.x * 2 + 1)];
        half data_a = __ldg(A + stride_batch_A * blockIdx.x + stride_n_heads_A * blockIdx.y + stride_l_A * blockIdx.z + stride_m_A * i);
        half data_b_1 = __ldg(B + stride_batch_B * blockIdx.x + stride_n_heads_B * blockIdx.y + stride_m_B * i + stride_n_B * threadIdx.x * 2);
        half data_b_2 = __ldg(B + stride_batch_B * blockIdx.x + stride_n_heads_B * blockIdx.y + stride_m_B * i + stride_n_B * (threadIdx.x * 2 + 1));
        half2 a = __half2half2(data_a);
        half2 b = __halves2half2(data_b_1, data_b_2);
        result = __hfma2(a, b, result);
    }
//    C[stride_batch_C * blockIdx.x + stride_n_heads_C * blockIdx.y + stride_l_C * blockIdx.z + stride_n_C * threadIdx.x * 2] = __low2half(result);
//    C[stride_batch_C * blockIdx.x + stride_n_heads_C * blockIdx.y + stride_l_C * blockIdx.z + stride_n_C * (threadIdx.x * 2 + 1)] = __high2half(result);
}

torch::Tensor gemm_cuda(torch::Tensor A,
                        torch::Tensor B,
                        const int stride_batch_A,
                        const int stride_n_heads_A,
                        const int stride_l_A,
                        const int stride_m_A,
                        const int stride_batch_B,
                        const int stride_n_heads_B,
                        const int stride_m_B,
                        const int stride_n_B,
                        const int l,
                        const int m,
                        const int n) {
    auto options = torch::TensorOptions().dtype(torch::kByte).device(A.device());
    torch::Tensor C = torch::empty({A.size(0), A.size(1), l, n * 2}, options);
    half* A_ptr = (half*)A.data_ptr<unsigned char>();
    half* B_ptr = (half*)B.data_ptr<unsigned char>();
    half* C_ptr = (half*)C.data_ptr<unsigned char>();
    int stride_batch_C = C.stride(0);
    int stride_n_heads_C = C.stride(1);
    int stride_l_C = C.stride(2);
    int stride_n_C = C.stride(3);
    dim3 grid(A.size(0), A.size(1), l);
    dim3 block(n >> 1, 1, 1);
    gemm_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr,
                                 stride_batch_A, stride_n_heads_A, stride_l_A, stride_m_A,
                                 stride_batch_B, stride_n_heads_B, stride_m_B, stride_n_B,
                                 stride_batch_C, stride_n_heads_C, stride_l_C, stride_n_C,
                                 m, n);
    return C;
}