#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>


#define max_function(a,b) ((a)>(b)?(a):(b))
struct __align__(8) MD//引入MD结构体，同时更新最大值和全局求和
{
float max_tmp;//负责存储最大值
float sum_tmp;//负责存储求和
};
__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.max_tmp > b.max_tmp);
    MD bigger = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.sum_tmp = bigger.sum_tmp + smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);
    res.max_tmp = bigger.max_tmp;
    return res;
}
double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec*1e-6);
}
template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
void _attention_kernel(float *Q, float *K, int N, int d, float *output, bool useshare){
    if(useshare){
        __shared__ float Qds[BLOCK_DIM][BLOCK_DIM];
        __shared__ float Kds[BLOCK_DIM][BLOCK_DIM];

        int row = threadIdx.x + blockIdx.x*blockDim.x;
        int col = threadIdx.y + blockIdx.y*blockDim.y;
        int phase_num = (N + BLOCK_DIM - 1)/BLOCK_DIM;


        float sum = 0;
        for(int ph = 0; ph < phase_num; ph++){
            if(ph*BLOCK_DIM + threadIdx.y < N && ph*BLOCK_DIM + threadIdx.x < N){
                Qds[threadIdx.x][threadIdx.y] = Q[row*N + ph*BLOCK_DIM + threadIdx.y];
                Kds[threadIdx.x][threadIdx.y] = K[(ph*BLOCK_DIM + threadIdx.x)*d + col];
            }
            else{
                Qds[threadIdx.x][threadIdx.y] = 0.0f;
                Kds[threadIdx.x][threadIdx.y] = 0.0f;
            }
            __syncthreads();
            for(int ind  = 0; ind < BLOCK_DIM; ind++) {
                sum += Qds[threadIdx.x][ind]*Kds[ind][threadIdx.y];

            }
            __syncthreads();
            output[row*d + col] = sum;
        }
    }
    else{
        int row = threadIdx.x + blockIdx.x*blockDim.x;
        int col = threadIdx.y + blockIdx.y*blockDim.y;

        float sum = 0;
        for(int index = 0; index < N; index++){
            sum += Q[row*N + index]*K[index*d + col];
        }
        output[row*d + col] = sum;

    }

}


void attention(float *cpu_Q, float *cpu_K, int N, int d, float *cpu_output, bool useshare){
    double st, ela;
    st = get_walltime();

    float *Q, *K, *output;
    cudaMalloc((void **) &Q, N*N*sizeof(float));
    cudaMalloc((void **) &K, N*d*sizeof(float));
    cudaMalloc((void **) &output, N*d*sizeof(float));
    cudaMemcpy(Q, cpu_Q, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K, cpu_K, N*d*sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    int BLOCK_DIM = 32;
    int num_block_y = ceil(max_function(N,d)/(double)(BLOCK_DIM));
    dim3 grid_dim(ceil(N/(double)(BLOCK_DIM)),num_block_y,1);
    dim3 block_dim(BLOCK_DIM,BLOCK_DIM,1);
    int share_mem = 2*BLOCK_DIM*BLOCK_DIM*sizeof(float);
    if(useshare){
        _attention_kernel<32><<<grid_dim, block_dim, share_mem>>>(Q, K, N, d, output, useshare);
    }
    else{
        _attention_kernel<32><<<grid_dim, block_dim>>>(Q, K, N, d, output, useshare);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_output, output, N*d*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(output);
    ela = get_walltime() - st;

    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);

}
int main() {
    int N = 2048;
    int d = 512;

    int size_Q = N*N;
    int size_K = N*d;
    int size_O = N*d;

    float *cpu_Q, *cpu_K, *cpu_output;
    cpu_Q = (float *)malloc(size_Q*sizeof(float));
    cpu_K = (float *)malloc(size_K*sizeof(float));
    cpu_output = (float *)malloc(size_O*sizeof(float));
    for(int i = 0; i < size_Q; i++){
        cpu_Q[i] = i%10;
    }
    for(int i = 0; i < size_K; i++){
        cpu_K[i] = i%10;
    }
    float tmp[size_O];
    float err = 0;
    bool useshare = false;
    attention(cpu_Q, cpu_K, N, d, cpu_output, useshare);
    for(int i = 0; i < size_O; i++){
        tmp[i] = cpu_output[i];
        //printf("out:%.4e\n",cpu_output[i]);
    }
    useshare = true;
    attention(cpu_Q, cpu_K, N, d, cpu_output, useshare);
    for(int i = 0; i < size_O; i++){
        //printf("out:%.4e\n",cpu_output[i]);
        err = max_function(err, fabs(cpu_output[i] - tmp[i]));
    }
    printf("err:%.4e\n",err);
    free(cpu_Q);
    free(cpu_K);
    free(cpu_output);


    return 0;
}
