/*
CSC630/730 Advanced Parallel Computing
GPU-Accelerated Monte Carlo - FIXED VERSION
Authors: Tomas Nader and Isaac Larbi

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARP_SIZE 32
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    double price;
    double std_error;
    double conf_interval_lower;
    double conf_interval_upper;
    double execution_time;
} MonteCarloResult;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

//-------------------------------------------------------------------------
// Warp shuffle reduction
//-------------------------------------------------------------------------
__device__ double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//-------------------------------------------------------------------------
// Block reduction with warp shuffles
//-------------------------------------------------------------------------
__device__ double block_reduce_sum(double val) {
    __shared__ double shared[WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//-------------------------------------------------------------------------
// GPU Kernel - Monte Carlo Simulation
//-------------------------------------------------------------------------
__global__ void monte_carlo_kernel(
    double S0, double K, double r, double sigma, double T,
    int n_steps, char option_type, int n_paths,
    double *payoffs, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double payoff = 0.0;
    
    if (idx < n_paths) {
        // Initialize cuRAND with unique sequence per thread
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        double dt = T / n_steps;
        double drift = (r - 0.5 * sigma * sigma) * dt;
        double vol = sigma * sqrt(dt);
        
        double S = S0;
        for (int step = 0; step < n_steps; step++) {
            double Z = curand_normal_double(&state);
            S *= exp(drift + vol * Z);
        }
        
        if (option_type == 'C' || option_type == 'c') {
            payoff = fmax(S - K, 0.0);
        } else {
            payoff = fmax(K - S, 0.0);
        }
        
        // Store directly to global memory
        payoffs[idx] = payoff;
    }
}

//-------------------------------------------------------------------------
// Separate reduction kernel for better performance
//-------------------------------------------------------------------------
__global__ void reduction_kernel(double *input, double *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double val = (idx < n) ? input[idx] : 0.0;
    double val_sq = val * val;
    
    double sum = block_reduce_sum(val);
    double sum_sq = block_reduce_sum(val_sq);
    
    if (threadIdx.x == 0) {
        output[blockIdx.x * 2] = sum;
        output[blockIdx.x * 2 + 1] = sum_sq;
    }
}

//-------------------------------------------------------------------------
// GPU Monte Carlo
//-------------------------------------------------------------------------
MonteCarloResult monte_carlo_gpu(
    double S0, double K, double r, double sigma, double T,
    int n_paths, int n_steps, char option_type)
{
    MonteCarloResult result;
    double start_time = get_time();
    
    int num_blocks = (n_paths + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate memory for all payoffs
    double *d_payoffs;
    CUDA_CHECK(cudaMalloc(&d_payoffs, n_paths * sizeof(double)));
    
    unsigned long long seed = (unsigned long long)time(NULL);
    
    // Launch simulation kernel
    monte_carlo_kernel<<<num_blocks, BLOCK_SIZE>>>(
        S0, K, r, sigma, T, n_steps, option_type, n_paths, d_payoffs, seed);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Allocate for reduction results
    double *d_reduction;
    CUDA_CHECK(cudaMalloc(&d_reduction, num_blocks * 2 * sizeof(double)));
    
    // Launch reduction kernel
    reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_payoffs, d_reduction, n_paths);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy reduction results to host
    double *h_reduction = (double*)malloc(num_blocks * 2 * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_reduction, d_reduction, 
                          num_blocks * 2 * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Final reduction on CPU
    double total_payoff = 0.0;
    double total_payoff_sq = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total_payoff += h_reduction[i * 2];
        total_payoff_sq += h_reduction[i * 2 + 1];
    }
    
    // Calculate statistics
    double mean_payoff = total_payoff / n_paths;
    double variance = (total_payoff_sq - (total_payoff * total_payoff) / n_paths) / (n_paths - 1);
    double std_error = sqrt(variance / n_paths);
    
    double discount_factor = exp(-r * T);
    result.price = discount_factor * mean_payoff;
    result.std_error = discount_factor * std_error;
    result.conf_interval_lower = result.price - 1.96 * result.std_error;
    result.conf_interval_upper = result.price + 1.96 * result.std_error;
    result.execution_time = get_time() - start_time;
    
    // Cleanup
    free(h_reduction);
    cudaFree(d_payoffs);
    cudaFree(d_reduction);
    
    return result;
}

//-------------------------------------------------------------------------
// CPU Implementation
//-------------------------------------------------------------------------
double generate_random_normal() {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

MonteCarloResult monte_carlo_cpu(
    double S0, double K, double r, double sigma, double T,
    int n_paths, int n_steps, char option_type)
{
    MonteCarloResult result;
    double start_time = get_time();
    
    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * sqrt(dt);
    
    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;
    
    for (int i = 0; i < n_paths; i++) {
        double S = S0;
        
        for (int step = 0; step < n_steps; step++) {
            double Z = generate_random_normal();
            S *= exp(drift + vol * Z);
        }
        
        double payoff;
        if (option_type == 'C' || option_type == 'c') {
            payoff = fmax(S - K, 0.0);
        } else {
            payoff = fmax(K - S, 0.0);
        }
        
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }
    
    double mean_payoff = sum_payoff / n_paths;
    double variance = (sum_payoff_sq - (sum_payoff * sum_payoff) / n_paths) / (n_paths - 1);
    double std_error = sqrt(variance / n_paths);
    
    double discount_factor = exp(-r * T);
    result.price = discount_factor * mean_payoff;
    result.std_error = discount_factor * std_error;
    result.conf_interval_lower = result.price - 1.96 * result.std_error;
    result.conf_interval_upper = result.price + 1.96 * result.std_error;
    result.execution_time = get_time() - start_time;
    
    return result;
}

//-------------------------------------------------------------------------
// Black-Scholes
//-------------------------------------------------------------------------
double cumulative_normal(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

double black_scholes_price(double S0, double K, double r, double sigma, 
                          double T, char option_type) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    if (option_type == 'C' || option_type == 'c') {
        return S0 * cumulative_normal(d1) - K * exp(-r * T) * cumulative_normal(d2);
    } else {
        return K * exp(-r * T) * cumulative_normal(-d2) - S0 * cumulative_normal(-d1);
    }
}

//-------------------------------------------------------------------------
// Main
//-------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    double S0 = 100.0;
    double K = 105.0;
    double r = 0.03;
    double sigma = 0.2;
    double T = 1.0;
    int n_paths = 1000000;
    int n_steps = 252;
    char option_type = 'P';
    
    if (argc > 1) n_paths = atoi(argv[1]);
    if (argc > 2) n_steps = atoi(argv[2]);
    if (argc > 3) option_type = argv[3][0];
    
    printf("================================================================\n");
    printf("GPU-Accelerated Monte Carlo Option Pricing - FIXED VERSION\n");
    printf("CSC630/730 Advanced Parallel Computing\n");
    printf("Authors: Tomas Nader and Isaac Larbi\n");
    printf("================================================================\n\n");
    
    printf("Input Parameters:\n");
    printf("  Stock Price (S0):      $%.2f\n", S0);
    printf("  Strike Price (K):      $%.2f\n", K);
    printf("  Risk-free Rate (r):    %.2f%%\n", r * 100);
    printf("  Volatility (sigma):    %.2f%%\n", sigma * 100);
    printf("  Time to Maturity (T):  %.2f years\n", T);
    printf("  Option Type:           %s\n", option_type == 'C' ? "Call" : "Put");
    printf("  Number of Paths:       %d\n", n_paths);
    printf("  Time Steps per Path:   %d\n\n", n_steps);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU Device: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    double analytical = black_scholes_price(S0, K, r, sigma, T, option_type);
    
    printf("Running Serial CPU Implementation...\n");
    MonteCarloResult cpu_result = monte_carlo_cpu(S0, K, r, sigma, T, 
                                                    n_paths, n_steps, option_type);
    printf("  Completed in %.4f seconds\n\n", cpu_result.execution_time);
    
    printf("Running GPU Implementation...\n");
    MonteCarloResult gpu_result = monte_carlo_gpu(S0, K, r, sigma, T,
                                                   n_paths, n_steps, option_type);
    printf("  Completed in %.4f seconds\n\n", gpu_result.execution_time);
    
    printf("================================================================\n");
    printf("RESULTS COMPARISON\n");
    printf("================================================================\n\n");
    
    printf("Black-Scholes Analytical:  $%.6f\n\n", analytical);
    
    printf("Serial CPU Monte Carlo:\n");
    printf("  Price:       $%.6f\n", cpu_result.price);
    printf("  Std Error:   $%.6f\n", cpu_result.std_error);
    printf("  95%% CI:      [$%.6f, $%.6f]\n", 
           cpu_result.conf_interval_lower, cpu_result.conf_interval_upper);
    printf("  Time:        %.4f seconds\n", cpu_result.execution_time);
    printf("  Error vs BS: %.4f%%\n", 
           100.0 * fabs(cpu_result.price - analytical) / analytical);
    printf("  Throughput:  %.2f M paths/sec\n\n",
           n_paths / cpu_result.execution_time / 1e6);
    
    printf("GPU Monte Carlo (Warp Shuffle):\n");
    printf("  Price:       $%.6f\n", gpu_result.price);
    printf("  Std Error:   $%.6f\n", gpu_result.std_error);
    printf("  95%% CI:      [$%.6f, $%.6f]\n",
           gpu_result.conf_interval_lower, gpu_result.conf_interval_upper);
    printf("  Time:        %.4f seconds\n", gpu_result.execution_time);
    printf("  Error vs BS: %.4f%%\n", 
           100.0 * fabs(gpu_result.price - analytical) / analytical);
    printf("  Throughput:  %.2f M paths/sec\n\n",
           n_paths / gpu_result.execution_time / 1e6);
    
    printf("================================================================\n");
    printf("PERFORMANCE\n");
    printf("================================================================\n");
    double speedup = cpu_result.execution_time / gpu_result.execution_time;
    printf("  Speedup: %.2fx\n", speedup);
    
    if (analytical >= gpu_result.conf_interval_lower && 
        analytical <= gpu_result.conf_interval_upper) {
        printf("  ✓ Analytical price within 95%% CI\n");
    } else {
        printf("  ✗ Analytical price outside 95%% CI\n");
    }
    
    printf("================================================================\n");
    
    return 0;
}