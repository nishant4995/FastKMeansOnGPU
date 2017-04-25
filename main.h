#ifndef _MAIN_H
#define _MAIN_H value

// #ifdef MNIST
	// #define NUM_POINTS 70000
	// #define DIMENSION 784
	// #define ROUNDED_DIMENSION 1024
	// #define NUM_CLUSTER 10
	// #define ROUNDED_CLUSTER 16
	// #define DATA "mnist"
	// #else
	// #ifdef THREE_D
	// #define NUM_POINTS 434874
	// #define DIMENSION 3
	// #define ROUNDED_DIMENSION 4
	// #define NUM_CLUSTER 5
	// #define ROUNDED_CLUSTER 8
	// #define DATA "3d_pro_5"
	// #else
	// #ifdef CIFAR
	// #define NUM_POINTS 6000
	// #define DIMENSION 3072
	// #define ROUNDED_DIMENSION 4096
	// #define NUM_CLUSTER 10
	// #define ROUNDED_CLUSTER 16
	// #define DATA "cifar"
	// #else
	// #ifdef BIRCH1
	// #define NUM_POINTS 100000
	// #define DIMENSION 2
	// #define ROUNDED_DIMENSION 2
	// #define NUM_CLUSTER 100
	// #define ROUNDED_CLUSTER 128
	// #define DATA "birch1"
	// #else
	// #ifdef BIRCH2
	// #define NUM_POINTS 100000
	// #define DIMENSION 2
	// #define ROUNDED_DIMENSION 2
	// #define NUM_CLUSTER 100
	// #define ROUNDED_CLUSTER 128
	// #define DATA "birch2"
	// #else
	// #ifdef BIRCH3
	// #define NUM_POINTS 100000
	// #define DIMENSION 2
	// #define ROUNDED_DIMENSION 2
	// #define NUM_CLUSTER 100
	// #define ROUNDED_CLUSTER 128
	// #define DATA "birch3"
	// #endif
	// #endif
	// #endif
	// #endif
	// #endif
	// #endif

#define NUM_POINTS 100000
#define DIMENSION 2
#define ROUNDED_DIMENSION 2
#define NUM_CLUSTER 100
#define ROUNDED_CLUSTER 128
#define DATA "birch1"

// Should be power of 2 and Max value is 1024 which is max number of threads_per_block on my GPU
#define SCAN_BLOCK_SIZE 32

// Implementation of step efficient scan is correct is WARP_SIZE for the GPU on which this code is running is 32.
// If warp size if different, then we need to change the implementation accordingly
// WARP_SIZE is the size of each partition/segement of the distance array for which scan operation is performed
#define WARP_SIZE 32  
// #define BLOCK_SIZE 32

#define ceil(a,b)  (a + b - 1)/b
#define roundUp(a,b)  ((a + b - 1)/b)*b

#define ROUNDED_NUM_POINTS roundUp(NUM_POINTS,WARP_SIZE)
#define NUM_PARTITIONS ceil(NUM_POINTS,WARP_SIZE)
#define ROUNDED_NUM_PARTITIONS roundUp( NUM_PARTITIONS , WARP_SIZE )

#define NUM_META_PARTITIONS ceil(NUM_PARTITIONS,WARP_SIZE)


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "sys/time.h"
#include "omp.h"
#include "float.h"
#include <helper_cuda.h>
#include "cuda_profiler_api.h"
#include <thrust/scan.h>

static inline float sd(float* a, int n);
static inline float mean(float* a, int n);
static inline float get_time_diff(struct timeval, struct timeval);

__global__ void kernelAddConstant(int *g_a, const int b);
int correctResult(int *data, const int n, const int b);
int main_check();
__global__ void comp_dist(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
__global__ void comp_dist_2(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
__global__ void comp_dist_glbl(float* dev_data,float* dev_distances,float* dev_partition_sums,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
__global__ void comp_dist_glbl_strided(float* dev_data,float* dev_distances,int centerIter,int numPoints,int dev_dimension, int rndedNumPoints);
__global__ void comp_dist_strided(float* dev_data,float* dev_distances,float* dev_centers,int centerIter,int numPoints,int dev_dimension,int rndedNumPoints);
__global__ void comp_dist_package(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads,float* dev_rnd);
__global__ void comp_dist_package_with_loop(float* dev_data,float* dev_distances_scanned,float* dev_partition_sums, float* dev_centers,int numPoints,float *dev_rnd);
__global__ void comp_dist_package_with_loop_original(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int numPoints,float *dev_rnd);



int sample_from_distribution (float* probabilities, int startIndex, int endIndex, float prob);
__global__ void sample_from_distribution_gpu(float* dev_partition_sums, float* dev_distances, int* dev_sampled_indices, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples);
__global__ void sample_from_distribution_gpu_copy(float* dev_partition_sums, float* dev_distances, float* dev_multiset, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples,float* dev_data);
__global__ void sample_from_distribution_gpu_strided(float* dev_distances, int* dev_sampled_indices, float* dev_rnd, int dev_NUM_POINTS, int dev_num_samples);
__global__ void sample_from_distribution_gpu_strided_copy(float* dev_distances, float* dev_multiset, float* dev_rnd, int dev_NUM_POINTS, int dev_num_samples, float* dev_data);


__global__ void exc_scan_2(float* inData,float* outData,int n);
__global__ void inc_scan_1(float* inData,float* outData,int n);
__global__ void inc_scan_1_block(float* inData,float* outData,int n,float* block_sums);
__global__ void inc_scan_1_rev(float* inData,float* outData,int n);
__global__ void inc_scan_1_add(float* block,float* block_sums,int n);
__global__ void inc_scan_1_block_SE(float* inData,float* outData,float* block_sums);

void testScan();
__global__ void copy_to_multiset(float* dev_multiset,float* dev_data,int* dev_sampled_indices);


float distance(float* p1, float* p2);
float* mean_heuristic(float* multiset,int multisetSize);
float* mean_heuristic_assign(float* multiset,int multisetSize,float* level_2_sample);
__global__ void mean_heuristic_assign_gpu(float* dev_multiset,int multisetSize, float* dev_l2_samples,float* dev_centers_temp);

float* d2_sample(float* data,float* centers,int numPts, int numSamples, int size);
float* d2_sample_2(float* data,float* centers,int numPts, int numSamples, int size, float* distances);
float* d2_sample_3(float* data,float* centers,int numPts, int size, float* distances);
void write_centers_to_file(float* centers);
#endif