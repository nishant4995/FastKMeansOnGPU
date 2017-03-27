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

static inline float sd(double* a, int n);
static inline float mean(double* a, int n);
static inline double get_time_diff(struct timeval, struct timeval);

__global__ void kernelAddConstant(int *g_a, const int b);
int correctResult(int *data, const int n, const int b);
int main_check();
__global__ void comp_dist(double* dev_data,double* dev_distances,double* dev_partition_sums, double* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
__global__ void comp_dist_2(double* dev_data,double* dev_distances,double* dev_partition_sums, double* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
__global__ void comp_dist_glbl(double* dev_data,double* dev_distances,double* dev_partition_sums,int centerIter,int numPoints,int dev_dimension,int numGPUThreads);
int sample_from_distribution (double* probabilities, int startIndex, int endIndex, double prob);
__global__ void sample_from_distribution_gpu(double* dev_partition_sums, double* dev_distances, int* dev_sampled_indices, double* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_N);
double distance(double* p1, double* p2);
double* mean_heuristic(double* multiset,int multisetSize);
double* d2_sample(double* data,double* centers,int numPts, int numSamples, int size);
double* d2_sample_2(double* data,double* centers,int numPts, int numSamples, int size, double* distances);
void write_centers_to_file(double* centers);
#endif