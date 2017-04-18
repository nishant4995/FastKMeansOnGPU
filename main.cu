#include "main.h"

// g++ -D BIRCH1 -g mainOMP.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch1 -lm

// This will work for NUM_POINTS = 1024*1024 as we are using BLOCK_SIZE = 32

// random_device rd;
// mt19937 gen(rd());
// unsigned int max_val = gen.max();
int numThreads = 1;
__constant__ float dev_centers_global[NUM_CLUSTER*DIMENSION]; // For using constant memory
int main(int argc, char const *argv[])
{
	// Due to limited precision of float, getting back actual distance array from scanned version of it causes errors,
	// This errors apparently makes things worse when we start using BLOCK_SIZE = 64 and more.
	// This is an instance of underflow which can be observed when first center is fixed to be data[2]
	// float a1 = 119756262526522409091072.0;   
	// float a2 = 4516751081472.0;
	// float a3 = a1+a2;
	// float a4 = a3 - a1;
	// printf("a3\t%f\n",a3 );
	// printf("a4\t%f\n",a4 );
	// exit(0);

	// testScan();
	// Currently no argument processing logic, will always run birch1 for 2 times with N=10k
	srand(time(NULL));
		int numRuns,method;
		int N = 0;

		// for k-means parallel
		int rounds = 5;
		float oversampling = NUM_CLUSTER;
		oversampling = 2*oversampling;

		char dataFileName[100];
		char mode[100];
		char baseLogFile[100];
		char resultFile[100];

		sprintf(dataFileName,"%s%s","../data/",DATA);
		sprintf(mode,"%s",argv[1]);
		sprintf(baseLogFile,"../logs/%s/%s_",DATA,mode);
		numRuns 	= atoi(argv[2]);
		method 		= -1;
		numThreads 	= 1;
		if(getenv("OMP_NUM_THREADS") != NULL)
		{
			numThreads = atoi(getenv("OMP_NUM_THREADS"));
			printf("numThreads as gotten from env::%d\n",numThreads);
			if(numThreads == 0)
			{
				numThreads = 1;
			}
		}
		else
		{
			printf("numThreads as gotten by default::%d\n",numThreads);
		}

		if(strcmp(mode,"random")==0)
		{
			method = 0;
		}
		if(strcmp(mode,"kmeans++")==0)
		{
			method = 1;
		}
		if(strcmp(mode,"d2-seeding")==0)
		{
			method = 2;
			N = floor(NUM_CLUSTER * atof(argv[3]));
			sprintf(baseLogFile,"%sN=%sk_",baseLogFile,argv[3]);
		}
		if(strcmp(mode,"kmeans-par")==0)
		{
			method 			= 3;
			oversampling 	= NUM_CLUSTER * atof(argv[3]);
			rounds 			= atoi(argv[4]);
			sprintf(baseLogFile,"%sl=%sk_r=%d_",baseLogFile,argv[3],rounds);
		}

		// base log file name for individual runs
		sprintf(baseLogFile,"%sthreads=%d_",baseLogFile,numThreads);

		// log file for combined results. Mean and standard deviations
		sprintf(resultFile,"%sresult.txt",baseLogFile);
		sprintf(baseLogFile,"%srunNo=",baseLogFile);

		struct timeval start,end;
		// collect stats about all relevant parameters
		float initTime[numRuns];
		float iterTime[numRuns];
		float totalTime[numRuns];
		float initCost[numRuns];
		float finalCost[numRuns];
		float numIter[numRuns];

		// read the data into a vector of "vector"

		float* data;
		FILE* reader;
		int i = 0,j = 0;
		data 	= (float*)malloc(NUM_POINTS*DIMENSION*sizeof(float));
		reader 	= fopen(dataFileName,"r");

		gettimeofday(&start,NULL);

		while(i < NUM_POINTS)
		{
			j = 0;
			while(j < DIMENSION)
			{
				int k =	fscanf(reader,"\t%f",&(data[i*DIMENSION + j]));
				j++;
			}
			i++;
		}
		gettimeofday(&end,NULL);
		// printf("Time take to read data::%f::%f\n",get_time_diff(start,end),get_time_diff(start,end)/NUM_CLUSTER); 


	gettimeofday(&start,NULL);
	// Copy data onto device memory
	float* dev_data;
	cudaMalloc((void**)&dev_data,DIMENSION*NUM_POINTS*sizeof(float));
	cudaMemcpy(dev_data,data,DIMENSION*NUM_POINTS*sizeof(float),cudaMemcpyHostToDevice);
	gettimeofday(&end,NULL);
	// printf("Time take to copy data::%f::%f\n",get_time_diff(start,end),get_time_diff(start,end)/NUM_CLUSTER); 

	FILE* logger;
	int runNum;
	for(runNum = 0; runNum < numRuns ; runNum++)
	{
		float samplingTime_1[NUM_CLUSTER];
		float samplingTime_2[NUM_CLUSTER];
		printf("Running runNum::%d\n",runNum );
		gettimeofday(&start,NULL);

		int numBlocks 			= 8;
		int numThreadsPerBlock 	= 1024;
		int numSampleBlocks 	= 128;
		int numSampleTperB 		= 32;
		int numGPUThreads 		= numBlocks*numThreadsPerBlock;

		// If using strided memory access pattern
		int META_BLOCK_SUM_SIZE = NUM_META_PARTITIONS - 1; 
		int ctr = 0;
		// Code to roundUp to next power of 2 if not already
		while(META_BLOCK_SUM_SIZE >= 1 )
		{
			META_BLOCK_SUM_SIZE /= 2;
			ctr += 1;
		}
		META_BLOCK_SUM_SIZE = 1<<ctr;

		int ROUNDED_N = N - 1; 
		ctr = 0;
		// Code to roundUp to next power of 2 if not already
		while(ROUNDED_N >= 1 )
		{
			ROUNDED_N /= 2;
			ctr += 1;
		}
		ROUNDED_N = 1<<ctr;

		float* distances 	= (float*)malloc(NUM_POINTS*sizeof(float));
		float* centers 		= (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
		float* rnd 			= (float*)malloc(2*N*sizeof(float));
		float* multiset    	= (float*)malloc(N*DIMENSION*sizeof(float));
		float*  l2_centers 	= (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
		int*   sampled_indices 	= (int*)malloc(N*sizeof(int));
		int*  l2_center_indices = (int*)malloc(NUM_CLUSTER*sizeof(int));

		float* partition_sums 	= (float*)malloc(  numGPUThreads*sizeof(float)); // For blocked access pattern for distance array

		float* dev_distances;
		float* dev_distances_scanned;
		float* dev_partition_sums;
		float* dev_meta_block_sums;
		float* dev_rnd;
		int*   dev_sampled_indices;

		float* dev_multiset;
		float* dev_multiset_dist;
		float* dev_multiset_dist_scanned;
		float* dev_multiset_dist_partition_sums;
		float* dev_l2_centers;
		int*   dev_l2_center_indices;

		float* dev_centers_temp;
		cudaMalloc((void**)&dev_centers_temp,DIMENSION*sizeof(float));
					
		
		gettimeofday(&start,NULL);
		// float* dev_centers; // Needed when not using constant memory for centers
		printf("ROUNDED_NUM_POINTS::%d\n",ROUNDED_NUM_POINTS );
		checkCudaErrors(cudaMalloc((void**)&dev_distances,ROUNDED_NUM_POINTS*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_distances_scanned,ROUNDED_NUM_POINTS*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,numGPUThreads*sizeof(float))); // For blocked access pattern for distance array
		// checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,ROUNDED_NUM_PARTITIONS*sizeof(float))); // For strided access pattern for distance array
		checkCudaErrors(cudaMalloc((void**)&dev_meta_block_sums,META_BLOCK_SUM_SIZE*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_rnd,2*N*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_sampled_indices,N*sizeof(float)));

		checkCudaErrors(cudaMalloc((void**)&dev_multiset,N*DIMENSION*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_multiset_dist,ROUNDED_N*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_multiset_dist_scanned,ROUNDED_N*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_multiset_dist_partition_sums,ROUNDED_N*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_l2_centers,NUM_CLUSTER*DIMENSION*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_l2_center_indices,NUM_CLUSTER*sizeof(int)));

		// checkCudaErrors(cudaMalloc((void**)&dev_centers,NUM_CLUSTER*DIMENSION*sizeof(float))); // No need when using constant memory
		gettimeofday(&end,NULL);

		// printf("Time take to init array of GPU::%f::%f\n",get_time_diff(start,end),get_time_diff(start,end)/NUM_CLUSTER); 

		float tempTime;
		cudaEvent_t start_gpu,stop_gpu;
		cudaEventCreate(&start_gpu);
		cudaEventCreate(&stop_gpu);
		cudaStream_t stream1, stream2, stream3;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		cudaStreamCreate(&stream3);
		// initialize the initial centers
		if(method == 2) // d2-seeding
		{  
			// ---------------------- GPU-Based Implementation Start ------------------------------------
			cudaProfilerStart();
			// First choosing the first point uniformly at random, no need to sample N points and all here
			int tempPointIndex 	= (((float) rand())/RAND_MAX)*NUM_POINTS;
			memcpy(centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float));
			checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float),0,cudaMemcpyHostToDevice));
			// checkCudaErrors(cudaMemcpy(dev_centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float),cudaMemcpyHostToDevice));
			float compDistTime = 0, makeCumulativeTime = 0, samplingTime = 0, meanHeuristicTime = 0;
			for(i = 1; i < NUM_CLUSTER; i++)
			{
				// -----------------------GPU-Based implementation of D2-Sample ends------------------------------

					// printf("Iterations number %d\n",i);
					// struct timeval sample_start,sample_end;
					cudaEventRecord(start_gpu,0);
					// gettimeofday(&sample_start,NULL);
					for(j = 0; j < N; ++j)
					{
						rnd[2*j] 	= ((float) rand())/RAND_MAX;
						rnd[2*j+1] 	= ((float) rand())/RAND_MAX;
					}
					checkCudaErrors(cudaMemcpy(dev_rnd,rnd,2*N*sizeof(float),cudaMemcpyHostToDevice));// Can be overlapped with computation
					// comp_dist<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, dev_centers, i, NUM_POINTS, DIMENSION, numGPUThreads);
					
					// For blocked access pattern
						// comp_dist_glbl<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, i, NUM_POINTS, DIMENSION, numGPUThreads);
						// checkCudaErrors(cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(float),cudaMemcpyDeviceToHost));	
						// checkCudaErrors(cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(float),cudaMemcpyDeviceToHost));	
						// for (j = 1; j < numGPUThreads; ++j) // Need to do this scan operation on GPU only, but testing things first
						// {
						// 	partition_sums[j] += partition_sums[j-1];
						// }
						// checkCudaErrors(cudaMemcpy(dev_partition_sums,partition_sums,numGPUThreads*sizeof(float),cudaMemcpyHostToDevice));
		
						// int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;
						// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances, dev_sampled_indices, dev_rnd, per_thread, NUM_POINTS, N);

					// For strided memory access pattern
						// Compute Cost/distance for each point
						comp_dist_glbl_strided<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, i, NUM_POINTS, DIMENSION, ROUNDED_NUM_POINTS);

						// Make the cost/distance cumulative
						// inc_scan_1_block<<<NUM_PARTITIONS,BLOCK_SIZE,BLOCK_SIZE*sizeof(float)>>>(dev_distances,dev_distances_scanned, BLOCK_SIZE,dev_partition_sums);
						inc_scan_1_block_SE<<<NUM_PARTITIONS,WARP_SIZE,WARP_SIZE*sizeof(float)>>>(dev_distances, dev_distances_scanned, dev_partition_sums);
						
						// Make partition_sums cumulative,No need to zero out extra values in partition_sums
						// It is made cumulative in 2 steps, first a segemented scan is done on partition_sums and then
						// then an array containing sum of each segment is scanned and results added to resp. segment
						// so as to get the scan of partition_sums array
						// inc_scan_1_block<<<NUM_META_PARTITIONS,BLOCK_SIZE,BLOCK_SIZE*sizeof(float)>>>(dev_partition_sums, dev_partition_sums, BLOCK_SIZE, dev_meta_block_sums);
						inc_scan_1_block_SE<<<NUM_META_PARTITIONS,WARP_SIZE,WARP_SIZE*sizeof(float)>>>(dev_partition_sums, dev_partition_sums, dev_meta_block_sums);
						inc_scan_1<<<1,META_BLOCK_SUM_SIZE,META_BLOCK_SUM_SIZE*sizeof(float)>>>(dev_meta_block_sums, dev_meta_block_sums, META_BLOCK_SUM_SIZE);
						inc_scan_1_add<<<NUM_META_PARTITIONS,WARP_SIZE>>>(dev_partition_sums,dev_meta_block_sums,WARP_SIZE);
						
						// Sample points from distribution
						// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances_scanned, dev_sampled_indices, dev_rnd, WARP_SIZE, NUM_POINTS, N);
						sample_from_distribution_gpu_copy<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances_scanned, dev_multiset, dev_rnd, WARP_SIZE,  NUM_POINTS, N,dev_data);

						// If scan of distance array is done in-place then we need to undo the scan operation so that we can use the
						// distance array to optimize cost computation step
						// inc_scan_1_rev<<<NUM_PARTITIONS,BLOCK_SIZE,2*BLOCK_SIZE*sizeof(float)>>>(dev_distances,dev_distances,BLOCK_SIZE);

					// Copy back indices of sampled points, no need to copy those points as we have the data here as well
					// checkCudaErrors(cudaMemcpy(sampled_indices,dev_sampled_indices,N*sizeof(int),cudaMemcpyDeviceToHost));
					// for (int copy_i = 0; copy_i < N; ++copy_i)
					// {
					// 	int index = sampled_indices[copy_i];
					// 	for (int copy_j = 0; copy_j < DIMENSION; ++copy_j)
					// 	{
					// 		multiset[copy_i*DIMENSION + copy_j] = data[index*DIMENSION + copy_j];
					// 	}
					// }
					// checkCudaErrors(cudaMemcpy(multiset, dev_multiset, N*DIMENSION*sizeof(float), cudaMemcpyDeviceToHost));
					// copy_to_multiset<<<1,N>>>(dev_multiset,dev_distances,dev_sampled_indices); -- Not tested

					// gettimeofday(&sample_end,NULL);
					// compDistTime += get_time_diff(sample_start,sample_end);
					cudaEventRecord(stop_gpu,0);
					cudaEventSynchronize(stop_gpu);
					cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
					compDistTime += tempTime;

					// Code for sampling on CPU (first GPU implementation)
						// // copy back to host memory for sampling purpose, 
						// cudaMemcpy(distances,dev_distances,NUM_POINTS*sizeof(float),cudaMemcpyDeviceToHost);
						// cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(float),cudaMemcpyDeviceToHost);	

						// // Make it cumulative for sampling purpose, can be done on GPU as well
					
						// // Already made cumulative above
						// gettimeofday(&sample_start,NULL);
						// for (j = 1; j < numGPUThreads; ++j)
						// {
						// 	partition_sums[j] += partition_sums[j-1];
						// }
						// gettimeofday(&sample_end,NULL);
						// makeCumulativeTime += get_time_diff(sample_start,sample_end);
						
						// int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;

						// gettimeofday(&sample_start,NULL);
						// for(j = 0 ; j < N ; j++)
						// {
						// 	rnd[2*j] 	= ((float) rand())/RAND_MAX;
						// 	rnd[2*j+1] 	= ((float) rand())/RAND_MAX;

						// 	int numValidPartitions = NUM_POINTS/per_thread + 1;
						// 	// first pick a block from the local_sums distribution
						// 	int groupNo = sample_from_distribution(partition_sums, 0, numValidPartitions, rnd[2*j]*partition_sums[numValidPartitions-1]);
						// 	// the start and end index of this block
						// 	int startIndex 	= groupNo * per_thread;
						// 	int endIndex 	= (groupNo + 1) * per_thread;

						// 	if(groupNo == numGPUThreads - 1) endIndex = NUM_POINTS;
						// 	// now sample from the cumulative distribution of the block
						// 	int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*j+1]*distances[endIndex-1]);
						// 	for (int k = 0; k < DIMENSION; ++k)
						// 	{
						// 		multiset[j*DIMENSION + k] = data[pointIndex*DIMENSION + k];
						// 	}
						// }
						// gettimeofday(&sample_end,NULL);
						// samplingTime += get_time_diff(sample_start,sample_end);

				// -----------------------GPU-Based implementation of D2-Sample ends------------------------------

				// gettimeofday(&sample_start,NULL);
				cudaEventRecord(start_gpu,0);
				// ----------------------- Mean-Heuristic on GPU starts -------------------------------------------
					int m_i = 0;
					tempPointIndex 	= (((float) rand())/RAND_MAX)*N; // Choose first point uniformly at random here itself
					cudaMemcpy(dev_l2_centers, dev_multiset + tempPointIndex*DIMENSION, DIMENSION*sizeof(float),cudaMemcpyDeviceToDevice);
					for (m_i = 0; m_i < NUM_CLUSTER; ++m_i) // 1 rnd num used in one iteratn,so NUM_CLUSTER random nums required
					{
						rnd[2*m_i] 		= ((float) rand())/RAND_MAX;
						rnd[2*m_i+1] 	= ((float) rand())/RAND_MAX;
					}
					cudaMemcpy(dev_rnd,rnd,2*N*sizeof(float),cudaMemcpyHostToDevice);// Can be overlapped with computation

					// for(m_i = 1; m_i< NUM_CLUSTER; ++m_i) // get NUM_CLUSTER samples out of sampled points 
					// {
						// comp_dist_strided<<<ROUNDED_N/32 ,32,0>>>(dev_multiset, dev_multiset_dist, dev_l2_centers, m_i, N, DIMENSION, ROUNDED_N);
						// // comp_dist<<<1,WARP_SIZE>>>(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, m_i, N, DIMENSION, ROUNDED_N);
						
						// // if(ROUNDED_N != WARP_SIZE*WARP_SIZE)
						// // {
						// // 	printf("ROUNDED_N is not WARP_SIZE*WARP_SIZE\n"); exit(0);							
						// // }
						// // inc_scan_1_block_SE<<<1,WARP_SIZE,WARP_SIZE*sizeof(float)>>>(dev_multiset_dist_partition_sums, dev_multiset_dist_partition_sums, dev_multiset_dist_scanned);

						// // Make the cost/distance cumulative -- try using step efficient scan here
						// // inc_scan_1<<<1,ROUNDED_N,ROUNDED_N*sizeof(float),stream2>>>(dev_multiset_dist, dev_multiset_dist_scanned, ROUNDED_N);
						
						// inc_scan_1_block_SE<<<ROUNDED_N/WARP_SIZE,WARP_SIZE,WARP_SIZE*sizeof(float)>>>(dev_multiset_dist, dev_multiset_dist_scanned, dev_multiset_dist_partition_sums);
						// inc_scan_1_block_SE<<<ROUNDED_N/1024,WARP_SIZE,WARP_SIZE*sizeof(float)>>>(dev_multiset_dist_partition_sums, dev_multiset_dist_partition_sums, dev_multiset_dist_partition_sums + WARP_SIZE - 1);
						// inc_scan_1_add<<<1,WARP_SIZE,0>>>(dev_multiset_dist_scanned,dev_multiset_dist_partition_sums,WARP_SIZE);

						// // Sample points from distribution
						// sample_from_distribution_gpu_copy<<<1,WARP_SIZE,0>>>(dev_multiset_dist_partition_sums, dev_multiset_dist_scanned, dev_l2_centers + m_i*DIMENSION, dev_rnd + 2*m_i, WARP_SIZE, ROUNDED_N, 1,dev_multiset);  // for SE scan

						// // sample_from_distribution_gpu_strided_copy<<<1,WARP_SIZE,0,stream3>>>(dev_multiset_dist_scanned, dev_l2_centers + m_i*DIMENSION, dev_rnd + m_i, ROUNDED_N, 1,dev_multiset); // for simple scan
						
						// // sample_from_distribution_gpu_strided_copy<<<1,WARP_SIZE>>>(dev_multiset_dist, dev_l2_centers + m_i*DIMENSION, dev_rnd + m_i, ROUNDED_N, 1,dev_multiset); // blocked mem access
					// }

					// for(m_i =  0 + 1; m_i< NUM_CLUSTER; ++m_i) // Combining kernels on GPU
					// {

					// 	comp_dist_package<<<1,WARP_SIZE>>>(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, m_i, N, DIMENSION, WARP_SIZE, dev_rnd + 2*m_i); // Blocked computation
						
					// 	// Sample points from distribution
					// 	// sample_from_distribution_gpu_copy<<<1,WARP_SIZE,0>>>(dev_multiset_dist_partition_sums, dev_multiset_dist, dev_l2_centers + m_i*DIMENSION, dev_rnd + 2*m_i, WARP_SIZE, N, 1,dev_multiset);
					// }

					comp_dist_package_with_loop<<<1,WARP_SIZE>>>(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, N, DIMENSION, WARP_SIZE, dev_rnd); // Blocked computation

					// gettimeofday(&sample_end,NULL);
					// samplingTime += get_time_diff(sample_start,sample_end);
					cudaEventRecord(stop_gpu,0);
					cudaEventSynchronize(stop_gpu);
					cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
					samplingTime += tempTime;

					cudaEventRecord(start_gpu,0);
					// gettimeofday(&sample_start,NULL);
					// Find largest cluster now
					// CPU Version 
					cudaMemcpy(l2_centers, dev_l2_centers, NUM_CLUSTER*DIMENSION*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(multiset, dev_multiset, N*DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
					float*	nextCenter =  mean_heuristic_assign(multiset,N,l2_centers);
					memcpy(centers + i*DIMENSION,nextCenter,DIMENSION*sizeof(float));

					// GPU version
					// mean_heuristic_assign_gpu<<<1,64>>>(dev_multiset,N,dev_l2_centers,dev_centers_temp);
					// cudaMemcpy(centers + i*DIMENSION,dev_centers_temp,DIMENSION*sizeof(float),cudaMemcpyDeviceToHost);

					cudaMemcpyToSymbol(dev_centers_global , centers + i*DIMENSION, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyHostToDevice);

				// ------------------------ Mean-Heuristic on GPU ends ---------------------------------------------

				// ----------------------- Mean-Heuristic on CPU starts ------------------------------------------
					// checkCudaErrors(cudaMemcpy(multiset, dev_multiset, N*DIMENSION*sizeof(float), cudaMemcpyDeviceToHost));
					// float* nextCenter = mean_heuristic(multiset,N);
					// memcpy(centers + i*DIMENSION,nextCenter,DIMENSION*sizeof(float));
					// // checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyHostToDevice));
					// cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
					// // checkCudaErrors(cudaMemcpy(dev_centers + i*DIMENSION , nextCenter, DIMENSION*sizeof(float), cudaMemcpyHostToDevice));
				
				// ------------------------ Mean-Heuristic on CPU ends --------------------------------------------
				// gettimeofday(&sample_end,NULL);
				// meanHeuristicTime += get_time_diff(sample_start,sample_end);
				cudaEventRecord(stop_gpu,0);
				cudaEventSynchronize(stop_gpu);
				cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
				meanHeuristicTime += tempTime;
			}
			compDistTime /= 1000;
			samplingTime /= 1000; // GPU events givet time in ms
			meanHeuristicTime /= 1000;
			printf("compDistTime\t\t%2.5f\t%2.5f\n",compDistTime,compDistTime/(NUM_CLUSTER-1) );
			printf("makeCumulativeTime\t%2.5f\t%2.5f\n",makeCumulativeTime,makeCumulativeTime/(NUM_CLUSTER-1) );
			printf("meanHeuristicTime_G\t%2.5f\t%2.5f\n",samplingTime,samplingTime/(NUM_CLUSTER-1) );
			printf("meanHeuristicTime_C\t%2.5f\t%2.5f\n",meanHeuristicTime,meanHeuristicTime/(NUM_CLUSTER-1) );
			cudaProfilerStop();
			// ---------------------- GPU-Based Implementation End --------------------------------------
			
			// ---------------------- CPU-Based Implementation Start ------------------------------------
				// for(i = 0; i < NUM_CLUSTER; i++)
				// {
				// 	struct timeval sample_start,sample_end;
				// 	gettimeofday(&sample_start,NULL);
				// 	multiset = d2_sample(data,centers,NUM_POINTS,N,i);
				// 	// multiset = d2_sample_2(data,centers,NUM_POINTS,N,i,distances);
				// 	gettimeofday(&sample_end,NULL);
				// 	// printf("Time taken for d2_sample::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
				// 	samplingTime_1[i] = get_time_diff(sample_start,sample_end);
				// 	gettimeofday(&sample_start,NULL);
				// 	float* nextCenter = mean_heuristic(multiset,N);
				// 	for (int j = 0; j < DIMENSION; ++j)
				// 	{
				// 		centers[i*DIMENSION + j] = nextCenter[j];
				// 	}
				// 	gettimeofday(&sample_end,NULL);
				// 	// printf("Time taken for mean_heuristic::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
				// 	samplingTime_2[i] = get_time_diff(sample_start,sample_end);
				// }
			// ---------------------- CPU-Based Implementation End --------------------------------------
		}
		else
		{
			printf("Only d2-seeding support for now::%d\n",method);
			printf("Mode::%s\n",mode );
			exit(0);
		}
		gettimeofday(&end,NULL);
		initTime[runNum] = get_time_diff(start,end);

		// now the Lloyd's iterations
		// first we need to figure out the assignments
		gettimeofday(&start,NULL);
		float prev_cost = FLT_MAX;
		int iteration = 0;

		char tempFileName[100];
		sprintf(tempFileName,"%s%d.txt",baseLogFile,runNum);
		logger = fopen(tempFileName,"w");

		// Can make first two static arrays
		int* cluster_counts 	= (int*)malloc(NUM_CLUSTER*sizeof(int)); // number of points assigned to each cluster
		float* cluster_sums		= (float*)malloc(DIMENSION*NUM_CLUSTER*sizeof(float)); // sum of points assigned to each cluster
		int** cluster_counts_pointers 	= (int**)malloc(numThreads*sizeof(int*)); // pointers to local "number of points assigned to each cluster"
		float** cluster_sums_pointers 	= (float**)malloc(numThreads*sizeof(float*)); // pointers to local "sum of points assigned to each cluster"
		
		while(true)
		{
			iteration++;
    		// initially, set everything to zero
    		for(int i = 0; i < NUM_CLUSTER; i++)
    		{
				cluster_counts[i] = 0;
				for(int j = 0; j < DIMENSION; j++)
				{
					cluster_sums[i*DIMENSION + j] = 0;
				}
			}
			// cost according to the current solution
			float current_cost = 0.0;
    		#pragma omp parallel reduction(+: current_cost) 
    		{
    			int tid = omp_get_thread_num();
    			int local_cluster_counts[NUM_CLUSTER]; // local "number of points assigned to each cluster"
    			float local_cluster_sums[DIMENSION*NUM_CLUSTER]; // local "sum of points assigned to each cluster"
    			for(int i = 0; i < NUM_CLUSTER; i++)
    			{
    				local_cluster_counts[i] = 0;
    				for(int j = 0; j < DIMENSION; j++)
    				{
    					local_cluster_sums[i*DIMENSION + j] = 0;
    				}
    			}

    			cluster_counts_pointers[tid] 	= local_cluster_counts; // set the pointer
    			cluster_sums_pointers[tid] 		= local_cluster_sums; // set the pointer
    			int index;
    			float min_dist;
    			float current_dist;
    			// assign each point to their cluster center in parallel. 
    			// update the cost of current solution and keep updating local counts and sums
    			#pragma omp for schedule(static)
    			for (int i = 0; i < NUM_POINTS; i++) 
    			{
    				index = 0;
    				min_dist = FLT_MAX;
    				current_dist = 0;
    				for(int j = 0; j < NUM_CLUSTER; j++)
    				{
    					current_dist = distance(data + i*DIMENSION, centers + j*DIMENSION);
    					if(current_dist < min_dist)
    					{
    						index = j;
    						min_dist = current_dist;
    					}
    				}
    				current_cost += min_dist;
    				local_cluster_counts[index] += 1;
    				for(int j = 0; j < DIMENSION; j++)
    				{
    					local_cluster_sums[index*DIMENSION + j] = local_cluster_sums[index*DIMENSION + j] + data[i*DIMENSION + j];
    				}
		        }

		        // aggregate counts and sums across all threads
		        #pragma omp for schedule(static)
		        for(int i = 0; i < NUM_CLUSTER; i++)
		        {
		        	for(int j = 0; j < numThreads; j++)
		        	{
		        		cluster_counts[i] = cluster_counts[i] + cluster_counts_pointers[j][i];
		        		for(int k = 0; k < DIMENSION; k++)
		        		{
		        			cluster_sums[i*DIMENSION + k] = cluster_sums[i*DIMENSION + k] + cluster_sums_pointers[j][i*DIMENSION + k];
		        		}
		        	}
		        }
		    }
		   
		    if(iteration == 1)
		    {
		    	initCost[runNum] = current_cost;
		    }
		    // now scale all the sums by the number of points at each cluster
		    for(int i = 0; i < NUM_CLUSTER; i++)
		    {
		    	int scaler = cluster_counts[i];
		    	for(int j = 0; j < DIMENSION; j++)
		    	{
		    		centers[i*DIMENSION + j] = cluster_sums[i*DIMENSION + j]/scaler;
		    	}
		    }
		    // log entry
		    fprintf(logger,"Iteration: %d Cost:%f\n",iteration,current_cost);
		    // termination criteria
		    if(1 - current_cost/prev_cost < 0.0001)
		    {
		    	prev_cost = current_cost;
		    	break;
		    }
		    prev_cost = current_cost;
		}

		gettimeofday(&end,NULL);
		finalCost[runNum] 	= prev_cost;
		numIter[runNum] 	= iteration;
		iterTime[runNum] 	= get_time_diff(start,end)/numIter[runNum];
		totalTime[runNum] 	= iterTime[runNum]*numIter[runNum] + initTime[runNum];

		fprintf(logger, "Number of iterations:%f\n",numIter[runNum]);
		fprintf(logger, "Initialization time:%f\n",initTime[runNum]);
		fprintf(logger, "Initialization cost:%f\n",initCost[runNum]);
		fprintf(logger, "Final cost:%f\n",finalCost[runNum]);
		fprintf(logger, "Total time:%f\n",totalTime[runNum]);
		fprintf(logger, "Per iteration time:%f\n",iterTime[runNum]);
		fprintf(logger, "Total iteration time:%f\n",iterTime[runNum]*numIter[runNum]);
		if(method == 2) // d2-seeding
		{
			fprintf(logger,"samplingTime_1:%f\n",mean(samplingTime_1,NUM_CLUSTER));
			fprintf(logger,"samplingTime_2:%f\n",mean(samplingTime_2,NUM_CLUSTER));
		}
		fclose(logger);

		free(cluster_counts);
		free(cluster_sums);
		free(cluster_counts_pointers);
		free(cluster_sums_pointers);

		free(centers);
		free(rnd);
		free(multiset);
		free(distances);
		// free(partition_sums); // partion_sums is used only when using blockedMem Access pattern with and it is brought to CPU for making this sum cumulative

		cudaFree((void**)&dev_distances);
		cudaFree((void**)&dev_partition_sums);
		cudaFree((void**)&dev_sampled_indices);
		cudaFree((void**)&dev_rnd);
		cudaFree((void**)&dev_meta_block_sums);
		
	}

	logger = fopen(resultFile,"w");
	fprintf(logger, "Initial cost: %f %f\n",mean(initCost,numRuns),sd(initCost,numRuns));
	fprintf(logger, "Final cost:   %f %f\n",mean(finalCost,numRuns),sd(finalCost,numRuns));
	fprintf(logger, "Number of iterations: %f %f\n",mean(numIter,numRuns),sd(numIter,numRuns));
	fprintf(logger, "Initialization time:  %f %f\n",mean(initTime,numRuns),sd(initTime,numRuns));
	fprintf(logger, "Per iteration time:   %f %f\n",mean(iterTime,numRuns),sd(iterTime,numRuns));
	fclose(logger);
	return 0;
}

__global__ void copy_to_multiset(float* dev_multiset,float* dev_data,int* dev_sampled_indices)
{
	int tid 	= threadIdx.x;
	int index 	= dev_sampled_indices[tid];
	for (int j = 0; j < DIMENSION; ++j)
	{
		dev_multiset[tid*DIMENSION + j] = dev_data[index*DIMENSION + j];
	}
}


void testScan()
{
	printf("ceil 10/4::%d\n",ceil(10,4));
	printf("roundup 10/4::%d\n",roundUp(10,4));
	printf("ceil 100000/32::%d\n",ceil(100000/32,32));
	printf("roundup 100000/32::%d\n",roundUp(100000/32,32));

	exit(0);
	int num = 8;
	int array_size = SCAN_BLOCK_SIZE*num;
	float* data = (float*)malloc(array_size*sizeof(float));
	float* data_scanned_exc = (float*)malloc(array_size*sizeof(float));
	float* data_scanned_inc = (float*)malloc(array_size*sizeof(float));
	float* block_sums		= (float*)malloc(num*sizeof(float));
	for (int i = 0; i < array_size; ++i)
	{
		data[i] = 1;
		data_scanned_exc[i] = -1;
		data_scanned_inc[i] = -1;
	}
	float* dev_data;
	float* dev_data_scanned_exc;
	float* dev_data_scanned_inc;
	float* dev_block_sums;
	cudaMalloc((void**)&dev_data,array_size*sizeof(float));
	cudaMalloc((void**)&dev_data_scanned_exc,array_size*sizeof(float));
	cudaMalloc((void**)&dev_data_scanned_inc,array_size*sizeof(float));
	cudaMalloc((void**)&dev_block_sums,num*sizeof(float));

	cudaMemcpy(dev_data,data,array_size*sizeof(float),cudaMemcpyHostToDevice);
	// exc_scan_2<<< num , SCAN_BLOCK_SIZE/2,SCAN_BLOCK_SIZE*sizeof(float)>>>(dev_data,dev_data_scanned_exc,SCAN_BLOCK_SIZE);
	inc_scan_1_block<<< num , SCAN_BLOCK_SIZE  ,SCAN_BLOCK_SIZE*sizeof(float)>>>(dev_data,dev_data_scanned_inc,SCAN_BLOCK_SIZE,dev_block_sums);
	inc_scan_1<<< 1 , num  , SCAN_BLOCK_SIZE*sizeof(float)>>>(dev_block_sums ,dev_block_sums,num);

	inc_scan_1_add<<< num, SCAN_BLOCK_SIZE>>>(dev_data_scanned_inc,dev_block_sums,SCAN_BLOCK_SIZE);

	// inc_scan_1_rev<<< num , SCAN_BLOCK_SIZE  ,2*SCAN_BLOCK_SIZE*sizeof(float)>>>(dev_data_scanned_inc,dev_data_scanned_exc,SCAN_BLOCK_SIZE);
	// cudaMemcpy(data_scanned_exc,dev_data,array_size*sizeof(float),cudaMemcpyDeviceToHost);
	// cudaMemcpy(data_scanned_inc,dev_data,array_size*sizeof(float),cudaMemcpyDeviceToHost);
	// cudaMemcpy(data_scanned_exc,dev_data_scanned_exc,array_size*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(data_scanned_inc,dev_data_scanned_inc,array_size*sizeof(float),cudaMemcpyDeviceToHost);
	for (int i = 0; i < array_size; ++i)
	{

		printf("%d\t%.1f\t%.1f\t%.1f\n",i,data[i],data_scanned_exc[i],data_scanned_inc[i]);
	}
	printf("Scan finished successfully\n");
	exit(0);
}

// Code written for 1-D thread-block and 1-D grids ONLY
// Works correctly in-place as well
// Can handle 2*num_threads_per_block elements at max(which in turn is upper bounded by GPU specification)
// Size of shared memory also limits number of elemenst it can handle ...
// Because the part of array scanned by a thread-block needs to copied in the shared memory
// n is size of array/sub-array to be scanned by the thread-block
__global__ void exc_scan_2(float* inData,float* outData,int n)
{
	extern __shared__ 	float temp[]; 	// allocated at run-time
	int thid 		= threadIdx.x;
	int startIndex 	= n*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
	int offset 		= 1;

	
	// Load data into shared memory
	temp[2*thid] 	= inData[startIndex + 2*thid]; // load input into shared memory
	temp[2*thid+1] 	= inData[startIndex + 2*thid+1];

	// perform up-scan 
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d) // This makes sure that only first d threads are doing some work, d is halved in every iteration
		{
			int ai = offset*(2*thid+1) - 1; // -1 converts them to valid indices of the array 
			int bi = offset*(2*thid+2) - 1;
		
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element

	// perform down-scan, same pattern as in up-scan followed but in reverse
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1; // Dividing offset by two
		__syncthreads();
		if (thid < d)
		{
			int ai 		= offset*(2*thid+1) - 1;
			int bi 		= offset*(2*thid+2) - 1;
			float t		= temp[ai];
			temp[ai] 	= temp[bi];
			temp[bi] 	+= t;
		}
	}

	__syncthreads();
	outData[startIndex + 2*thid] 	= temp[2*thid]; // write results to device memory
	outData[startIndex + 2*thid+1] 	= temp[2*thid+1];
}

// Code written for 1-D thread-block ONLY
// Max Size of array it can handle = num_threads in the thread block(which in turn is upper bounded by GPU specification)
// Handles only 1 element per thread, can make it handle 2 or 4 or more  elements but not for now
// n is size of array/sub-array to be scanned by the thread-block
__global__ void inc_scan_1(float* inData,float* outData,int n)
{
	extern __shared__ 	float temp[];
	int thid 		= threadIdx.x;
	int startIndex 	= n*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
	int offset = 1;

	// load input into shared memory
	temp[thid] 	= inData[startIndex + thid]; 

	// perform up-scan 
	for (int d = n>>1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (thid < d) // This makes sure that only first d threads are doing some work, d is halved in every iteration
		{
			int ai = offset*(2*thid+1) - 1; // -1 converts them to valid indices of the array 
			int bi = offset*(2*thid+2) - 1;
		
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// perform down-scan	
	int stride 	= n/4;
	int index 	= thid + 1; // I have written down the pseudo-code for array indices starting from 1
	for (int i = 1; i < n ; i <<= 1) // Need to perform log_2(n) - 1 iterations
	{
		__syncthreads();
		// Check condition for i1
		if (( index + stride < n) && ( index % (2*stride) == 0 ))
		{
			temp[ index + stride -1] += temp[index - 1];
		}
		stride >>= 1; // Half the stride
	}

	__syncthreads();
	outData[startIndex + thid] 	= temp[thid];	 // write results to device memory
}

// Code written for 1-D thread-block ONLY
// Max Size of array it can handle = num_threads in the thread block(which in turn is upper bounded by GPU specification)
// Handles only 1 element per thread, can make it handle 2 or 4 or more  elements but not for now
// n is size of array/sub-array to be scanned by the thread-block
__global__ void inc_scan_1_block(float* inData,float* outData,int n,float* block_sums)
{
	extern __shared__ 	float temp[];
	int thid 		= threadIdx.x;
	int startIndex 	= n*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
	int offset = 1;

	// load input into shared memory
	temp[thid] 	= inData[startIndex + thid]; 

	// perform up-scan 
	for (int d = n>>1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (thid < d) // This makes sure that only first d threads are doing some work, d is halved in every iteration
		{
			int ai = offset*(2*thid+1) - 1; // -1 converts them to valid indices of the array 
			int bi = offset*(2*thid+2) - 1;
		
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// perform down-scan	
	int stride 	= n/4;
	int index 	= thid + 1; // I have written down the pseudo-code for array indices starting from 1
	for (int i = 1; i < n ; i <<= 1) // Need to perform log_2(n) - 1 iterations
	{
		__syncthreads();
		// Check condition for i1
		if (( index + stride < n) && ( index % (2*stride) == 0 ))
		{
			temp[ index + stride -1] += temp[index - 1];
		}
		stride >>= 1; // Half the stride
	}

	__syncthreads();
	outData[startIndex + thid] 	= temp[thid];	 // write results to device memory

	if(thid == blockDim.x - 1)
		block_sums[blockIdx.x] = temp[thid];
}

// Step-Efficient Naive implementation of scan
// Works for fixed array/sub-array of size = 32 (Warp-Size)
__global__ void inc_scan_1_block_SE(float* inData,float* outData,float* block_sums)
{
	extern __shared__ 	float temp[];
	int thid 		= threadIdx.x;
	int startIndex 	= 32*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

	// load input into shared memory
	temp[thid] 	= inData[startIndex + thid]; 

	if(thid >= 1)
		temp[thid] = temp[thid-1] + temp[thid];
	if(thid >= 2)
		temp[thid] = temp[thid-2] + temp[thid];
	if(thid >= 4)
		temp[thid] = temp[thid-4] + temp[thid];
	if(thid >= 8)
		temp[thid] = temp[thid-8] + temp[thid];
	if(thid >= 16)
		temp[thid] = temp[thid-16] + temp[thid];

	outData[startIndex + thid] = temp[thid];
	if(thid == blockDim.x - 1)
		block_sums[blockIdx.x] = temp[thid];
}

__device__ void inc_scan_1_block_SE_package(float* inData,float* outData)
{
	extern __shared__ 	float temp[];
	int thid 		= threadIdx.x;
	int startIndex 	= 32*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

	// load input into shared memory
	temp[thid] 	= inData[startIndex + thid]; 

	if(thid >= 1)
		temp[thid] = temp[thid-1] + temp[thid];
	if(thid >= 2)
		temp[thid] = temp[thid-2] + temp[thid];
	if(thid >= 4)
		temp[thid] = temp[thid-4] + temp[thid];
	if(thid >= 8)
		temp[thid] = temp[thid-8] + temp[thid];
	if(thid >= 16)
		temp[thid] = temp[thid-16] + temp[thid];

	outData[startIndex + thid] = temp[thid];
}

// Add result of to all segments/blocks of the array in order to convert an array with segmented scan to full scan
__global__ void inc_scan_1_add(float* block,float* block_sums,int n)
{
	int thid 		= threadIdx.x;
	int startIndex 	= n*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

	// toAdd should be sum of previous block
	float toAdd = blockIdx.x > 0 ? block_sums[blockIdx.x -1]:0;
	block[startIndex + thid] += toAdd; 
}
// Reverses what an inclusive scan does
// Assumes (blockDim.x == n)
// Size of shared memory array = 2*n
__global__ void inc_scan_1_rev(float* inData,float* outData,int n)
{
	extern __shared__ 	float temp[];
	int thid 		= threadIdx.x;
	int startIndex 	= n*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

	// Size of temp = 2*n
	// First half is calculating result for odd indices
	// Second half is calculating result for even indices
	// load input into shared memory
	temp[thid] 		= inData[startIndex + thid];
	temp[thid + n] 	= temp[thid];

	int curr = ((thid + 1)%2)*n + thid;
	int prev = (thid + n - 1)% n + ((thid + 1)%2)*n;

	outData[startIndex + thid] = thid > 0 ? (temp[curr] - temp[prev]) : temp[curr];
}

int sample_from_distribution(float* probabilities, int startIndex, int endIndex, float prob) 
{
    int start = startIndex,end = endIndex - 1;
    int mid;
    while(start <= end) 
    {
        mid = (start+end)/2;
        if(prob < probabilities[mid-1]) 
        {
            end = mid-1;
        } 
        else if(prob > probabilities[mid]) 
        {
            start = mid+1;
        } 
        else 
        {
            break;
        }
    }
    return mid;
}

// GPU version of sampling code
__global__ void sample_from_distribution_gpu(float* dev_partition_sums, float* dev_distances, int* dev_sampled_indices, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples) 
{

	int numValidPartitions = (dev_NUM_POINTS + per_thread - 1)/per_thread ;
	int start,mid,end,groupNo,pointIndex;
	float prob;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if( i < dev_num_samples)
	{
		// first pick a block from the local_sums distribution
		// int groupNo =sample_from_distribution(partition_sums,0, numValidPartitions, rnd[2*i]*partition_sums[numValidPartitions-1]);

		start 	= 0;
		end 	= numValidPartitions - 1;
	    prob 	= dev_rnd[2*i]*dev_partition_sums[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_partition_sums[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_partition_sums[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    groupNo =  mid;

		// the start and end index of this block
		// int startIndex 	= groupNo*per_thread;
		// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
		// now sample from the cumulative distribution of the block
		// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1]);

		start 	= groupNo*per_thread;
		end 	= min((groupNo + 1)*per_thread, NUM_POINTS) - 1;
	    prob 	= dev_rnd[2*i+1]*dev_distances[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_distances[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_distances[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    pointIndex = mid;
		dev_sampled_indices[i] = pointIndex;
	}
}

// This function does not copy indices to dev_sampled_indices, instead it copies relevant points to dev_multiset directly
__global__ void sample_from_distribution_gpu_copy(float* dev_partition_sums, float* dev_distances, float* dev_multiset, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples,float* dev_data) 
{

	int numValidPartitions = (dev_NUM_POINTS + per_thread - 1)/per_thread ;
	int start,mid,end,groupNo,pointIndex;
	float prob;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if( tid < dev_num_samples)
	{
		// first pick a block from the local_sums distribution
		// int groupNo =sample_from_distribution(partition_sums,0, numValidPartitions, rnd[2*tid]*partition_sums[numValidPartitions-1]);

		start 	= 0;
		end 	= numValidPartitions - 1;
	    prob 	= dev_rnd[2*tid]*dev_partition_sums[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_partition_sums[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_partition_sums[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    groupNo =  mid;

		// the start and end index of this block
		// int startIndex 	= groupNo*per_thread;
		// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
		// now sample from the cumulative distribution of the block
		// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*tid+1]*distances[endIndex-1]);

		start 	= groupNo*per_thread;
		end 	= min((groupNo + 1)*per_thread, NUM_POINTS) - 1;
	    prob 	= dev_rnd[2*tid+1]*dev_distances[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_distances[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_distances[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    pointIndex = mid;
	    for (int j = 0; j < DIMENSION; ++j)
	    {
	    	dev_multiset[tid*DIMENSION + j] = dev_data[pointIndex*DIMENSION + j];
	    }
		// dev_sampled_indices[tid] = pointIndex;
	}
}

// Sampling for case of strided memory access pattern, no dev_partition_sums here
__global__ void sample_from_distribution_gpu_strided(float* dev_distances, int* dev_sampled_indices, float* dev_rnd, int dev_NUM_POINTS, int dev_num_samples) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if( i < dev_num_samples)
	{
		int start,mid,end,pointIndex;
		float prob;
		
		// the start and end index of this block
		// int startIndex 	= groupNo*per_thread;
		// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
		// now sample from the cumulative distribution of the block
		// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1]);

		start 	= 0;
		end 	= dev_NUM_POINTS - 1;
	    prob 	= dev_rnd[i]*dev_distances[end];
	    mid 	= (start+end)/2;
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_distances[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_distances[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    pointIndex = mid;
		dev_sampled_indices[i] = pointIndex;
	}
}

// Copies sample directly into multiset instead of just copying indices of sampled points
__global__ void sample_from_distribution_gpu_strided_copy(float* dev_distances, float* dev_multiset, float* dev_rnd, int dev_NUM_POINTS, int dev_num_samples, float* dev_data) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if( i < dev_num_samples)
	{
		int start,mid,end,pointIndex;
		float prob;
		
		// the start and end index of this block
		// int startIndex 	= groupNo*per_thread;
		// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
		// now sample from the cumulative distribution of the block
		// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1]);

		start 	= 0;
		end 	= dev_NUM_POINTS - 1;
	    prob 	= dev_rnd[i]*dev_distances[end];
	    mid 	= (start+end)/2;
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_distances[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_distances[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    pointIndex = mid;
		// dev_sampled_indices[i] = pointIndex;
		for (int j = 0; j < DIMENSION; ++j)
		{
			dev_multiset[i*DIMENSION + j] = dev_data[pointIndex*DIMENSION + j];
		}
	}
}

// This function calcuates required distance for all points and partitions
// Need to do an all-prefix sum after this to make this thing cumulative
// Can be optimized by using distances calculated in previous iteration, i.e. when the previous center was sampled
// This does not do any sampling business
// Need not call this function when centerIter = 0,
// Not optimized to use distance calculted in previous iteration to calculate distance/cost for points 
__global__ void comp_dist_2(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// int jump = blockDim.x*gridDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	float min_dist = FLT_MAX, local_dist,temp,prev_val = 0;
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
	{
		min_dist = FLT_MAX;
		for (int i = 0; i < centerIter; ++i)
		{
			local_dist = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist);
		}
		dev_distances[dataIndex] = min_dist*min_dist + prev_val;
		// dev_distances[dataIndex] = min_dist*min_dist;
		prev_val = dev_distances[dataIndex];
	}
	dev_partition_sums[tid] = prev_val;
}

// Optimised to use previous distance values to calculate min_dist for points in next iteration
__global__ void comp_dist(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	float min_dist = FLT_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
	{

		if (centerIter == 1) // This is the first time dev_distances will get its values
		{
			min_dist 	= 0;
			int i = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
				min_dist += temp*temp;
			}
			dev_distances[dataIndex] = min_dist*min_dist + prev_val; // make it cumulative as you calculate it
			prev_val = dev_distances[dataIndex];
		}
		else
		{
			int i = centerIter - 1; // i denotes the last center that was added to the list of centers
			min_dist 	= dev_distances[dataIndex] - old_prev_val;
			old_prev_val= dev_distances[dataIndex];
			local_dist 	= 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist*local_dist);
			dev_distances[dataIndex] = min_dist + prev_val;  // No need to square min_dist here, it is already squared value
			prev_val = dev_distances[dataIndex];
		}
		
	}
	dev_partition_sums[tid] = prev_val;
}


__global__ void comp_dist_package(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads,float *dev_rnd)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	float min_dist = FLT_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
	{

		if (centerIter == 1) // This is the first time dev_distances will get its values
		{
			min_dist 	= 0;
			int i = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
				min_dist += temp*temp;
			}
			dev_distances[dataIndex] = min_dist*min_dist + prev_val; // make it cumulative as you calculate it
			prev_val = dev_distances[dataIndex];
		}
		else
		{
			int i = centerIter - 1; // i denotes the last center that was added to the list of centers
			min_dist 	= dev_distances[dataIndex] - old_prev_val;
			old_prev_val= dev_distances[dataIndex];
			local_dist 	= 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist*local_dist);
			dev_distances[dataIndex] = min_dist + prev_val;  // No need to square min_dist here, it is already squared value
			prev_val = dev_distances[dataIndex];
		}		
	}
	dev_partition_sums[tid] = prev_val;

	if( blockIdx.x == 0)
	{
		__shared__ float temp[WARP_SIZE];
		int thid 		= threadIdx.x;
		int startIndex 	= 32*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

		// load input into shared memory
		temp[thid] 	= dev_partition_sums[startIndex + thid]; 

		if(thid >= 1)
			temp[thid] = temp[thid-1] + temp[thid];
		if(thid >= 2)
			temp[thid] = temp[thid-2] + temp[thid];
		if(thid >= 4)
			temp[thid] = temp[thid-4] + temp[thid];
		if(thid >= 8)
			temp[thid] = temp[thid-8] + temp[thid];
		if(thid >= 16)
			temp[thid] = temp[thid-16] + temp[thid];

		dev_partition_sums[startIndex + thid] = temp[thid];
	}

	// comp(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, m_i, N, DIMENSION, WARP_SIZE); // Blocked computation

	// sample(dev_multiset_dist_partition_sums, dev_multiset_dist, dev_l2_centers + m_i*DIMENSION, dev_rnd + 2*m_i, WARP_SIZE, N, 1,dev_multiset);  // for SE scan

	// sample_from_distribution_gpu_copy(float* dev_partition_sums,float* dev_distances, 
	// 	float* dev_multiset, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples,float* dev_data) 

	if ((blockIdx.x == 0) && (threadIdx.x == 0))
	{
		float* dev_multiset = dev_centers + centerIter*DIMENSION; 
		int numValidPartitions = (numPoints + per_thread - 1)/per_thread ;
		int start,mid,end,groupNo,pointIndex;
		float prob;
		int tid = blockIdx.x*blockDim.x + threadIdx.x;

		// first pick a block from the local_sums distribution
		// int groupNo =sample_from_distribution(partition_sums,0, numValidPartitions, rnd[2*tid]*partition_sums[numValidPartitions-1]);

		start 	= 0;
		end 	= numValidPartitions - 1;
	    prob 	= dev_rnd[2*tid]*dev_partition_sums[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_partition_sums[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_partition_sums[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    groupNo =  mid;

		// the start and end index of this block
		// int startIndex 	= groupNo*per_thread;
		// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
		// now sample from the cumulative distribution of the block
		// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*tid+1]*distances[endIndex-1]);

		start 	= groupNo*per_thread;
		end 	= min((groupNo + 1)*per_thread, NUM_POINTS) - 1;
	    prob 	= dev_rnd[2*tid+1]*dev_distances[end];
	    while(start <= end) 
	    {
	        mid = (start+end)/2;
	        if(prob < dev_distances[mid-1]) 
	        {
	            end = mid-1;
	        } 
	        else if(prob > dev_distances[mid]) 
	        {
	            start = mid+1;
	        } 
	        else 
	        {
	            break;
	        }
	    }
	    pointIndex = mid;
	    for (int j = 0; j < DIMENSION; ++j)
	    {
	    	dev_multiset[tid*DIMENSION + j] = dev_data[pointIndex*DIMENSION + j];
	    }
	}
	
}

__global__ void comp_dist_package_with_loop(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int numPoints,int dev_dimension,int numGPUThreads,float *dev_rnd)
{
	for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	{
		// Starting off with very simplistic 1-D threads blocks and 1-D grids
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
		int startIndex = tid*per_thread;
		int endIndex = min((tid + 1)*per_thread,numPoints);
		float min_dist = FLT_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
		for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
		{

			if (centerIter == 1) // This is the first time dev_distances will get its values
			{
				min_dist 	= 0;
				int i = 0;
				for (int j = 0; j < dev_dimension; ++j)
				{
					temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
					min_dist += temp*temp;
				}
				dev_distances[dataIndex] = min_dist*min_dist + prev_val; // make it cumulative as you calculate it
				prev_val = dev_distances[dataIndex];
			}
			else
			{
				int i = centerIter - 1; // i denotes the last center that was added to the list of centers
				min_dist 	= dev_distances[dataIndex] - old_prev_val;
				old_prev_val= dev_distances[dataIndex];
				local_dist 	= 0;
				for (int j = 0; j < dev_dimension; ++j)
				{
					temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
					local_dist += temp*temp;
				}
				min_dist = min(min_dist,local_dist*local_dist);
				dev_distances[dataIndex] = min_dist + prev_val;  // No need to square min_dist here, it is already squared value
				prev_val = dev_distances[dataIndex];
			}		
		}
		dev_partition_sums[tid] = prev_val;

		if( blockIdx.x == 0)
		{
			__shared__ float temp[WARP_SIZE];
			int thid 		= threadIdx.x;
			int startIndex 	= 32*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

			// load input into shared memory
			temp[thid] 	= dev_partition_sums[startIndex + thid]; 

			if(thid >= 1)
				temp[thid] = temp[thid-1] + temp[thid];
			if(thid >= 2)
				temp[thid] = temp[thid-2] + temp[thid];
			if(thid >= 4)
				temp[thid] = temp[thid-4] + temp[thid];
			if(thid >= 8)
				temp[thid] = temp[thid-8] + temp[thid];
			if(thid >= 16)
				temp[thid] = temp[thid-16] + temp[thid];

			dev_partition_sums[startIndex + thid] = temp[thid];
		}

		// comp(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, m_i, N, DIMENSION, WARP_SIZE); // Blocked computation

		// sample(dev_multiset_dist_partition_sums, dev_multiset_dist, dev_l2_centers + m_i*DIMENSION, dev_rnd + 2*m_i, WARP_SIZE, N, 1,dev_multiset);  // for SE scan

		// sample_from_distribution_gpu_copy(float* dev_partition_sums,float* dev_distances, 
		// 	float* dev_multiset, float* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_num_samples,float* dev_data) 

		if ((blockIdx.x == 0) && (threadIdx.x == 0))
		{
			float* dev_multiset = dev_centers + centerIter*DIMENSION; 
			int numValidPartitions = (numPoints + per_thread - 1)/per_thread ;
			int start,mid,end,groupNo,pointIndex;
			float prob;
			int tid = blockIdx.x*blockDim.x + threadIdx.x;

			// first pick a block from the local_sums distribution
			// int groupNo =sample_from_distribution(partition_sums,0, numValidPartitions, rnd[2*tid]*partition_sums[numValidPartitions-1]);

			start 	= 0;
			end 	= numValidPartitions - 1;
		    prob 	= dev_rnd[2*centerIter + 2*tid]*dev_partition_sums[end];
		    while(start <= end) 
		    {
		        mid = (start+end)/2;
		        if(prob < dev_partition_sums[mid-1]) 
		        {
		            end = mid-1;
		        } 
		        else if(prob > dev_partition_sums[mid]) 
		        {
		            start = mid+1;
		        } 
		        else 
		        {
		            break;
		        }
		    }
		    groupNo =  mid;

			// the start and end index of this block
			// int startIndex 	= groupNo*per_thread;
			// int endIndex 	= min((groupNo + 1)*per_thread, NUM_POINTS);
			// now sample from the cumulative distribution of the block
			// int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*tid+1]*distances[endIndex-1]);

			start 	= groupNo*per_thread;
			end 	= min((groupNo + 1)*per_thread, NUM_POINTS) - 1;
		    prob 	= dev_rnd[2*centerIter + 2*tid+1]*dev_distances[end];
		    while(start <= end) 
		    {
		        mid = (start+end)/2;
		        if(prob < dev_distances[mid-1]) 
		        {
		            end = mid-1;
		        } 
		        else if(prob > dev_distances[mid]) 
		        {
		            start = mid+1;
		        } 
		        else 
		        {
		            break;
		        }
		    }
		    pointIndex = mid;
		    for (int j = 0; j < DIMENSION; ++j)
		    {
		    	dev_multiset[tid*DIMENSION + j] = dev_data[pointIndex*DIMENSION + j];
		    }
		}
	}
	
}
// Optimised to use previous distance values to calculate min_dist for points in next iteration
// Also makes use of constant memory for storing centers
__global__ void comp_dist_glbl(float* dev_data,float* dev_distances,float* dev_partition_sums,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	float min_dist = FLT_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
	{

		if (centerIter == 1) // This is the first time dev_distances will get its values
		{
			min_dist = 0;
			int i = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[i*dev_dimension + j];
				min_dist += temp*temp;
			}
			dev_distances[dataIndex] = min_dist*min_dist + prev_val; // make it cumulative as you calculate it
			prev_val = dev_distances[dataIndex];
		}
		else
		{
			int i = centerIter - 1; // i denotes the last center that was added to the list of centers
			min_dist 	= dev_distances[dataIndex] - old_prev_val;
			old_prev_val= dev_distances[dataIndex];
			local_dist 	= 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[i*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist*local_dist);
			dev_distances[dataIndex] = min_dist + prev_val;  // No need to square min_dist here, it is already squared value
			prev_val = dev_distances[dataIndex];
		}
	}
	dev_partition_sums[tid] = prev_val;
}

// This addtionally does things in strided fashion as opposed to assigning each thread some fixed block
__global__ void comp_dist_glbl_strided(float* dev_data,float* dev_distances,int centerIter,int numPoints,int dev_dimension,int rndedNumPoints)
{
	int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;
	int stride 		= blockDim.x*gridDim.x;
	float min_dist, local_dist, temp;
	while(dataIndex < numPoints)
	{
		if (centerIter == 1) // This is the first time dev_distances will get its values
		{
			min_dist = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[j]; // Accessing 0th center of dev_center_global
				min_dist += temp*temp;
			}
			dev_distances[dataIndex] = min_dist*min_dist;
		}
		else
		{
			// Assuming that dev_distances has been made cumulative, after this function call
			min_dist 	= dev_distances[dataIndex];
			local_dist 	= 0.0; 
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[(centerIter-1)*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist*local_dist);
			dev_distances[dataIndex] = min_dist;  // --No-- Need to square min_dist here, it is *not*  already squared value
			
			// Bad way of doing things, we need not iterate over all previous centers
			// min_dist = FLT_MAX;
			// for (int i = 0; i < centerIter; ++i)
			// {
			// 	local_dist 	= 0;
			// 	for (int j = 0; j < dev_dimension; ++j)
			// 	{
			// 		temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[i*dev_dimension + j];
			// 		local_dist += temp*temp;
			// 	}
			// 	min_dist = min(min_dist,local_dist);
			// }
			// dev_distances[dataIndex] = min_dist*min_dist;  // --No-- Need to square min_dist here, it is *not*  already squared value
		}
		dataIndex += stride;
	}

	// Zero out the extra region which was padded to make size of distance array a multiple of BLOCK_SIZE
	if( dataIndex < rndedNumPoints)
	{
		dev_distances[dataIndex] = 0;
	}
}

// This addtionally does things in strided fashion as opposed to assigning each thread some fixed block
// Does not use constant memory
__global__ void comp_dist_strided(float* dev_data,float* dev_distances,float* dev_centers,int centerIter,int numPoints,int dev_dimension,int rndedNumPoints)
{
	int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;
	int stride 		= blockDim.x*gridDim.x;
	float min_dist, local_dist, temp;
	while(dataIndex < numPoints)
	{
		if (centerIter == 1) // This is the first time dev_distances will get its values
		{
			min_dist = 0;
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[j]; // Accessing 0th center of dev_center_global
				min_dist += temp*temp;
			}
			dev_distances[dataIndex] = min_dist*min_dist;
		}
		else
		{
			// Assuming that dev_distances has been made cumulative, after this function call
			min_dist 	= dev_distances[dataIndex];
			local_dist 	= 0.0; 
			for (int j = 0; j < dev_dimension; ++j)
			{
				temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[(centerIter-1)*dev_dimension + j];
				local_dist += temp*temp;
			}
			min_dist = min(min_dist,local_dist*local_dist);
			dev_distances[dataIndex] = min_dist;  // --No-- Need to square min_dist here, it is *not*  already squared value
			
			// Bad way of doing things, we need not iterate over all previous centers
			// min_dist = FLT_MAX;
			// for (int i = 0; i < centerIter; ++i)
			// {
			// 	local_dist 	= 0;
			// 	for (int j = 0; j < dev_dimension; ++j)
			// 	{
			// 		temp = dev_data[dataIndex*dev_dimension + j] - dev_centers[i*dev_dimension + j];
			// 		local_dist += temp*temp;
			// 	}
			// 	min_dist = min(min_dist,local_dist);
			// }
			// dev_distances[dataIndex] = min_dist*min_dist;  // --No-- Need to square min_dist here, it is *not*  already squared value
		}
		dataIndex += stride;
	}

	// Zero out the extra region which was padded to make size of distance array a multiple of BLOCK_SIZE
	if( dataIndex < rndedNumPoints)
	{
		dev_distances[dataIndex] = 0;
	}
}

// generate numSamples sized multiset from weighted data with weights wrt. centers where the current size of centers is size
// numPts : number of points in data
// numSamples: number of points to sample
// size : size of centers i.e. number of centers chosen already
float* d2_sample(float* data,float* centers,int numPts, int numSamples, int size)
{
	
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
	float* distances 	= (float*)malloc(numPts*sizeof(float));
    float* local_sums	= (float*)malloc(numThreads*sizeof(float));   // local sums. first is sum for [0...numPts/numThreads-1], and so on. This is also a cumulative distribution.
    float* result 		= (float*)malloc(numSamples*DIMENSION*sizeof(float));
    for (int i = 0; i < numSamples; ++i)
    {
    	for (int j = 0; j < DIMENSION; ++j)
    	{
    		result[i*DIMENSION + j] = 0;
    	}
    }
    // we're gonna need 2*numSamples random numbers. 
    float* rnd 		= (float*)malloc(2*numSamples*sizeof(float));
    int i;
	for(i = 0; i < 2*numSamples; i++){
		rnd[i] = ((float) rand())/RAND_MAX;
	}
    #pragma omp parallel
    {
    	struct timeval start,end;
    	gettimeofday(&start,NULL);
    	// create blocks of data
        int tid 		= omp_get_thread_num();
        int per_thread 	= (numPts + numThreads - 1) / numThreads;
        int lower 		= tid * per_thread;
        int higher 		= (tid + 1) * per_thread;
        if(tid == numThreads - 1) higher = numPts;
        int block_size 	= higher - lower;
        float min_dist, local_dist;
        float* p;
        float prev_val = 0;
        // cost of each block
        float local_sum = 0;
        int center_size = size;
        int i,j;
        for(i = 0;i < block_size;i++)
        {    
            if(center_size == 0){
                local_sum += 1;
                distances[lower+i] = 1 + prev_val;
            } else{
                p = data + (lower+i)*DIMENSION;
                min_dist = distance(p,centers);
                for (j = 1; j < center_size; j++) {
                    local_dist = distance(p, centers + j*DIMENSION);
                    min_dist = min(min_dist, local_dist); // calculating minimum distances
                }
                local_sum +=  min_dist * min_dist;
                distances[lower+i] =  min_dist * min_dist + prev_val; // make cumulative 
            }
            prev_val = distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<numThreads;i++){
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
            // printf("Number of threads::%d\n",omp_get_num_threads());
        }
    	gettimeofday(&end,NULL);
    	float cost_time = get_time_diff(start,end);
        #pragma omp barrier
        gettimeofday(&start,NULL);
        #pragma omp for
        for(int i = 0;i < numSamples;i++){
        	// first pick a block from the local_sums distribution
            int groupNo = sample_from_distribution(local_sums, 0, numThreads, rnd[i*2]*local_sums[numThreads-1]);
            // the start and end index of this block
            int startIndex = groupNo * per_thread;
            int endIndex = (groupNo + 1) * per_thread;
            if(groupNo == numThreads - 1) endIndex = numPts;
            // now sample from the cumulative distribution of the block
            int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1]);
            for (int j = 0; j < DIMENSION; ++j)
            {
            	result[i*DIMENSION + j] = data[pointIndex*DIMENSION + j];
            }
            // memcpy(result + i*DIMENSION, data + pointIndex*DIMENSION, DIMENSION*sizeof(float));
        }
        gettimeofday(&end,NULL);
        float sample_time = get_time_diff(start,end);
        // if (center_size >= 99)
        // {
		    // printf("Cost computation time 	::%f\n",cost_time);
			// printf("Sampling time 		::%f\n",sample_time);	
        // } 	
    }

    free(distances);
    free(local_sums);
    return result;
}

// This version of d2_sample has optimized cost calculation by using cost computed in last iteration
float* d2_sample_2(float* data,float* centers,int numPts, int numSamples, int size, float* distances)
{
	
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
    float* local_sums	= (float*)malloc(numThreads*sizeof(float));   // local sums. first is sum for [0...numPts/numThreads-1], and so on. This is also a cumulative distribution.
    float* result 		= (float*)malloc(numSamples*DIMENSION*sizeof(float));
    for (int i = 0; i < numSamples; ++i)
    {
    	for (int j = 0; j < DIMENSION; ++j)
    	{
    		result[i*DIMENSION + j] = 0;
    	}
    }
    // we're gonna need 2*numSamples random numbers. 
    float* rnd 		= (float*)malloc(2*numSamples*sizeof(float));
    int i;
	for(i = 0; i < 2*numSamples; i++){
		rnd[i] = ((float) rand())/RAND_MAX;
	}
    #pragma omp parallel
    {
    	struct timeval start,end;
    	gettimeofday(&start,NULL);
    	// create blocks of data
        int tid 		= omp_get_thread_num();
        int per_thread 	= (numPts + numThreads - 1) / numThreads;
        int lower 		= tid * per_thread;
        int higher 		= (tid + 1) * per_thread;
        if(tid == numThreads - 1) higher = numPts;
        int block_size 	= higher - lower;
        float min_dist, local_dist;
        float* p;
        float prev_val = 0, old_prev_val = 0;
        // cost of each block
        float local_sum = 0;
        int center_size = size;
        int i;
        for(i = 0;i < block_size;i++)
        {    
            if(center_size == 0)
            {
                local_sum += 1;
                distances[lower+i] = 1 + prev_val;
            } 
            else if (center_size == 1)
            {
                p = data + (lower+i)*DIMENSION;
                min_dist = distance(p,centers);
                local_sum +=  min_dist * min_dist;
                distances[lower+i] =  min_dist * min_dist + prev_val; // make cumulative 
            }
            else
            {
            	p = data + (lower+i)*DIMENSION;
                // min_dist = distance(p,centers[0]);
                min_dist   		= distances[lower+i] - old_prev_val;
            	old_prev_val 	= distances[lower+i];  // Important for it to have old value of distance[lower+i] so that min_dist is correct in next iteration
                local_dist 		= distance(p,centers + (center_size-1)*DIMENSION); // Find distance wrt last added new center;
                local_dist 		= local_dist*local_dist;
                min_dist 		= min(min_dist,local_dist);
                
                local_sum 		+=  min_dist; // min_dist is already squared here because it is calculated usign cumulative distance
                distances[lower+i] = min_dist + prev_val; // make cumulative 
            	prev_val = distances[lower+i];
            }
            prev_val = distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<numThreads;i++){
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
        }
    	gettimeofday(&end,NULL);
    	float cost_time = get_time_diff(start,end);
        #pragma omp barrier
        gettimeofday(&start,NULL);

        #pragma omp for
        for(int i = 0;i < numSamples;i++){
        	// first pick a block from the local_sums distribution
            int groupNo = sample_from_distribution(local_sums, 0, numThreads, rnd[i*2]*local_sums[numThreads-1]);
            // the start and end index of this block
            int startIndex = groupNo * per_thread;
            int endIndex = (groupNo + 1) * per_thread;
            if(groupNo == numThreads - 1) endIndex = numPts;
            // now sample from the cumulative distribution of the block
            int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1]);
            for (int j = 0; j < DIMENSION; ++j)
            {
            	result[i*DIMENSION + j] = data[pointIndex*DIMENSION + j];
            }
            // memcpy(result + i*DIMENSION, data + pointIndex*DIMENSION, DIMENSION*sizeof(float));
        }
        gettimeofday(&end,NULL);
        float sample_time = get_time_diff(start,end);
        // if (center_size >= 99)
        // {
		    // printf("Cost computation time 	::%f\n",cost_time);
			// printf("Sampling time 		::%f\n",sample_time);	
        // } 	
    }

    free(local_sums);
    return result;
}


// This version of d2_sample has optimized cost calculation by using cost computed in last iteration
// This is specially optimized for mean_heurisitic
float* d2_sample_3(float* data,float* centers,int numPts, int size, float* distances)
{
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
    float* local_sums	= (float*)malloc(numThreads*sizeof(float));   // local sums. first is sum for [0...numPts/numThreads-1], and so on. This is also a cumulative distribution.
    float* result 		= (float*)malloc(DIMENSION*sizeof(float));
    
    for (int j = 0; j < DIMENSION; ++j)
    {
    	result[j] = 0;
    }
    // we're gonna need 2*numSamples random numbers. 
    float rnd[2];// 		= (float*)malloc(2*sizeof(float));
	rnd[0] = ((float) rand())/RAND_MAX;
	rnd[1] = ((float) rand())/RAND_MAX;

    #pragma omp parallel
    {
    	// create blocks of data
        int tid 		= omp_get_thread_num();
        int per_thread 	= (numPts + numThreads - 1) / numThreads;
        int lower 		= tid * per_thread;
        int higher 		= (tid + 1) * per_thread;
        if(tid == numThreads - 1) higher = numPts;
        int block_size 	= higher - lower;
        float min_dist, local_dist;
        float* p;
        float prev_val = 0, old_prev_val = 0;
        // cost of each block
        float local_sum = 0;
        int center_size = size;
        int i;
        for(i = 0;i < block_size;i++)
        {    
            if(center_size == 0)
            {
                local_sum += 1;
                distances[lower+i] = 1 + prev_val;
            } 
            else if (center_size == 1)
            {
                p = data + (lower+i)*DIMENSION;
                min_dist = distance(p,centers);
                local_sum +=  min_dist * min_dist;
                distances[lower+i] =  min_dist * min_dist + prev_val; // make cumulative 
            }
            else
            {
            	p = data + (lower+i)*DIMENSION;
                // min_dist = distance(p,centers[0]);
                min_dist   		= distances[lower+i] - old_prev_val;
            	old_prev_val 	= distances[lower+i];  // Important for it to have old value of distance[lower+i] so that min_dist is correct in next iteration
                local_dist 		= distance(p,centers + (center_size-1)*DIMENSION); // Find distance wrt last added new center;
                local_dist 		= local_dist*local_dist;
                min_dist 		= min(min_dist,local_dist);
                
                local_sum 		+=  min_dist; // min_dist is already squared here because it is calculated usign cumulative distance
                distances[lower+i] = min_dist + prev_val; // make cumulative 
            }
            prev_val = distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<numThreads;i++)
            {
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
        }
    }

	int per_thread 	= (numPts + numThreads - 1) / numThreads;
    // first pick a block from the local_sums distribution
	int groupNo = sample_from_distribution(local_sums, 0, numThreads, rnd[0]*local_sums[numThreads-1]);
	// the start and end index of this block
	int startIndex = groupNo * per_thread;
	int endIndex = (groupNo + 1) * per_thread;
	if(groupNo == numThreads - 1) endIndex = numPts;
	// now sample from the cumulative distribution of the block
	int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[1]*distances[endIndex-1]);
	for (int j = 0; j < DIMENSION; ++j)
	{
		result[j] = data[pointIndex*DIMENSION + j];
    }
    free(local_sums);
    return result;
}

float* mean_heuristic(float* multiset,int multisetSize)
{
	// first do a kmeans++ initialiation on the multiset
	int i,j;
	// gettimeofday(&start,NULL);
	float* level_2_sample = (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
	float* distances = (float*)malloc(multisetSize*sizeof(float));
	for(i = 0; i < NUM_CLUSTER; i++)
	{
		// float* point = d2_sample(multiset,level_2_sample,multisetSize,1,i);
		// float* point = d2_sample_2(multiset,level_2_sample,multisetSize,1,i,distances);
		float* point = d2_sample_3(multiset,level_2_sample,multisetSize,i,distances);
		for (j = 0; j < DIMENSION; ++j)
		{
			level_2_sample[i*DIMENSION + j] = point[j];
		}
		// memcpy(level_2_sample + i*DIMENSION, point, DIMENSION*sizeof(float)) ;
	}

	// gettimeofday(&end,NULL);
	// printf("Time taken to choose k centers::%f\n",get_time_diff(start,end));
	// gettimeofday(&start,NULL);
	int* counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int)); // number of points assigned to each kmeans++ center
    float* cluster_means 	= (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float)); // for taking the centroid later on. We maintain a sum of all points assigned to a center here.
    for (i = 0; i < NUM_CLUSTER; i++)
    {
        counts[i] = 0;
        for(j = 0; j< DIMENSION; j++)
        {
        	cluster_means[i*DIMENSION + j] = 0;
        }
    }
    // here the heuristic does things in a parallel fashion
    // maintain a local structure for each thread to keep track of cluster sums and counts
    int** local_tmp_counts 				= (int**)malloc(numThreads*sizeof(int*));
    float** local_tmp_cluster_means 	= (float**)malloc(numThreads*sizeof(float*));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* local_counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int));
        float* local_cluster_means = (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
        for (int i = 0; i < NUM_CLUSTER; i++) 
        {
        	local_counts[i] = 0;
            for(int j = 0; j < DIMENSION; j++) 
            {
            	local_cluster_means[i*DIMENSION + j] = 0;
            }
        }
        local_tmp_counts[tid] = local_counts;  // save the pointers to local data structures
        local_tmp_cluster_means[tid] = local_cluster_means;
        float min_dist, tmp_dist;
        int index;
        #pragma omp for schedule(static)
        for (int i = 0; i < multisetSize; i++) 
        {
            min_dist = distance(level_2_sample,multiset + i*DIMENSION);  // distance of each kmeans++ center from the points in sampled_set
            index = 0;
            for (int j = 1; j < NUM_CLUSTER; j++) 
            {
                tmp_dist = distance(level_2_sample + j*DIMENSION, multiset+ i*DIMENSION); // figure out the minimum and assign the point to that kmeans++ center
                if (tmp_dist < min_dist) 
                {
                    min_dist = tmp_dist;
                    index = j;
                }
            }
            for(int j = 0; j < DIMENSION; j++)
            {
            	local_cluster_means[index*DIMENSION + j] += multiset[i*DIMENSION + j];
            }
            local_counts[index]++;
        }
        // aggregate across all threads
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_CLUSTER; i++) 
        {
            for (int p = 0; p < numThreads ; p++) 
            {
            	for(int j = 0; j < DIMENSION; j++)
            	{
            		cluster_means[i*DIMENSION + j] += local_tmp_cluster_means[p][i*DIMENSION + j];
            	}
                counts[i] += local_tmp_counts[p][i];
            }
        }
        free(local_counts);
        free(local_cluster_means);
    }
    int max = counts[0];
    int index = 0;
    for (int i = 1; i < NUM_CLUSTER; i++) 
    {
        if (counts[i] > max) 
        {
            max = counts[i];
            index = i; // largest cluster with maximum points from sampled_set assigned to it.
        }
    }
    // gettimeofday(&end,NULL);
    // printf("Time for finding partitions::%f\n",get_time_diff(start,end));
    // do the scaling to find the mean
    for(int i = 0; i < DIMENSION; i++){
    	cluster_means[index*DIMENSION + i] /= counts[index];
    }
    free(counts);
    free(cluster_means);
    free(local_tmp_counts);
    free(local_tmp_cluster_means);
    return cluster_means + index*DIMENSION;
}

// Just assigns points to clusters
float* mean_heuristic_assign(float* multiset,int multisetSize,float* level_2_sample)
{
	// first do a kmeans++ initialiation on the multiset
	int i,j;
	// gettimeofday(&start,NULL);

	// float* level_2_sample = (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
	// for(i = 0; i < NUM_CLUSTER; i++)
	// {
	// 	float* point = d2_sample(multiset,level_2_sample,multisetSize,1,i);
	// 	for (j = 0; j < DIMENSION; ++j)
	// 	{
	// 		level_2_sample[i*DIMENSION + j] = point[j];
	// 	}
	// 	// memcpy(level_2_sample + i*DIMENSION, point, DIMENSION*sizeof(float)) ;
	// }

	// gettimeofday(&end,NULL);
	// printf("Time taken to choose k centers::%f\n",get_time_diff(start,end));
	// gettimeofday(&start,NULL);
	int* counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int)); // number of points assigned to each kmeans++ center
    float* cluster_means 	= (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float)); // for taking the centroid later on. We maintain a sum of all points assigned to a center here.
    for (i = 0; i < NUM_CLUSTER; i++)
    {
        counts[i] = 0;
        for(j = 0; j< DIMENSION; j++)
        {
        	cluster_means[i*DIMENSION + j] = 0;
        }
    }
    // here the heuristic does things in a parallel fashion
    // maintain a local structure for each thread to keep track of cluster sums and counts
    int** local_tmp_counts 				= (int**)malloc(numThreads*sizeof(int*));
    float** local_tmp_cluster_means 	= (float**)malloc(numThreads*sizeof(float*));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* local_counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int));
        float* local_cluster_means = (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));
        for (int i = 0; i < NUM_CLUSTER; i++) 
        {
        	local_counts[i] = 0;
            for(int j = 0; j < DIMENSION; j++) 
            {
            	local_cluster_means[i*DIMENSION + j] = 0;
            }
        }
        local_tmp_counts[tid] = local_counts;  // save the pointers to local data structures
        local_tmp_cluster_means[tid] = local_cluster_means;
        float min_dist, tmp_dist;
        int index;
        #pragma omp for schedule(static)
        for (int i = 0; i < multisetSize; i++) 
        {
            min_dist = distance(level_2_sample,multiset + i*DIMENSION);  // distance of each kmeans++ center from the points in sampled_set
            index = 0;
            for (int j = 1; j < NUM_CLUSTER; j++) 
            {
                tmp_dist = distance(level_2_sample + j*DIMENSION, multiset+ i*DIMENSION); // figure out the minimum and assign the point to that kmeans++ center
                if (tmp_dist < min_dist) 
                {
                    min_dist = tmp_dist;
                    index = j;
                }
            }
            for(int j = 0; j < DIMENSION; j++)
            {
            	local_cluster_means[index*DIMENSION + j] += multiset[i*DIMENSION + j];
            }
            local_counts[index]++;
        }
        // aggregate across all threads
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_CLUSTER; i++) 
        {
            for (int p = 0; p < numThreads ; p++) 
            {
            	for(int j = 0; j < DIMENSION; j++)
            	{
            		cluster_means[i*DIMENSION + j] += local_tmp_cluster_means[p][i*DIMENSION + j];
            	}
                counts[i] += local_tmp_counts[p][i];
            }
        }
        free(local_counts);
        free(local_cluster_means);
    }
    int max = counts[0];
    int index = 0;
    for (int i = 1; i < NUM_CLUSTER; i++) 
    {
        if (counts[i] > max) 
        {
            max = counts[i];
            index = i; // largest cluster with maximum points from sampled_set assigned to it.
        }
    }
    // gettimeofday(&end,NULL);
    // printf("Time for finding partitions::%f\n",get_time_diff(start,end));
    // do the scaling to find the mean
    for(int i = 0; i < DIMENSION; i++){
    	cluster_means[index*DIMENSION + i] /= counts[index];
    }
    free(counts);
    free(cluster_means);
    free(local_tmp_counts);
    free(local_tmp_cluster_means);
    return cluster_means + index*DIMENSION;
}

// Working with just 1 thread-block with 32 threads in it for now
__global__ void mean_heuristic_assign_gpu(float* dev_multiset,int multisetSize, float* dev_l2_samples, float* dev_centers_temp)
{
	__shared__ int counts[NUM_CLUSTER];
	// __shared__ int counts_index[NUM_CLUSTER];
	__shared__ float cluster_means[NUM_CLUSTER*DIMENSION];
	int tid = threadIdx.x;

	// Initialize all the means and counts to zero
	int pointIndex = tid;
	while(pointIndex < NUM_CLUSTER*DIMENSION)
	{
		cluster_means[pointIndex] = 0;
		pointIndex += blockDim.x;
	}
	pointIndex = tid;
    while (pointIndex < NUM_CLUSTER)
    {
    	counts[pointIndex] = 0;
		pointIndex += blockDim.x;
    }
	pointIndex = tid;
    while(pointIndex < multisetSize)
    {
		// min_dist = distance(dev_l2_samples,multiset + i*DIMENSION);
		float min_dist  = 0,temp = 0;
		int centerIndex = 0;
		for (int k = 0; k < DIMENSION; ++k)
		{
			 temp  = dev_l2_samples[0*DIMENSION +  k] - dev_multiset[pointIndex*DIMENSION + k];
			 min_dist += temp*temp;
		}
		for (int j = 1; j < NUM_CLUSTER; j++) 
		{
			// tmp_dist = distance(dev_l2_samples + j*DIMENSION, multiset+ i*DIMENSION);
			float tmp_dist = 0;
			for (int k = 0; k < DIMENSION; ++k)
			{
				 temp = dev_l2_samples[j*DIMENSION + k] - dev_multiset[pointIndex*DIMENSION + k];
				 tmp_dist += temp*temp;
			}
			if (tmp_dist < min_dist) 
			{
				min_dist = tmp_dist;
				centerIndex = j;
			}
		}
		for(int j = 0; j < DIMENSION; j++)
		{
			cluster_means[centerIndex*DIMENSION + j] += dev_multiset[pointIndex*DIMENSION + j]; // Need to make sure this is atomic
		}
		counts[centerIndex]++;
		pointIndex += blockDim.x;
    }
	
	// // Finds max count but need to find index for maxCount
	// 	if(thid >= 1)
	// 	{
	// 		counts[thid] = max(counts[thid-1] , counts[thid]);
	// 		if( counts[thid - 1] > counts[thid])
	// 		{
	// 			counts[thid] = counts[thid - 1];
	// 			counts_index[thid] = thid - 1; 
	// 		}
	// 		else
	// 		{
	// 			counts[thid] = counts[thid];
	// 			counts_index[thid] = thid;
	// 		}
	// 	}
	// 	if(thid >= 2)
	// 	{
	// 		counts[thid] = max(counts[thid-2] , counts[thid]);
	// 		if( counts[thid - 2] > counts[thid])
	// 		{
	// 			counts[thid] = counts[thid - 2];
	// 			counts_index[thid] = thid - 2; 
	// 		}
	// 		else
	// 		{
	// 			counts[thid] = counts[thid];
	// 			counts_index[thid] = thid;
	// 		}
	// 	}
	// 	if(thid >= 4)
	// 	{
	// 		counts[thid] = max(counts[thid-4] , counts[thid]);
	// 		if( counts[thid - 4] > counts[thid])
	// 		{
	// 			counts[thid] = counts[thid - 4];
	// 			counts_index[thid] = thid - 4; 
	// 		}
	// 		else
	// 		{
	// 			counts[thid] = counts[thid];
	// 			counts_index[thid] = thid;
	// 		}
	// 	}
	// 	if(thid >= 8)
	// 	{
	// 		counts[thid] = max(counts[thid-8] , counts[thid]);
	// 		if( counts[thid - 8] > counts[thid])
	// 		{
	// 			counts[thid] = counts[thid - 8];
	// 			counts_index[thid] = thid - 8; 
	// 		}
	// 		else
	// 		{
	// 			counts[thid] = counts[thid];
	// 			counts_index[thid] = thid;
	// 		}
	// 	}
	// 	if(thid >= 16)
	// 	{
	// 		counts[thid] = max(counts[thid-16] , counts[thid]);
	// 		if( counts[thid - 16] > counts[thid])
	// 		{
	// 			counts[thid] = counts[thid - 16];
	// 			counts_index[thid] = thid - 16; 
	// 		}
	// 		else
	// 		{
	// 			counts[thid] = counts[thid];
	// 			counts_index[thid] = thid;
	// 		}
	// 	}

	if( tid == 0)
	{
		int maxSoFar = counts[0];
		int maxIndex = 0;
		for (int i = 1; i < NUM_CLUSTER; ++i)
		{
			if (counts[i] > maxSoFar)
			{
				maxSoFar = counts[i];
				maxIndex = i;
			}
		}
		for(int i = 0; i < DIMENSION; i++)
	    {
	    	dev_centers_temp[i] =  cluster_means[maxIndex*DIMENSION + i] / counts[maxIndex];
	    }
	}
}

float distance(float* p1, float* p2)
{
	int i;
	float localSum = 0;
	for (i = 0; i < DIMENSION; ++i)
	{
		localSum += (p1[i] - p2[i])*(p1[i] - p2[i]);
	}
	return localSum;
}

void write_centers_to_file(float* centers)
{
	FILE* writer = fopen("dataExchange.txt","w");

	for (int i = 0; i < NUM_CLUSTER; ++i)
	{
		for (int j = 0; j < DIMENSION; ++j)
		{
			fprintf(writer, "%f ",centers[i*DIMENSION + j] );
		}
		fprintf(writer, "\n");
	}
}

static inline float mean(float* a, int n)
{
	float sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	return sum/n;
}

static inline float sd(float* a, int n)
{
	float sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	float mean = sum/n;
	sum = 0;
	for(int i = 0; i < n; i++){
		sum += (a[i] - mean) * (a[i] - mean);
	}
	return sqrt(sum/n);
}

static inline float get_time_diff(struct timeval t1, struct timeval t2){
	return t2.tv_sec - t1.tv_sec + 1e-6 * (t2.tv_usec - t1.tv_usec);
}
