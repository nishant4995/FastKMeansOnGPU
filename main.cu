#include "main.h"

// Initial cost: 109883399602176.000000 0.000000 For seed 32
// Final cost:   95299603267584.000000 0.000000
// Number of iterations: 8.000000 0.000000
// Initialization time:  0.110906 0.000000
// Per iteration time:   0.014346 0.000000

// g++ -D BIRCH1 -g mainOMP.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch1 -lm

// This will work for NUM_POINTS = 1024*1024 as we are using BLOCK_SIZE = 32

// random_device rd;
// mt19937 gen(rd());
// unsigned int max_val = gen.max();

void test()
{
	FILE* reader = fopen("../data/train-images.idx3-ubyte","rb");
	int ctr = 0;
	while(!feof(reader))
	{
		unsigned int temp;
		int f = fread(&temp,4,1,reader);
		// fscanf(reader,"%d",&temp);
		printf("temp::%u\n",temp);
		ctr++;
		if(ctr > 4)
			break;

	}
	exit(0);
}


int numThreads = 1;
__constant__ float dev_centers_global[NUM_CLUSTER*DIMENSION]; // For using constant memory


int main(int argc, char const *argv[])
{
	// test();
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
	// srand(32);
	srand(time(NULL));

		// for (int i = 0; i < 10; ++i)
		// {
		// 	int tempPointIndex 	= (((float) rand())/RAND_MAX)*NUM_POINTS;
		// 	printf("%d\t%d\n",i,tempPointIndex );
		// }
		// exit(0);

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
		printf("Running runNum::%d for %s\n",runNum,DATA );
		gettimeofday(&start,NULL);

		int numBlocks 			= 8;
		int numThreadsPerBlock 	= 1024;
		int numSampleBlocks 	= 128;
		int numSampleTperB 		= 32;
		int numGPUThreads 		= numBlocks*numThreadsPerBlock;

		// If using strided memory access pattern
		int META_BLOCK_SUM_SIZE = roundToPowerOf2(NUM_META_PARTITIONS); 
		int ROUNDED_N 			= roundToPowerOf2(N); 

		float* centers;// 		= (float*)malloc(NUM_CLUSTER*DIMENSION*sizeof(float));	
		float* rnd;// 			= (float*)malloc(2*N*sizeof(float));
		int rndNumReqd = NUM_CLUSTER*(2*N + 2*NUM_CLUSTER); //2N for sampling multiset,2*NUM_CLUSTER for level2sampling from multiset
		cudaHostAlloc((void**)&centers,NUM_CLUSTER*DIMENSION*sizeof(float),cudaHostAllocDefault);
		cudaHostAlloc((void**)&rnd,rndNumReqd*sizeof(float),cudaHostAllocDefault);

		float* dev_distances;
		float* dev_distances_scanned;
		float* dev_partition_sums;
		float* dev_meta_block_sums;
		float* dev_rnd;
		int*   dev_sampled_indices;

		float* dev_multiset;
		float* dev_multiset_dist_scanned;
		float* dev_l2_centers;
		int*   dev_l2_center_indices;

		float* dev_centers_temp;
		cudaMalloc((void**)&dev_centers_temp,DIMENSION*sizeof(float));
					
		
		gettimeofday(&start,NULL);
		// float* dev_centers; // Needed when not using constant memory for centers
		// printf("ROUNDED_NUM_POINTS::%d\n",ROUNDED_NUM_POINTS );
		// printf("rndNumReqd::%d\n",rndNumReqd );
		checkCudaErrors(cudaMalloc((void**)&dev_distances,ROUNDED_NUM_POINTS*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_distances_scanned,ROUNDED_NUM_POINTS*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,numGPUThreads*sizeof(float))); // For blocked access pattern for distance array
		// checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,ROUNDED_NUM_PARTITIONS*sizeof(float))); // For strided access pattern for distance array
		checkCudaErrors(cudaMalloc((void**)&dev_meta_block_sums,META_BLOCK_SUM_SIZE*sizeof(float)));
		// checkCudaErrors(cudaMalloc((void**)&dev_rnd,2*N*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_rnd,rndNumReqd*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_sampled_indices,N*sizeof(float)));

		checkCudaErrors(cudaMalloc((void**)&dev_multiset,N*DIMENSION*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_multiset_dist_scanned,ROUNDED_N*sizeof(float)));

		checkCudaErrors(cudaMalloc((void**)&dev_l2_centers,NUM_CLUSTER*DIMENSION*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&dev_l2_center_indices,NUM_CLUSTER*sizeof(int)));

		// checkCudaErrors(cudaMalloc((void**)&dev_centers,NUM_CLUSTER*DIMENSION*sizeof(float))); // No need when using constant memory
		gettimeofday(&end,NULL);

		// printf("Time take to init array of GPU::%f::%f\n",get_time_diff(start,end),get_time_diff(start,end)/NUM_CLUSTER); 

		float tempTime;
		cudaEvent_t start_gpu,stop_gpu, totat_gpu_start,total_gpu_end;
		cudaEventCreate(&start_gpu);
		cudaEventCreate(&stop_gpu);
		cudaEventCreate(&totat_gpu_start);
		cudaEventCreate(&total_gpu_end);
		cudaStream_t copyStream, compStream, stream3;
		cudaStreamCreate(&copyStream);
		cudaStreamCreate(&compStream);
		cudaStreamCreate(&stream3);

		gettimeofday(&start,NULL);
		// initialize the initial centers
		if(method == 2) // d2-seeding
		{  
			// ---------------------- GPU-Based Implementation Start ------------------------------------
				// cudaProfilerStart();
				// cudaEventRecord(totat_gpu_start,compStream);

				// // First choosing the first point uniformly at random, no need to sample N points and all here
				// int tempPointIndex 	= (((float) rand())/RAND_MAX)*NUM_POINTS;
				// memcpy(centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float));
				// checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float),0,cudaMemcpyHostToDevice));
				// // checkCudaErrors(cudaMemcpy(dev_centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(float),cudaMemcpyHostToDevice));
				// int randNumIter = 0;
				// for(randNumIter = 0; randNumIter < 2*N; ++randNumIter)
				// 	rnd[randNumIter] 	= ((float) rand())/RAND_MAX;
				// checkCudaErrors(cudaMemcpy(dev_rnd,rnd,2*N*sizeof(float),cudaMemcpyHostToDevice));
			
				// float compDistTime = 0, meanHeuristicTime = 0;
				// for(i = 1; i < NUM_CLUSTER; i++)
				// {
				// 	// printf("Choosing center::%d\n",i );
				// 	// cudaEventRecord(start_gpu,0);
				// 	// -----------------------GPU-Based implementation of D2-Sample ends------------------------------
					
				// 		// For blocked access pattern
				// 			// comp_dist_glbl<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, i, NUM_POINTS, DIMENSION, numGPUThreads);
				// 			// checkCudaErrors(cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(float),cudaMemcpyDeviceToHost));	
				// 			// checkCudaErrors(cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(float),cudaMemcpyDeviceToHost));	
				// 			// for (j = 1; j < numGPUThreads; ++j) // Need to do this scan operation on GPU only, but testing things first
				// 			// {
				// 			// 	partition_sums[j] += partition_sums[j-1];
				// 			// }
				// 			// checkCudaErrors(cudaMemcpy(dev_partition_sums,partition_sums,numGPUThreads*sizeof(float),cudaMemcpyHostToDevice));
			
				// 			// int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;
				// 			// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances, dev_sampled_indices, dev_rnd, per_thread, NUM_POINTS, N);

				// 		// For strided memory access pattern
				// 			// Compute Cost/distance for each point
				// 			comp_dist_glbl_strided<<<numBlocks,numThreadsPerBlock,0,compStream>>>(dev_data, dev_distances, i, NUM_POINTS, DIMENSION, ROUNDED_NUM_POINTS);

				// 			// Make the cost/distance cumulative
				// 			// inc_scan_1_block<<<NUM_PARTITIONS,BLOCK_SIZE,BLOCK_SIZE*sizeof(float)>>>(dev_distances,dev_distances_scanned, BLOCK_SIZE,dev_partition_sums);
				// 			inc_scan_1_block_SE<<<NUM_PARTITIONS,WARP_SIZE,WARP_SIZE*sizeof(float),compStream>>>(dev_distances, dev_distances_scanned, dev_partition_sums);
							
				// 			// Make partition_sums cumulative,No need to zero out extra values in partition_sums
				// 			// It is made cumulative in 2 steps, first a segemented scan is done on partition_sums and then
				// 			// then an array containing sum of each segment is scanned and results added to resp. segment
				// 			// so as to get the scan of partition_sums array
				// 			// inc_scan_1_block<<<NUM_META_PARTITIONS,BLOCK_SIZE,BLOCK_SIZE*sizeof(float)>>>(dev_partition_sums, dev_partition_sums, BLOCK_SIZE, dev_meta_block_sums);
				// 			inc_scan_1_block_SE<<<NUM_META_PARTITIONS,WARP_SIZE,WARP_SIZE*sizeof(float),compStream>>>(dev_partition_sums, dev_partition_sums, dev_meta_block_sums);
				// 			inc_scan_1<<<1,META_BLOCK_SUM_SIZE,META_BLOCK_SUM_SIZE*sizeof(float),compStream>>>(dev_meta_block_sums, dev_meta_block_sums, META_BLOCK_SUM_SIZE);
				// 			inc_scan_1_add<<<NUM_META_PARTITIONS,WARP_SIZE,0,compStream>>>(dev_partition_sums, dev_meta_block_sums, WARP_SIZE);
							
				// 			// Sample points from distribution
				// 			// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances_scanned, dev_sampled_indices, dev_rnd, WARP_SIZE, NUM_POINTS, N);
				// 			sample_from_distribution_gpu_copy<<<numSampleBlocks,numSampleTperB,0,compStream>>>(dev_partition_sums, dev_distances_scanned, dev_multiset, dev_rnd + (i-1)*(2*N + 2*NUM_CLUSTER), WARP_SIZE,  NUM_POINTS, N,dev_data);

				// 	// -----------------------GPU-Based implementation of D2-Sample ends------------------------------
					
				// 	// Copying 2*NUM_CLUSTER random numbers on GPU to be used by meanHeuristic in next iteration
				// 	int tempBound = i*(2*N + 2*NUM_CLUSTER);
				// 	for(; randNumIter < tempBound; ++randNumIter)
				// 		rnd[randNumIter] 	= ((float) rand())/RAND_MAX;
				// 	checkCudaErrors(cudaMemcpyAsync(dev_rnd + tempBound - 2*NUM_CLUSTER,rnd + tempBound - 2*NUM_CLUSTER,2*NUM_CLUSTER*sizeof(float),cudaMemcpyHostToDevice, compStream));

				// 	// cudaEventRecord(stop_gpu,compStream);
				// 	// cudaEventSynchronize(stop_gpu);
				// 	// cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
				// 	// compDistTime += tempTime;

				// 	// cudaEventRecord(start_gpu,compStream);
				// 	// ----------------------- Mean-Heuristic on GPU starts -------------------------------------------

				// 		// comp_dist_package_with_loop<<<1,WARP_SIZE*WARP_SIZE,0,compStream>>>(dev_multiset,dev_l2_centers, N,dev_rnd + i*(2*N + 2*NUM_CLUSTER)-2*NUM_CLUSTER ); // Blocked computation

				// 		// comp_dist_package_with_loop_original<<<1,WARP_SIZE>>>(dev_multiset, dev_multiset_dist, dev_multiset_dist_partition_sums, dev_l2_centers, N,dev_rnd); // Blocked computation, correct

				// 		// checkCudaErrors(cudaEventRecord(stop_gpu,compStream));
				// 		// cudaEventSynchronize(stop_gpu);
				// 		// cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
				// 		// meanHeuristicTime += tempTime;
				// 		// cudaEventRecord(start_gpu,compStream);

				// 		// Find largest cluster now
				// 		// CPU Version 
				// 			// cudaMemcpy(l2_centers, dev_l2_centers, NUM_CLUSTER*DIMENSION*sizeof(float),cudaMemcpyDeviceToHost);
				// 			// cudaMemcpy(multiset, dev_multiset, N*DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
				// 			// float*	nextCenter =  mean_heuristic_assign(multiset,N,l2_centers);
				// 			// memcpy(centers + i*DIMENSION,nextCenter,DIMENSION*sizeof(float));

				// 		// GPU version
				// 		// mean_heuristic_assign_gpu<<<1,WARP_SIZE*WARP_SIZE,0,compStream>>>(dev_multiset, N, dev_l2_centers, dev_centers_temp);


				// 		// Combined comp_dist_package and mean_heuristic_assign_gpu into 1 function
				// 		mean_heuristic_GPU<<<1,WARP_SIZE*WARP_SIZE,0,compStream>>>(dev_multiset,dev_l2_centers, N, dev_rnd + i*(2*N + 2*NUM_CLUSTER) - 2*NUM_CLUSTER, dev_centers_temp );

				// 		// Copying 2*N random numbers on GPU to be used by sampling multiset in next iteration
				// 		if( i < NUM_CLUSTER -1)// We don't want any random number to be transeffered to GPU in the end, hence this if
				// 		{
				// 			int tempBound = 2*N + i*(2*N + 2*NUM_CLUSTER);
				// 			for(; randNumIter < tempBound; ++randNumIter)
				// 				rnd[randNumIter] 	= ((float) rand())/RAND_MAX;

				// 			checkCudaErrors(cudaMemcpyAsync(dev_rnd + tempBound - 2*N ,rnd + tempBound - 2*N, 2*N*sizeof(float),cudaMemcpyHostToDevice, compStream));
				// 		}

				// 		// Copy next chosen center to dev_centers_global and centers array(copying to centers array can be avoideds)
				// 		// checkCudaErrors(cudaMemcpy(centers + i*DIMENSION,dev_centers_temp,DIMENSION*sizeof(float),cudaMemcpyDeviceToHost)); -- Copying to centers array from GPU
				// 		// cudaMemcpyToSymbol(dev_centers_global , dev_centers_temp, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyDeviceToDevice); -- Now copying from centers array to dev_centers_GPU. These 2 copies are avoided. Directly copy to dev_centers_global

				// 		checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global, dev_centers_temp, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyDeviceToDevice));

				// 	// ------------------------ Mean-Heuristic on GPU ends ---------------------------------------------

				// 	// ----------------------- Mean-Heuristic on CPU starts ------------------------------------------
				// 		// checkCudaErrors(cudaMemcpy(multiset, dev_multiset, N*DIMENSION*sizeof(float), cudaMemcpyDeviceToHost));
				// 		// float* nextCenter = mean_heuristic(multiset,N);
				// 		// memcpy(centers + i*DIMENSION,nextCenter,DIMENSION*sizeof(float));
				// 		// // checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyHostToDevice));
				// 		// cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(float), i*DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
				// 		// // checkCudaErrors(cudaMemcpy(dev_centers + i*DIMENSION , nextCenter, DIMENSION*sizeof(float), cudaMemcpyHostToDevice));
					
				// 	// ------------------------ Mean-Heuristic on CPU ends --------------------------------------------
					
				// 	// cudaEventRecord(stop_gpu,compStream);
				// 	// cudaEventSynchronize(stop_gpu);
				// 	// cudaEventElapsedTime(&tempTime,start_gpu,stop_gpu);
				// 	// meanHeuristicTime += tempTime;
				// }
				// compDistTime /= 1000; // GPU events givet time in ms
				// meanHeuristicTime /= 1000;
				// // printf("compDistTime\t\t%2.5f\t%2.5f\n",compDistTime,compDistTime/(NUM_CLUSTER-1) );
				// // printf("meanHeuristicTime\t%2.5f\t%2.5f\n",meanHeuristicTime,meanHeuristicTime/(NUM_CLUSTER-1) );

				// checkCudaErrors(cudaMemcpyFromSymbol( centers, dev_centers_global, NUM_CLUSTER*DIMENSION*sizeof(float), 0, cudaMemcpyDeviceToHost ));

				// cudaEventRecord(total_gpu_end,compStream);
				// cudaEventSynchronize(total_gpu_end);
				// cudaEventElapsedTime(&tempTime,totat_gpu_start,total_gpu_end);
				// printf("Total time taken::%f\n",tempTime/1000);	
				// cudaProfilerStop();
			// ---------------------- GPU-Based Implementation End --------------------------------------
			
			// ---------------------- CPU-Based Implementation Start ------------------------------------
				float* distances = (float*)malloc(NUM_POINTS*sizeof(float));
				float* multiset;
				for(i = 0; i < NUM_CLUSTER; i++)
				{
					struct timeval sample_start,sample_end;
					gettimeofday(&sample_start,NULL);

					// multiset = d2_sample(data,centers,NUM_POINTS,N,i);
					// multiset = d2_sample_2(data,centers,NUM_POINTS,N,i,distances);
					multiset = d2_sample_4(data,centers,NUM_POINTS,N,i,distances); // Better for to avoid errors due to optimizations
					gettimeofday(&sample_end,NULL);
					// printf("Time taken for d2_sample::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
					samplingTime_1[i] = get_time_diff(sample_start,sample_end);
					gettimeofday(&sample_start,NULL);
					float* nextCenter = mean_heuristic(multiset,N);
					for (int j = 0; j < DIMENSION; ++j)
					{
						centers[i*DIMENSION + j] = nextCenter[j];
					}
					gettimeofday(&sample_end,NULL);
					// printf("Time taken for mean_heuristic::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
					samplingTime_2[i] = get_time_diff(sample_start,sample_end);
				}
				free(distances);
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
		printf("Time::%f\n",initTime[runNum] );
		// now run Lloyd's iterations
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

		// free(centers);
		// free(rnd);
		cudaFreeHost(rnd);
		cudaFreeHost(centers);
		// free(multiset);
		// free(distances);
		// free(partition_sums); // partion_sums is used only when using blockedMem Access pattern with and it is brought to CPU for making this sum cumulative

		cudaFree((void*)dev_distances);
		cudaFree((void*)dev_partition_sums);
		cudaFree((void*)dev_sampled_indices);
		cudaFree((void*)dev_rnd);
		cudaFree((void*)dev_meta_block_sums);
		
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

int roundToPowerOf2(int num)
{
	int roundedNum = num - 1; 
	int ctr = 0;
	// Code to roundUp to next power of 2 if not already
	while(roundedNum >= 1 )
	{
		roundedNum /= 2;
		ctr += 1;
	}
	roundedNum = 1<<ctr;
	return roundedNum;
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
	int startIndex 	= WARP_SIZE*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed

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

// Designed for mean_heuristic part to work on sampled multiset
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


// This is compact kernel which does all the level2-sampling work
// This should be launched with at least as many threads so that no thread needs to compute cost of more than one point

// This function is just to check how much benefit do we get after merging comp_dist_package_with_loop and mean_heuristic_assign_gpu
__global__ void mean_heuristic_GPU(float* dev_data, float* dev_centers,int numPoints,float *dev_rnd,float* dev_centers_temp)
{
	// In this form of things we can probably do away with distance array and just keep scanned_dist array
	// Just need one more var per thread to achieve this
	// __shared__ float dev_centers[NUM_CLUSTER*DIMENSION];
	float distance = 0;
	__shared__ int tempSample[1];
	__shared__ float dist_scanned[WARP_SIZE*WARP_SIZE];// Needed to be made dynamic for different datasets
	__shared__ float partition_sums[WARP_SIZE];// Needed to be made dynamic for different datasets

	if(threadIdx.x  == 0)
	{
		tempSample[0] = -1;
		int tempIndex = dev_rnd[0]*numPoints;
		for (int i = 0; i < DIMENSION; ++i)
		{
			dev_centers[i] = dev_data[tempIndex*DIMENSION + i];
		}
	}

	for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	{
		int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;		
		if(dataIndex < numPoints) // Rest of the threads will be idle!
		{
			if (centerIter == 1) // This is the first time dev_distances will get its values --> Can take this out of loop
			{
				float min_dist 	= 0, temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[j]; // Computing cost wrt 1st center
					min_dist += temp*temp;

				}
				distance = min_dist*min_dist;
			}
			else
			{
				int i = centerIter - 1; // i denotes the last center that was added to the list of centers
				float local_dist 	= 0,temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
					local_dist += temp*temp;
				}
				distance = fminf(distance , local_dist*local_dist);
			}
		}
		
		// __syncthreads();//-- Not needed here, the warp which has finished its cost computation work is the one which
		// goes ahead and scans its part of the data array
	
		// int startIndex 	= WARP_SIZE*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
		// dist_scanned[thid] 	= dev_distances[startIndex + thid]; // load input into shared memory
		dist_scanned[dataIndex] 	= distance; // load input into shared memory
		if((dataIndex % WARP_SIZE) >= 1)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-1] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 2)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-2] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 4)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-4] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 8)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-8] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 16)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-16] + dist_scanned[dataIndex];

		// printf("centerIter::%d,dataIndex:: %d distance::%f\tdist_scanned::%f\n",centerIter,dataIndex, distance,dist_scanned[dataIndex]);
		// dist_scanned[dataIndex] = dist_scanned[dataIndex];

		if((dataIndex+ 1)%WARP_SIZE == 0) // For 1 block and many threads
		{
			partition_sums[(dataIndex)/WARP_SIZE] = dist_scanned[dataIndex];
			// printf("iter::%d,partitionsums::%d::%d::%f\n",centerIter,dataIndex, dataIndex/WARP_SIZE, partition_sums[dataIndex/WARP_SIZE] );
		}

		__syncthreads(); // Needed as partition_sums need to be computed before it can be made cumulative

		
		if (dataIndex < WARP_SIZE) // Just need 1 warp to do scan partition_sums
		{
			if(dataIndex >= 1)
				partition_sums[dataIndex] = partition_sums[dataIndex-1] + partition_sums[dataIndex];
			if(dataIndex >= 2)
				partition_sums[dataIndex] = partition_sums[dataIndex-2] + partition_sums[dataIndex];
			if(dataIndex >= 4)
				partition_sums[dataIndex] = partition_sums[dataIndex-4] + partition_sums[dataIndex];
			if(dataIndex >= 8)
				partition_sums[dataIndex] = partition_sums[dataIndex-8] + partition_sums[dataIndex];
			if(dataIndex >= 16)
				partition_sums[dataIndex] = partition_sums[dataIndex-16] + partition_sums[dataIndex];

			// Use an entire WARP to sample faster
			// This works when each segment of distance array has size = WARP_SIZE and 
			// partition_sums also has size = WARP_SIZE. This is okay for Birch datasets as NUM_CLUSTER*10 = 1000
			// which is rounded to 32*32. Need to make sure this is modified/adapted for other datasets

			float* dev_multiset = dev_centers + centerIter*DIMENSION; 
			// first pick a block from the local_sums distribution
			int per_thread = WARP_SIZE;
			int partitionNum,end;
			float prob 	= dev_rnd[2*centerIter]*partition_sums[per_thread-1];
			if( prob < partition_sums[dataIndex])
			{
				if(dataIndex == 0)
				{
					// partitionNum  = 0;
					tempSample[0] = 0;
				}
				else if( partition_sums[dataIndex-1] < prob )
				{
					tempSample[0] = dataIndex; // Important to do the update in a shared var so that other threads can also see it
					// partitionNum = dataIndex;
				}
			}
			// the start and end index of this block
			// int startIndex 	= partitionNum*per_thread;
			// int endIndex 	= min((partitionNum + 1)*per_thread, NUM_POINTS);
			// now sample from the cumulative distribution of the block
			partitionNum 	= tempSample[0];
			// if ((partitionNum == -1) && (threadIdx.x == 0)) // For bebugging
			// {	
			// 	for (int tempI = 0; tempI < centerIter; ++tempI)
			// 	{
			// 		printf("center::%d\t",tempI );
			// 		for (int tempJ = 0; tempJ < DIMENSION; ++tempJ)
			// 		{
			// 			printf("%f\t",dev_centers[tempI*DIMENSION + tempJ]);
			// 		}
			// 		printf("\n");
			// 	}
			// 	for (int tempI = 0; tempI < WARP_SIZE*WARP_SIZE; ++tempI)
			// 	{
			// 		printf("distance::%d::%f\n",tempI,dist_scanned[tempI]);
			// 		if ((tempI + 1)%WARP_SIZE == 0)
			// 			printf("iter::%d,partitionsums::%d::%f\n",centerIter,tempI/WARP_SIZE, partition_sums[tempI/WARP_SIZE] );
			// 	}
			// 	printf("prob::%f,%f\n",prob,dev_rnd[2*centerIter]);			
			// 	assert(partitionNum != -1);
			// }
			
			end 		= min((partitionNum + 1)*per_thread, numPoints) - 1;
			dataIndex 	+= partitionNum*per_thread;
			prob 		= dev_rnd[2*centerIter + 1]*dist_scanned[end];
			if( prob < dist_scanned[dataIndex])
			{
				if( dataIndex%per_thread == 0 )
				{
					// pointIndex = dataIndex;
					for (int j = 0; j < DIMENSION; ++j)
				    {
				    	dev_multiset[j] = dev_data[dataIndex*DIMENSION + j];
				    }
				}
				else if (prob > dist_scanned[dataIndex-1])
				{
					// pointIndex = dataIndex;
					for (int j = 0; j < DIMENSION; ++j)
				    {
				    	dev_multiset[j] = dev_data[dataIndex*DIMENSION + j];
				    }
				}
			}
		}
		__syncthreads(); // Needed as l2_center needs to be updated before computing distances in for finding next center
	}


	// Now find the largest partition
	/////////////////////////////////////////////////////////////////////////////////////////////
	float *dev_multiset = dev_data;
	int multisetSize = numPoints;
	float* dev_l2_centers = dev_centers;

	// __global__ void mean_heuristic_assign_gpu(float* dev_multiset,int multisetSize, float* dev_l2_centers, float* dev_l2_centers_temp)
	// {
	
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
		// min_dist = distance(dev_l2_centers,multiset + i*DIMENSION);
		float min_dist  = 0,temp = 0;
		int centerIndex = 0;
		for (int k = 0; k < DIMENSION; ++k)
		{
			temp  = dev_multiset[pointIndex*DIMENSION + k] - dev_l2_centers[0*DIMENSION +  k];
			min_dist += temp*temp;
		}
		for (int j = 1; j < NUM_CLUSTER; j++) 
		{
			// tmp_dist = distance(dev_l2_centers + j*DIMENSION, multiset+ i*DIMENSION);
			float tmp_dist = 0;
			for (int k = 0; k < DIMENSION; ++k)
			{
				temp = dev_multiset[pointIndex*DIMENSION + k] - dev_l2_centers[j*DIMENSION + k] ;
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
			atomicAdd(cluster_means + centerIndex*DIMENSION + j, dev_multiset[pointIndex*DIMENSION + j]); 
		}
		// counts[centerIndex]++;
		atomicAdd(counts + 	centerIndex ,1);
		pointIndex += blockDim.x;
	}
	
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
	/////////////////////////////////////////////////////////////////////////////////////////////
}

// Designed for mean_heuristic part to work on sampled multiset
__global__ void comp_dist_package_with_loop(float* dev_data, float* dev_centers,int numPoints,float *dev_rnd)
{
	// In this form of things we can probabl do away with distance array and just keep scanned_dist array
	// Just need one more var per thread to achieve this
	float distance = 0;
	__shared__ int tempSample[1];
	// __shared__ float temp[WARP_SIZE*WARP_SIZE];// Needed to be made dynamic for different datasets
	__shared__ float dist_scanned[WARP_SIZE*WARP_SIZE];// Needed to be made dynamic for different datasets
	__shared__ float partition_sums[WARP_SIZE];// Needed to be made dynamic for different datasets

	if(threadIdx.x  == 0)
	{
		tempSample[0] = -1;
		int tempIndex = dev_rnd[0]*numPoints;
		for (int i = 0; i < DIMENSION; ++i)
		{
			// assert(tempIndex*DIMENSION + i < numPoints*DIMENSION);
			dev_centers[i] = dev_data[tempIndex*DIMENSION + i];
		}
	}
	for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	{
		int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;		
		if(dataIndex < numPoints) // Rest of the threads will be idle!
		{
			if (centerIter == 1) // This is the first time dev_distances will get its values --> Can take this out of loop
			{
				float min_dist 	= 0, temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[j]; // Computing cost wrt 1st center
					min_dist += temp*temp;
				}
				distance = min_dist*min_dist;
			}
			else
			{
				int i = centerIter - 1; // i denotes the last center that was added to the list of centers
				float local_dist 	= 0,temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
					local_dist += temp*temp;
				}
				distance = min(distance , local_dist*local_dist);
			}		
		}
		
		// __syncthreads();//-- Not needed here, the warp which has finished its cost computation work is the one which
		// goes ahead and scans its part of the data array
	
		// int startIndex 	= WARP_SIZE*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
		// dist_scanned[thid] 	= dev_distances[startIndex + thid]; // load input into shared memory
		dist_scanned[dataIndex] 	= distance; // load input into shared memory

		if((dataIndex % WARP_SIZE) >= 1)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-1] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 2)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-2] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 4)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-4] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 8)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-8] + dist_scanned[dataIndex];
		if((dataIndex % WARP_SIZE) >= 16)
			dist_scanned[dataIndex] = dist_scanned[dataIndex-16] + dist_scanned[dataIndex];

		// dist_scanned[dataIndex] = dist_scanned[dataIndex];

		if((dataIndex+ 1)%WARP_SIZE == 0) // For 1 block and many threads
			partition_sums[(dataIndex)/WARP_SIZE] = dist_scanned[dataIndex];

		__syncthreads(); // Needed as partition_sums need to be computed before it can be made cumulative

		
		if (dataIndex < WARP_SIZE) // Just need 1 warp to perform  scan on partition_sums
		{
			if(dataIndex >= 1)
				partition_sums[dataIndex] = partition_sums[dataIndex-1] + partition_sums[dataIndex];
			if(dataIndex >= 2)
				partition_sums[dataIndex] = partition_sums[dataIndex-2] + partition_sums[dataIndex];
			if(dataIndex >= 4)
				partition_sums[dataIndex] = partition_sums[dataIndex-4] + partition_sums[dataIndex];
			if(dataIndex >= 8)
				partition_sums[dataIndex] = partition_sums[dataIndex-8] + partition_sums[dataIndex];
			if(dataIndex >= 16)
				partition_sums[dataIndex] = partition_sums[dataIndex-16] + partition_sums[dataIndex];

			// Use an entire WARP to sample faster
			// This works when each segment of distance array has size = WARP_SIZE and 
			// partition_sums also has size = WARP_SIZE. This is okay for Birch datasets as NUM_CLUSTER*10 = 1000
			// which is rounded to 32*32. Need to make sure this is modified/adapted for other datasets

			// assert(centerIter < NUM_CLUSTER);
			float* dev_multiset = dev_centers + centerIter*DIMENSION; 
			// first pick a block from the local_sums distribution
			int per_thread = WARP_SIZE;
			int partitionNum,end;
			float prob 	= dev_rnd[2*centerIter]*partition_sums[per_thread-1];
			if( prob <= partition_sums[dataIndex])
			{
				if(dataIndex == 0)
				{
					// partitionNum  = 0;
					tempSample[0] = 0;
				}
				else if( partition_sums[dataIndex-1] < prob )
				{
					tempSample[0] = dataIndex; // Important to do the update in a shared var so that other threads can also see it
					// partitionNum = dataIndex;
				}
			}
			// the start and end index of this block
			// int startIndex 	= partitionNum*per_thread;
			// int endIndex 	= min((partitionNum + 1)*per_thread, NUM_POINTS);
			// now sample from the cumulative distribution of the block
			partitionNum 	= tempSample[0];
			end 		= min((partitionNum + 1)*per_thread, numPoints) - 1;//Can avoid this min here as dist_scanned is padded to have correct value at the lastindex as well
			dataIndex 	+= partitionNum*per_thread;
			prob 		= dev_rnd[2*centerIter + 1]*dist_scanned[end];

			if( prob < dist_scanned[dataIndex])
			{
				if( dataIndex%per_thread == 0 )
				{
					// pointIndex = dataIndex;
					for (int j = 0; j < DIMENSION; ++j)
				    {
				    	dev_multiset[j] = dev_data[dataIndex*DIMENSION + j];
				    }
				}
				else if (prob > dist_scanned[dataIndex-1])
				{
					// pointIndex = dataIndex;
					for (int j = 0; j < DIMENSION; ++j)
				    {
				    	dev_multiset[j] = dev_data[dataIndex*DIMENSION + j];
				    }
				}
			}
		}
		__syncthreads(); // Needed as l2_center needs to be updated before computing distances in for finding next center
	}
}
// THis version was meant to run for multiple blocks but then there can not be synchronization between thread blocks
// so this implementation resulted in erroneous execution
// Must be called with nBlocks,WARP_SIZE configuration, and make sure that nBlocks*WARP_SIZE >= numPoints
__global__ void comp_dist_package_with_loop_multipleBlocks(float* dev_data,float* dev_distances,float* dev_distances_scanned, float* dev_partition_sums,float* dev_partition_sums_scanned, float* dev_centers,int numPoints,float *dev_rnd)
{

	// In this form of things we can probabl do away with distance array and just keep scanned_dist array
	// Just need one more var per thread to achieve this
	// for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	{
		int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;		
		if(dataIndex < numPoints) // Rest of the threads will be idle!
		{
			if (centerIter == 1) // This is the first time dev_distances will get its values --> Can take this out of loop
			{
				float min_dist 	= 0, temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[j]; // Computing cost wrt 1st center
					min_dist += temp*temp;
				}
				dev_distances[dataIndex] = min_dist*min_dist;
			}
			else
			{
				// float local_dist 	= 0,temp;
				// float min_dist = FLT_MAX;
				// for (int i = 0; i < centerIter; ++i)
				// {
				// 	for (int j = 0; j < DIMENSION; ++j)
				// 	{
				// 		temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
				// 		local_dist += temp*temp;
				// 	}
				// 	min_dist = min(min_dist , local_dist*local_dist);
				// }
				// dev_distances[dataIndex] 		= min_dist; // No need to square min_dist here, it is already squared value

				int i = centerIter - 1; // i denotes the last center that was added to the list of centers
				float local_dist 	= 0,temp;
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
					local_dist += temp*temp;
				}
				float min_dist = min(dev_distances[dataIndex] , local_dist*local_dist);
				dev_distances[dataIndex] 		= min_dist; // No need to square min_dist here, it is already squared value
			}		
		}
		else
		{
			dev_distances[dataIndex]  = 0;  // Can do this once via memset as well, if that improves performance
		}
		
		__shared__ float temp[WARP_SIZE];
		int thid 		= threadIdx.x;  // threadId modulo threadBlock
		int startIndex 	= WARP_SIZE*blockIdx.x; // Each thread-block gets to scan an array of size n, with startIndex as computed
		
		temp[thid] 	= dev_distances[startIndex + thid]; // load input into shared memory

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

		dev_distances_scanned[startIndex + thid] = temp[thid];
		if(thid == WARP_SIZE - 1)
			dev_partition_sums[blockIdx.x] = temp[thid]; 

		if (blockIdx.x == 0)
		{
			temp[thid] 	= dev_partition_sums[thid]; // load input into shared memory

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

			dev_partition_sums_scanned[thid] = temp[thid];  // ??? can sync_threads solve the problem here because of which i needed to use _scanned for part sums
		}

		// This entire function takes around 55 units and this samling part  alone takes around 40 units out of it
		if ((blockIdx.x == 0) && (threadIdx.x == 0))
		{
			float* dev_multiset = dev_centers + centerIter*DIMENSION; 

			int per_thread = WARP_SIZE;
			int numValidPartitions = per_thread;
			int start,mid,end,groupNo,pointIndex;
			float prob;

			// first pick a block from the local_sums distribution
			// int groupNo =sample_from_distribution(partition_sums,0, numValidPartitions, rnd[2*tid]*partition_sums[numValidPartitions-1]);

			start 	= 0;
			end 	= numValidPartitions - 1;
		    prob 	= dev_rnd[2*centerIter]*dev_partition_sums_scanned[end];
		    while(start <= end) 
		    {
		        mid = (start+end)/2;
		        if(prob < dev_partition_sums_scanned[mid-1]) 
		        {
		            end = mid-1;
		        } 
		        else if(prob > dev_partition_sums_scanned[mid]) 
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
			end 	= min((groupNo + 1)*per_thread, numPoints) - 1;
		    prob 	= dev_rnd[2*centerIter + 1]*dev_distances_scanned[end];
		    while(start <= end) 
		    {
		        mid = (start+end)/2;
		        if(prob < dev_distances_scanned[mid-1]) 
		        {
		            end = mid-1;
		        } 
		        else if(prob > dev_distances_scanned[mid]) 
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
		    	dev_multiset[j] = dev_data[pointIndex*DIMENSION + j];
		    }
		}
	}
	
}
// This is the original version , just backing it up
__global__ void comp_dist_package_with_loop_original(float* dev_data,float* dev_distances,float* dev_partition_sums, float* dev_centers,int numPoints,float *dev_rnd)
{
	for (int centerIter = 1; centerIter < NUM_CLUSTER; ++centerIter)
	{
		// Starting off with very simplistic 1-D threads blocks and 1-D grids
		int numGPUThreads = gridDim.x*blockDim.x;
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
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
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
				for (int j = 0; j < DIMENSION; ++j)
				{
					temp = dev_data[dataIndex*DIMENSION + j] - dev_centers[i*DIMENSION + j];
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
// This one does not use the cumulative distance array to get costs computed in last iteration,
// instead it makes another copy of distance array which is cumulative, for sampling purposes
float* d2_sample_4(float* data,float* centers,int numPts, int numSamples, int size, float* distances)
{
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
	float* distances_cumulative 	= (float*)malloc(numPts*sizeof(float));
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
        int i;
        for(i = 0;i < block_size;i++)
        {    
            if(center_size == 0)
            {
                local_sum += 1;
                distances_cumulative[lower+i] = 1 + prev_val;
            } 
            else if (center_size == 1)
            {
                p = data + (lower+i)*DIMENSION;
                min_dist = distance(p,centers);
                min_dist = min_dist*min_dist;
                local_sum +=  min_dist;
                distances[lower+i] =  min_dist; // make cumulative 
                distances_cumulative[lower+i] =  min_dist + prev_val; // make cumulative 
            }
            else
            {
            	p = data + (lower+i)*DIMENSION;
                min_dist   		= distances[lower+i];
                local_dist 		= distance(p,centers + (center_size-1)*DIMENSION); // Find distance wrt last added new center;
                local_dist 		= local_dist*local_dist;
                min_dist 		= min(min_dist,local_dist);
                local_sum 		+=  min_dist; // min_dist is already squared here because it is calculated usign cumulative distance
                distances[lower+i] = min_dist;
                distances_cumulative[lower+i] = min_dist + prev_val; // make cumulative 
            	prev_val = distances_cumulative[lower+i];
            }
            prev_val = distances_cumulative[lower+i];
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
            int pointIndex = sample_from_distribution(distances_cumulative, startIndex, endIndex, rnd[2*i+1]*distances_cumulative[endIndex-1]);
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
    free(distances_cumulative);
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

// This version of d2_sample has optimized cost calculation by using cost computed in last iteration
// This is specially optimized for mean_heurisitic
// This one does not make distance array cumulative in-place, instead a separate array is used to to get cumulative distances
float* d2_sample_5(float* data,float* centers,int numPts, int size, float* distances)
{
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
	float* distances_cumulative 	= (float*)malloc(numPts*sizeof(float));
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
        float prev_val = 0;
        // cost of each block
        float local_sum = 0;
        int center_size = size;
        int i;
        for(i = 0;i < block_size;i++)
        {    
            if(center_size == 0)
            {
                local_sum += 1;
                distances_cumulative[lower+i] = 1 + prev_val;
            } 
            else if (center_size == 1)
            {
                p = data + (lower+i)*DIMENSION;
                min_dist = distance(p,centers);
                min_dist = min_dist*min_dist;
                local_sum +=  min_dist;
                distances[lower+i] =  min_dist; // make cumulative 
                distances_cumulative[lower+i] = min_dist + prev_val; // make cumulative 
            }
            else
            {
            	p = data + (lower+i)*DIMENSION;
                min_dist   		= distances[lower+i];
                local_dist 		= distance(p,centers + (center_size-1)*DIMENSION); // Find distance wrt last added new center;
                local_dist 		= local_dist*local_dist;
                min_dist 		= min(min_dist,local_dist);
                
                local_sum 		+=  min_dist; // min_dist is already squared here because it is calculated usign cumulative distance
                distances[lower+i] = min_dist; // make cumulative 
                distances_cumulative[lower+i] = min_dist + prev_val; // make cumulative 
            }
            prev_val = distances_cumulative[lower+i];
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
	int pointIndex = sample_from_distribution(distances_cumulative, startIndex, endIndex, rnd[1]*distances_cumulative[endIndex-1]);
	for (int j = 0; j < DIMENSION; ++j)
	{
		result[j] = data[pointIndex*DIMENSION + j];
    }
    free(local_sums);
    free(distances_cumulative);
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
		// float* point = d2_sample_3(multiset,level_2_sample,multisetSize,i,distances);
		float* point = d2_sample_5(multiset,level_2_sample,multisetSize,i,distances); // Better for avoiding errors due to cost compute optimizations
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
			 temp  = dev_multiset[pointIndex*DIMENSION + k] - dev_l2_samples[0*DIMENSION +  k];
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
			// cluster_means[centerIndex*DIMENSION + j] += dev_multiset[pointIndex*DIMENSION + j]; // Need to make sure this is atomic
			
			atomicAdd(cluster_means + centerIndex*DIMENSION + j, dev_multiset[pointIndex*DIMENSION + j]); // Need to make sure this is atomic
		}
		// counts[centerIndex]++;
		atomicAdd(counts + 	centerIndex ,1);
		pointIndex += blockDim.x;
    }
	
	////// Finds max count but need to find index for maxCount
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
	/////////////////////////////////////////////////////////////////////////////////

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
	// printf("tid::%d\n",tid );
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
