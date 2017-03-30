#include "main.h"

// g++ -D BIRCH1 -g mainOMP.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch1 -lm

// random_device rd;
// mt19937 gen(rd());
// unsigned int max_val = gen.max();
int numThreads = 1;
__constant__ double dev_centers_global[NUM_CLUSTER*DIMENSION]; // For using constant memory
int main(int argc, char const *argv[])
{
	// Currently no argument processing logic, will always run birch1 for 2 times with N=10k
	srand(time(NULL));
		int numRuns,method;
		int N = 0;

		// for k-means parallel
		int rounds = 5;
		double oversampling = NUM_CLUSTER;
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
		double initTime[numRuns];
		double iterTime[numRuns];
		double totalTime[numRuns];
		double initCost[numRuns];
		double finalCost[numRuns];
		double numIter[numRuns];

		// read the data into a vector of "vector"

		double* data;
		FILE* reader;
		int i = 0,j = 0;
		data 	= (double*)malloc(NUM_POINTS*DIMENSION*sizeof(double));
		reader 	= fopen(dataFileName,"r");
		while(i < NUM_POINTS)
		{
			j = 0;
			while(j < DIMENSION)
			{
				fscanf(reader,"\t%lf",&(data[i*DIMENSION + j]));
				j++;
			}
			i++;
		}

	// Copy data onto device memory
	double* dev_data;
	cudaMalloc((void**)&dev_data,DIMENSION*NUM_POINTS*sizeof(double));
	cudaMemcpy(dev_data,data,DIMENSION*NUM_POINTS*sizeof(double),cudaMemcpyHostToDevice);

	FILE* logger;
	int runNum;
	for(runNum = 0; runNum < numRuns ; runNum++)
	{
		double samplingTime_1[NUM_CLUSTER];
		double samplingTime_2[NUM_CLUSTER];
		printf("Running runNum::%d\n",runNum );
		gettimeofday(&start,NULL);

		int numBlocks 			= 8;
		int numThreadsPerBlock 	= 1024;
		int numSampleBlocks 	= 128;
		int numSampleTperB 		= 32;
		int numGPUThreads 		= numBlocks*numThreadsPerBlock;

		// double* distances_debug	= (double*)malloc(NUM_POINTS*sizeof(double));
		double* distances; // Using page-locked memory for distances
		cudaHostAlloc((void**)&distances,NUM_POINTS*sizeof(double),cudaHostAllocDefault);
		double* centers 		= (double*)malloc(NUM_CLUSTER*DIMENSION*sizeof(double));
		double* rnd 			= (double*)malloc(2*N*sizeof(double));
		double* multiset    	= (double*)malloc(N*DIMENSION*sizeof(double));
		double* partition_sums 	= (double*)malloc(numGPUThreads*sizeof(double));
		// double* partition_sums_debug 	= (double*)malloc(numGPUThreads*sizeof(double));
		int* 	sampled_indices = (int*)malloc(N*sizeof(int));

		double* dev_distances;
		double* dev_partition_sums;
		double* dev_rnd;
		int* 	dev_sampled_indices;

		// double* dev_centers; // When not using constant memory for centers
		checkCudaErrors(cudaMalloc((void**)&dev_distances,NUM_POINTS*sizeof(double)));
		checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,numGPUThreads*sizeof(double)));
		checkCudaErrors(cudaMalloc((void**)&dev_sampled_indices,N*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&dev_rnd,2*N*sizeof(double)));
		// checkCudaErrors(cudaMalloc((void**)&dev_centers,NUM_CLUSTER*DIMENSION*sizeof(double))); // No need when using constant memory

		// initialize the initial centers
		if(method == 2) // d2-seeding
		{  
			// ---------------------- GPU-Based Implementation Start ------------------------------------
			cudaProfilerStart();
			
			// First choosing the first point uniformly at random, no need to sample N points and all here
			int tempPointIndex 	= (((double) rand())/RAND_MAX)*NUM_POINTS;
			memcpy(centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(double));
			checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(double),0,cudaMemcpyHostToDevice));
			// checkCudaErrors(cudaMemcpy(dev_centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(double),cudaMemcpyHostToDevice));

			double compDistTime = 0, makeCumulativeTime = 0, samplingTime = 0, meanHeuristicTime = 0;
			for(i = 1; i < NUM_CLUSTER; i++)
			{
				struct timeval sample_start,sample_end;
				gettimeofday(&sample_start,NULL);
				for(j = 0; j < N; ++j)
				{
					rnd[2*j] 	= ((double) rand())/RAND_MAX;
					rnd[2*j+1] 	= ((double) rand())/RAND_MAX;
				}
				cudaMemcpy(dev_rnd,rnd,2*N*sizeof(double),cudaMemcpyHostToDevice);// Can be overlapped with computation
				// comp_dist<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, dev_centers, i, NUM_POINTS, DIMENSION, numGPUThreads);
				
				// For blocked access pattern
					// comp_dist_glbl<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, i, NUM_POINTS, DIMENSION, numGPUThreads);
					// cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(double),cudaMemcpyDeviceToHost);	
					// for (j = 1; j < numGPUThreads; ++j) // Need to do this scan operation on GPU only, but testing things first
					// {
					// 	partition_sums[j] += partition_sums[j-1];
					// }
					// cudaMemcpy(dev_partition_sums,partition_sums,numGPUThreads*sizeof(double),cudaMemcpyHostToDevice);

					// int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;
					// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances, dev_sampled_indices, dev_rnd, per_thread, NUM_POINTS, N);

				// For strided memory access pattern
					comp_dist_glbl_strided<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, i, NUM_POINTS, DIMENSION, numGPUThreads);
					
					cudaMemcpy(distances,dev_distances,NUM_POINTS*sizeof(double),cudaMemcpyDeviceToHost);
					for (j = 1; j < NUM_POINTS; ++j)
					{
						distances[j] += distances[j-1];
					}
					cudaMemcpy(dev_distances,distances,NUM_POINTS*sizeof(double),cudaMemcpyHostToDevice);
					sample_from_distribution_gpu_strided<<<numSampleBlocks,numSampleTperB>>>(dev_distances, dev_sampled_indices, dev_rnd, NUM_POINTS, N);
					
					// // Division of distance array into blocks so that sampling is similar to blocked cost calculation approach
					// int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;
					// cudaMemcpy(distances,dev_distances,NUM_POINTS*sizeof(double),cudaMemcpyDeviceToHost);
					// double prev_val = distances[0],prev_part_val=0;
					// int p_ctr = 0;
					// for (j = 1; j < NUM_POINTS; ++j)
					// {
					// 	distances[j] 	+= prev_val;
					// 	prev_val 		= distances[j];
					// 	if ((j+1)%per_thread == 0)
					// 	{
					// 		partition_sums[p_ctr] = distances[j] + prev_part_val;
					// 		prev_part_val = partition_sums[p_ctr];
					// 		p_ctr += 1;
					// 		prev_val = 0;
							
					// 	}
					// 	else if (j == NUM_POINTS -1)
					// 	{
					// 		partition_sums[p_ctr] = distances[j] + prev_part_val;
					// 		prev_part_val = partition_sums[p_ctr];
					// 		p_ctr += 1;
					// 		prev_val = 0;
					// 	}
					// }
					// cudaMemcpy(dev_distances,distances,NUM_POINTS*sizeof(double),cudaMemcpyHostToDevice);
					// cudaMemcpy(dev_partition_sums,partition_sums,numGPUThreads*sizeof(double),cudaMemcpyHostToDevice);
					// sample_from_distribution_gpu<<<numSampleBlocks,numSampleTperB>>>(dev_partition_sums, dev_distances, dev_sampled_indices, dev_rnd, per_thread, NUM_POINTS, N);

				// Copy back indices of sampled points, no need to copy those points as we have the data here as well
				cudaMemcpy(sampled_indices,dev_sampled_indices,N*sizeof(int),cudaMemcpyDeviceToHost);
				for (int copy_i = 0; copy_i < N; ++copy_i)
				{
					int index = sampled_indices[copy_i];
					for (int copy_j = 0; copy_j < DIMENSION; ++copy_j)
					{
						multiset[copy_i*DIMENSION + copy_j] = data[index*DIMENSION + copy_j];
					}
				}
				gettimeofday(&sample_end,NULL);
				compDistTime += get_time_diff(sample_start,sample_end);
				
				// Code for sampling on CPU (first GPU implementation)
					// // copy back to host memory for sampling purpose, 
					// cudaMemcpy(distances,dev_distances,NUM_POINTS*sizeof(double),cudaMemcpyDeviceToHost);
					// cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(double),cudaMemcpyDeviceToHost);	

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
					// 	rnd[2*j] 	= ((double) rand())/RAND_MAX;
					// 	rnd[2*j+1] 	= ((double) rand())/RAND_MAX;

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

				gettimeofday(&sample_start,NULL);
				double* nextCenter = mean_heuristic(multiset,N);
				memcpy(centers + i*DIMENSION,nextCenter,DIMENSION*sizeof(double));
				checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(double), i*DIMENSION*sizeof(double), cudaMemcpyHostToDevice));
				// checkCudaErrors(cudaMemcpy(dev_centers + i*DIMENSION , nextCenter, DIMENSION*sizeof(double), cudaMemcpyHostToDevice));
				gettimeofday(&sample_end,NULL);
				meanHeuristicTime += get_time_diff(sample_start,sample_end);
				
			}
			printf("compDistTime\t\t%2.5f\t%2.5f\n",compDistTime,compDistTime/(NUM_CLUSTER-1) );
			printf("makeCumulativeTime\t%2.5f\t%2.5f\n",makeCumulativeTime,makeCumulativeTime/(NUM_CLUSTER-1) );
			printf("samplingTime\t\t%2.5f\t%2.5f\n",samplingTime,samplingTime/(NUM_CLUSTER-1) );
			printf("meanHeuristicTime\t%2.5f\t%2.5f\n",meanHeuristicTime,meanHeuristicTime/(NUM_CLUSTER-1) );
			cudaProfilerStop();
			// ---------------------- GPU-Based Implementation End --------------------------------------
			
			// ---------------------- CPU-Based Implementation Start ------------------------------------
				// for(i = 0; i < NUM_CLUSTER; i++)
				// {
				// 	struct timeval sample_start,sample_end;
				// 	gettimeofday(&sample_start,NULL);
				// 	// multiset = d2_sample(data,centers,NUM_POINTS,N,i);
				// 	multiset = d2_sample_2(data,centers,NUM_POINTS,N,i,distances);
				// 	gettimeofday(&sample_end,NULL);
				// 	printf("Time taken for d2_sample::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
				// 	samplingTime_1[i] = get_time_diff(sample_start,sample_end);
				// 	gettimeofday(&sample_start,NULL);
				// 	double* nextCenter = mean_heuristic(multiset,N);
				// 	for (int j = 0; j < DIMENSION; ++j)
				// 	{
				// 		centers[i*DIMENSION + j] = nextCenter[j];
				// 	}
				// 	gettimeofday(&sample_end,NULL);
				// 	printf("Time taken for mean_heuristic::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
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
		double prev_cost = DBL_MAX;
		int iteration = 0;

		char tempFileName[100];
		sprintf(tempFileName,"%s%d.txt",baseLogFile,runNum);
		logger = fopen(tempFileName,"w");

		// Can make first two static arrays
		int* cluster_counts 	= (int*)malloc(NUM_CLUSTER*sizeof(int)); // number of points assigned to each cluster
		double* cluster_sums	= (double*)malloc(DIMENSION*NUM_CLUSTER*sizeof(double)); // sum of points assigned to each cluster
		int** cluster_counts_pointers 	= (int**)malloc(numThreads*sizeof(int*)); // pointers to local "number of points assigned to each cluster"
		double** cluster_sums_pointers 	= (double**)malloc(numThreads*sizeof(double*)); // pointers to local "sum of points assigned to each cluster"
		
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
			double current_cost = 0.0;
    		#pragma omp parallel reduction(+: current_cost) 
    		{
    			int tid = omp_get_thread_num();
    			int local_cluster_counts[NUM_CLUSTER]; // local "number of points assigned to each cluster"
    			double local_cluster_sums[DIMENSION*NUM_CLUSTER]; // local "sum of points assigned to each cluster"
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
    			double min_dist;
    			double current_dist;
    			// assign each point to their cluster center in parallel. 
    			// update the cost of current solution and keep updating local counts and sums
    			#pragma omp for schedule(static)
    			for (int i = 0; i < NUM_POINTS; i++) 
    			{
    				index = 0;
    				min_dist = DBL_MAX;
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
		cudaFreeHost(distances); // free this way when using page-locked memory for distances
		// free(distances);
		free(rnd);
		free(multiset);
		free(partition_sums);
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

int sample_from_distribution(double* probabilities, int startIndex, int endIndex, double prob) 
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
__global__ void sample_from_distribution_gpu(double* dev_partition_sums, double* dev_distances, int* dev_sampled_indices, double* dev_rnd,int per_thread, int dev_NUM_POINTS, int dev_N) 
{

	int numValidPartitions = dev_NUM_POINTS/per_thread + 1;
	int start,mid,end,groupNo,pointIndex;
	double prob;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if( i < dev_N)
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

// Sampling for case of strided memory access pattern, no dev_partition_sums here
__global__ void sample_from_distribution_gpu_strided(double* dev_distances, int* dev_sampled_indices, double* dev_rnd, int dev_NUM_POINTS, int dev_N) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if( i < dev_N)
	{
		int start,mid,end,pointIndex;
		double prob;
		
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

// This function calcuates required distance for all points and partitions
// Need to do an all-prefix sum after this to make this thing cumulative
// Can be optimized by using distances calculated in previous iteration, i.e. when the previous center was sampled
// This does not do any sampling business
// Need not call this function when centerIter = 0,
// Not optimized to use distance calculted in previous iteration to calculate distance/cost for points 
__global__ void comp_dist_2(double* dev_data,double* dev_distances,double* dev_partition_sums, double* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// int jump = blockDim.x*gridDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	double min_dist = DBL_MAX, local_dist,temp,prev_val = 0;
	for (int dataIndex = startIndex; dataIndex < endIndex; ++dataIndex)
	{
		min_dist = DBL_MAX;
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
__global__ void comp_dist(double* dev_data,double* dev_distances,double* dev_partition_sums, double* dev_centers,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	double min_dist = DBL_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
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


// Optimised to use previous distance values to calculate min_dist for points in next iteration
// Also makes use of constant memory for storing centers
__global__ void comp_dist_glbl(double* dev_data,double* dev_distances,double* dev_partition_sums,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	// Starting off with very simplistic 1-D threads blocks and 1-D grids
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int per_thread = (numPoints + numGPUThreads - 1)/numGPUThreads;// Term in the numerator is added to that we can get ceiling of numPoints/numGPUThreads
	int startIndex = tid*per_thread;
	int endIndex = min((tid + 1)*per_thread,numPoints);
	double min_dist = DBL_MAX, local_dist,temp,prev_val = 0,old_prev_val=0;
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
__global__ void comp_dist_glbl_strided(double* dev_data,double* dev_distances,double* dev_partition_sums,int centerIter,int numPoints,int dev_dimension,int numGPUThreads)
{
	int dataIndex 	= threadIdx.x + blockIdx.x*blockDim.x;
	int stride 		= blockDim.x*gridDim.x;
	double min_dist, local_dist, temp;
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
			// if (dataIndex == 0)
			// {
			// 	min_dist 	= dev_distances[dataIndex];
			// }
			// else
			// {
			// 	min_dist 	= dev_distances[dataIndex] - dev_distances[dataIndex - 1];
			// }
			min_dist = DBL_MAX;
			for (int i = 0; i < centerIter; ++i)
			{
				local_dist 	= 0;
				for (int j = 0; j < dev_dimension; ++j)
				{
					temp = dev_data[dataIndex*dev_dimension + j] - dev_centers_global[i*dev_dimension + j];
					local_dist += temp*temp;
				}
				min_dist = min(min_dist,local_dist*local_dist);
			}
			dev_distances[dataIndex] = min_dist;  // --No-- Need to square min_dist here, it is *not*  already squared value
		}
		dataIndex += stride;
	}
}

// generate numSamples sized multiset from weighted data with weights wrt. centers where the current size of centers is size
// numPts : number of points in data
// numSamples: number of points to sample
// size : size of centers i.e. number of centers chosen already
double* d2_sample(double* data,double* centers,int numPts, int numSamples, int size)
{
	
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
	double* distances 	= (double*)malloc(numPts*sizeof(double));
    double* local_sums	= (double*)malloc(numThreads*sizeof(double));   // local sums. first is sum for [0...numPts/numThreads-1], and so on. This is also a cumulative distribution.
    double* result 		= (double*)malloc(numSamples*DIMENSION*sizeof(double));
    for (int i = 0; i < numSamples; ++i)
    {
    	for (int j = 0; j < DIMENSION; ++j)
    	{
    		result[i*DIMENSION + j] = 0;
    	}
    }
    // we're gonna need 2*numSamples random numbers. 
    double* rnd 		= (double*)malloc(2*numSamples*sizeof(double));
    int i;
	for(i = 0; i < 2*numSamples; i++){
		rnd[i] = ((double) rand())/RAND_MAX;
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
        double min_dist, local_dist;
        double* p;
        double prev_val = 0;
        // cost of each block
        double local_sum = 0;
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
            // memcpy(result + i*DIMENSION, data + pointIndex*DIMENSION, DIMENSION*sizeof(double));
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
double* d2_sample_2(double* data,double* centers,int numPts, int numSamples, int size, double* distances)
{
	
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,numPts/numThreads], [numPts/numThreads+1,...,numPts*2/numThreads],... and so on. These groups contain cumulative distances.
    double* local_sums	= (double*)malloc(numThreads*sizeof(double));   // local sums. first is sum for [0...numPts/numThreads-1], and so on. This is also a cumulative distribution.
    double* result 		= (double*)malloc(numSamples*DIMENSION*sizeof(double));
    for (int i = 0; i < numSamples; ++i)
    {
    	for (int j = 0; j < DIMENSION; ++j)
    	{
    		result[i*DIMENSION + j] = 0;
    	}
    }
    // we're gonna need 2*numSamples random numbers. 
    double* rnd 		= (double*)malloc(2*numSamples*sizeof(double));
    int i;
	for(i = 0; i < 2*numSamples; i++){
		rnd[i] = ((double) rand())/RAND_MAX;
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
        double min_dist, local_dist;
        double* p;
        double prev_val = 0, old_prev_val = 0;
        // cost of each block
        double local_sum = 0;
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
            // memcpy(result + i*DIMENSION, data + pointIndex*DIMENSION, DIMENSION*sizeof(double));
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

double* mean_heuristic(double* multiset,int multisetSize)
{
	// first do a kmeans++ initialiation on the multiset
	int i,j;
	// gettimeofday(&start,NULL);
	double* level_2_sample = (double*)malloc(NUM_CLUSTER*DIMENSION*sizeof(double));
	for(i = 0; i < NUM_CLUSTER; i++)
	{
		double* point = d2_sample(multiset,level_2_sample,multisetSize,1,i);
		for (j = 0; j < DIMENSION; ++j)
		{
			level_2_sample[i*DIMENSION + j] = point[j];
		}
		// memcpy(level_2_sample + i*DIMENSION, point, DIMENSION*sizeof(double)) ;
	}

	// gettimeofday(&end,NULL);
	// printf("Time taken to choose k centers::%f\n",get_time_diff(start,end));
	// gettimeofday(&start,NULL);
	int* counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int)); // number of points assigned to each kmeans++ center
    double* cluster_means 	= (double*)malloc(NUM_CLUSTER*DIMENSION*sizeof(double)); // for taking the centroid later on. We maintain a sum of all points assigned to a center here.
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
    double** local_tmp_cluster_means 	= (double**)malloc(numThreads*sizeof(double*));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int* local_counts 			= (int*)malloc(NUM_CLUSTER*sizeof(int));
        double* local_cluster_means = (double*)malloc(NUM_CLUSTER*DIMENSION*sizeof(double));
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
        double min_dist, tmp_dist;
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

double distance(double* p1, double* p2)
{
	int i;
	double localSum = 0;
	for (i = 0; i < DIMENSION; ++i)
	{
		localSum += (p1[i] - p2[i])*(p1[i] - p2[i]);
	}
	return localSum;
}

void write_centers_to_file(double* centers)
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

static inline float mean(double* a, int n)
{
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	return sum/n;
}

static inline float sd(double* a, int n)
{
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	double mean = sum/n;
	sum = 0;
	for(int i = 0; i < n; i++){
		sum += (a[i] - mean) * (a[i] - mean);
	}
	return sqrt(sum/n);
}

static inline double get_time_diff(struct timeval t1, struct timeval t2){
	return t2.tv_sec - t1.tv_sec + 1e-6 * (t2.tv_usec - t1.tv_usec);
}
