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
		int numGPUThreads = numBlocks*numThreadsPerBlock;

		// double* distances 		= (double*)malloc(NUM_POINTS*sizeof(double));
		double* distances; // Using page-locked memory for distances
		cudaHostAlloc((void**)&distances,NUM_POINTS*sizeof(double),cudaHostAllocDefault);
		double* centers 		= (double*)malloc(NUM_CLUSTER*DIMENSION*sizeof(double));
		double* rnd 			= (double*)malloc(2*N*sizeof(double));
		double* multiset    	= (double*)malloc(N*DIMENSION*sizeof(double));
		double* partition_sums 	= (double*)malloc(numGPUThreads*sizeof(double));
		double* dev_distances;
		double* dev_partition_sums;
		// double* dev_centers;
		checkCudaErrors(cudaMalloc((void**)&dev_distances,NUM_POINTS*sizeof(double)));
		checkCudaErrors(cudaMalloc((void**)&dev_partition_sums,numGPUThreads*sizeof(double)));
		// checkCudaErrors(cudaMalloc((void**)&dev_centers,NUM_CLUSTER*DIMENSION*sizeof(double))); // No need when using constant memory
		
		// cudaEvent_t stopEvent;
		// cudaEventCreate( &stopEvent );

		// initialize the initial centers
		if(method == 2) // d2-seeding
		{  
			// all points have a weight of one. This is an unweighted kmeans++ problem
			// ---------------------- GPU-Based Implementation Start ------------------------------------
			cudaProfilerStart();
			// cudaStream_t stream_1, stream_2;
			// cudaStreamCreate( &stream_1 );
			// cudaStreamCreate( &stream_2 );
			
			// First choosing the first point uniformly at random, no need to sample N points and all here
			float tempRand = ((double) rand())/RAND_MAX;
			int tempPointIndex = tempRand*NUM_POINTS;
			checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(double),0,cudaMemcpyHostToDevice));
			// checkCudaErrors(cudaMemcpy(dev_centers, data+tempPointIndex*DIMENSION, DIMENSION*sizeof(double),cudaMemcpyHostToDevice));

			double compDistTime = 0, makeCumulativeTime = 0, samplingTime = 0, meanHeuristicTime = 0;
			for(i = 1; i < NUM_CLUSTER; i++)
			{
				struct timeval sample_start,sample_end;
				gettimeofday(&sample_start,NULL);
				// comp_dist<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, dev_centers, i, NUM_POINTS, DIMENSION, numGPUThreads);
				comp_dist_glbl<<<numBlocks,numThreadsPerBlock>>>(dev_data, dev_distances, dev_partition_sums, i, NUM_POINTS, DIMENSION, numGPUThreads);
				
				// copy back to host memory for sampling purpose
				cudaMemcpy(distances,dev_distances,NUM_POINTS*sizeof(double),cudaMemcpyDeviceToHost);
				cudaMemcpy(partition_sums,dev_partition_sums,numGPUThreads*sizeof(double),cudaMemcpyDeviceToHost);	
				// No need to copy back centers to host, this was done for debugging purposes only
				// cudaMemcpy(centers,dev_centers,NUM_CLUSTER*DIMENSION*sizeof(double),cudaMemcpyDeviceToHost); // for debugging purposes
				
				// cudaEventRecord( stopEvent,0);
				gettimeofday(&sample_end,NULL);
				// cudaEventSynchronize( stopEvent );
				// printf("Time taken for comp_dist::\t%d\t%f\n",i,get_time_diff(sample_start,sample_end));
				compDistTime += get_time_diff(sample_start,sample_end);
				// Make it cumulative for sampling purpose, can be done on GPU as well
				gettimeofday(&sample_start,NULL);
				for (j = 1; j < numGPUThreads; ++j)
				{
					partition_sums[j] += partition_sums[j-1];
				}
				gettimeofday(&sample_end,NULL);
				// printf("Time taken to make cumulative::\t%d\t%f\n",i,get_time_diff(sample_start,sample_end));
				makeCumulativeTime += get_time_diff(sample_start,sample_end);
				
				int per_thread = (NUM_POINTS + numGPUThreads-1)/numGPUThreads;
				// double tempSum = 0;
					// printf("numGPUThreads::%d\n",numGPUThreads);
					// printf("per_thread::%d\n",per_thread);
					// printf("per_thread::%d\n",NUM_POINTS/numGPUThreads);
					// for (j = (numGPUThreads-1)*per_thread; j < NUM_POINTS; ++j)
					// {
					// 	if(j % per_thread == 0)
					// 		tempSum = 0;

					// 	double local_dist = 0,min_dist = DBL_MAX;
					// 	for (int k = 0; k < i; ++k)
					// 	{
					// 		local_dist 	= distance(centers + k*DIMENSION, data + j*DIMENSION);
					// 		min_dist 	= min(min_dist,local_dist*local_dist);
					// 	}
					// 	tempSum += min_dist;
					// 	printf("%d\t%f\n",j,distances[j]);

					// 	// if((j+1) % per_thread == 0)
					// 	// {
					// 	// 	// printf("\t%d\t%f\t%f\n",j,tempSum,distances[j]);
					// 	// 	printf("%d\t%f\n",j,distances[j]);
					// 	// 	printf("%d\t%f\n\n",j/per_thread,partition_sums[j/per_thread] );						
					// 	// }
					// 	// else
					// 	// 	printf("%d\t%f\t%f\n",j,tempSum,distances[j]);
				// }

				gettimeofday(&sample_start,NULL);
				for(j = 0 ; j < N ; j++)
				{
					rnd[2*j] 	= ((double) rand())/RAND_MAX;
					rnd[2*j+1] 	= ((double) rand())/RAND_MAX;

					int numValidParitions = NUM_POINTS/per_thread + 1;
					// first pick a block from the local_sums distribution
					int groupNo = sample_from_distribution(partition_sums, 0, numValidParitions, rnd[2*j]*partition_sums[numValidParitions-1]);
					// the start and end index of this block
					int startIndex 	= groupNo * per_thread;
					int endIndex 	= (groupNo + 1) * per_thread;

					if(groupNo == numGPUThreads - 1) endIndex = NUM_POINTS;
					// now sample from the cumulative distribution of the block
					int pointIndex = sample_from_distribution(distances, startIndex, endIndex, rnd[2*j+1]*distances[endIndex-1]);
					for (int k = 0; k < DIMENSION; ++k)
					{
						multiset[j*DIMENSION + k] = data[pointIndex*DIMENSION + k];
					}
				}
				gettimeofday(&sample_end,NULL);
				// printf("Time taken for sampling::\t%d\t%f\n",i,get_time_diff(sample_start,sample_end));
				samplingTime += get_time_diff(sample_start,sample_end);

				gettimeofday(&sample_start,NULL);
				double* nextCenter = mean_heuristic(multiset,N);
				checkCudaErrors(cudaMemcpyToSymbol(dev_centers_global , nextCenter, DIMENSION*sizeof(double), i*DIMENSION, cudaMemcpyHostToDevice));
				// checkCudaErrors(cudaMemcpy(dev_centers + i*DIMENSION , nextCenter, DIMENSION*sizeof(double), cudaMemcpyHostToDevice));

				gettimeofday(&sample_end,NULL);
				// printf("Time taken for mean heuristic::\t%d\t%f\n",i,get_time_diff(sample_start,sample_end));
				meanHeuristicTime += get_time_diff(sample_start,sample_end);
			}
			printf("compDistTime\t\t%2.5f\t%2.5f\n",compDistTime,compDistTime/(NUM_CLUSTER-1) );
			printf("makeCumulativeTime\t%2.5f\t%2.5f\n",makeCumulativeTime,makeCumulativeTime/(NUM_CLUSTER-1) );
			printf("samplingTime\t\t%2.5f\t%2.5f\n",samplingTime,samplingTime/(NUM_CLUSTER-1) );
			printf("meanHeuristicTime\t%2.5f\t%2.5f\n",meanHeuristicTime,meanHeuristicTime/(NUM_CLUSTER-1) );
			cudaProfilerStop();
			exit(0);
			// ---------------------- GPU-Based Implementation End --------------------------------------
			

			// ---------------------- CPU-Based Implementation Start ------------------------------------
			// for(i = 0; i < NUM_CLUSTER; i++)
			// {
				// struct timeval sample_start,sample_end;
				// gettimeofday(&sample_start,NULL);
				// multiset = d2_sample(data,centers,NUM_POINTS,N,i);
				// gettimeofday(&sample_end,NULL);
				// printf("Time taken for d2_sample::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
				// samplingTime_1[i] = get_time_diff(sample_start,sample_end);

				// gettimeofday(&sample_start,NULL);
				// double* nextCenter = mean_heuristic(multiset,N);
				// for (int j = 0; j < DIMENSION; ++j)
				// {
				// 	centers[i*DIMENSION + j] = nextCenter[j];
				// }
				// gettimeofday(&sample_end,NULL);
				// printf("Time taken for mean_heuristic::%d-->%f\n",i,get_time_diff(sample_start,sample_end));
				// samplingTime_2[i] = get_time_diff(sample_start,sample_end);
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

		// free(centers);
		cudaFreeHost(distances); // free this way when using page-locked memory for distances
		free(distances);
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

int sample_from_distribution (double* probabilities, int startIndex, int endIndex, double prob) 
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

__global__ void kernelAddConstant(int *g_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
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

// a predicate that checks whether each array element is set to its index plus b
int correctResult(int *data, const int n, const int b)
{
    for (int i = 0; i < n; i++)
        if (data[i] != i + b)
            return 0;

    return 1;
}

int main_check()
{
    int num_gpus = 0;   // number of CUDA GPUs

    printf("Starting main_check...\n");

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");


    /////////////////////////////////////////////////////////////////
    // initialize data
    //
    unsigned int n = num_gpus * 8192;
    unsigned int nbytes = n * sizeof(int);
    int *a = 0;     // pointer to data on the CPU
    int b = 3;      // value by which the array is incremented
    a = (int *)malloc(nbytes);

    if (0 == a)
    {
        printf("couldn't allocate CPU memory\n");
        return 1;
    }

    for (unsigned int i = 0; i < n; i++)
        a[i] = i;


    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    //   each CPU thread controls a different device, processing its
    //   portion of the data.  It's possible to use more CPU threads
    //   than there are CUDA devices, in which case several CPU
    //   threads will be allocating resources and launching kernels
    //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
    //   Recall that all variables declared inside an "omp parallel" scope are
    //   local to each CPU thread
    //
    omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    //omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        checkCudaErrors(cudaSetDevice(cpu_thread_id % num_gpus));   // "% num_gpus" allows more CPU threads than GPU devices
        checkCudaErrors(cudaGetDevice(&gpu_id));
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);

        int *d_a = 0;   // pointer to memory on the device associated with this CPU thread
        int *sub_a = a + cpu_thread_id * n / num_cpu_threads;   // pointer to this CPU thread's portion of data
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        dim3 gpu_threads(128);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

        checkCudaErrors(cudaMalloc((void **)&d_a, nbytes_per_kernel));
        checkCudaErrors(cudaMemset(d_a, 0, nbytes_per_kernel));
        checkCudaErrors(cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice));
        kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);

        checkCudaErrors(cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_a));

    }
    printf("---------------------------\n");

    if (cudaSuccess != cudaGetLastError())
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));


    ////////////////////////////////////////////////////////////////
    // check the result
    //
    bool bResult = correctResult(a, n, b);

    if (a)
        free(a); // free CPU memory

    exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
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
