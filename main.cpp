#include "main.h"

int num_threads;
random_device rd;
mt19937 gen(rd());
unsigned int max_val = gen.max();

int main(int argc, char* argv[]){


	// Point a ,b;
	// for (int myIter = 0; myIter < DIMENSION; ++myIter)
	// {
	// 	a[myIter] = 2;
	// 	b[myIter] = 0;
	// }
	// printf("Dimension::%d\n",DIMENSION );
	// printf("Distance_1::%f\n",distance(a,b));

	// for (int myIter = 0; myIter < ROUNDED_DIMENSION; ++myIter)
	// {
	// 	a[myIter] = 2;
	// 	b[myIter] = 0;
	// }
	// printf("Distance_2::%f\n",distance(a,b));
	// return 1;
	// argument processing logic

	// FIRST ARGUMENT MUST SPECIFY THE MODE OF RUNNING. ALLOWED VALUES ARE random, kmeans++, d2-seeding, kmeans-par

	// SECOND ARGUMENT SPECIFIES HOW MANY TIMES TO RUN. MUST BE AN INTEGER > 0

	// NUMBER OF THREADS MUST BE SPECIFIED IN THE ENVIRONMENT VARIABLE OMP_NUM_THREADS. DEFAULT VALUE WILL BE 1.

	// FOR d2-seeding, THIRD ARGUMENT MUST SPECIFY THE RATIO N/k, WHERE N IS THE SIZE OF MULTISET

	// FOR kmeans-par, THIRD ARGUMENT MUST SPECIFY THE OVERSAMPLING RATE(l) AND FOURTH MUST SPECIFY THE NUMBER OF ROUNDS(r)

	string mode(argv[1]);
	int method = -1;
	int num_runs = atoi(argv[2]);
	num_threads = 1;
	if(getenv("OMP_NUM_THREADS") != NULL){
		num_threads = atoi(getenv("OMP_NUM_THREADS"));
		printf("num_threads as gotten from env::%d\n",num_threads);
		if(num_threads == 0){
			num_threads = 1;
		}
	}
	else
	{
		printf("num_threads as gotten by default::%d\n",num_threads);
	}

	string base_log_file = string("../logs/") + DATA + string("/") + mode + string("_");
	int N = 0;
	if(mode == "random"){
		method = 0;
	}
	if(mode == "kmeans++"){
		method = 1;
	}
	if(mode == "d2-seeding"){
		method = 2;
		N = floor(NUM_CLUSTER * atof(argv[3]));
		base_log_file += ("N=" + string(argv[3]) +"k_");
	}
	int rounds = 5;
	double oversampling = 2 * NUM_CLUSTER;
	if(mode == "kmeans-par"){
		method = 3;
		oversampling = NUM_CLUSTER * atof(argv[3]);
		base_log_file += ("l=" + string(argv[3]) + "k_");
		rounds = atoi(argv[4]);
		base_log_file += ("r=" + to_string(rounds) + "_");
	}
	// base log file name for individual runs
	base_log_file += ("threads=" + to_string(num_threads) + "_");

	// log file for combined results. Mean and standard deviations
	string result_file = base_log_file + "result.txt";
	base_log_file += "runNo=";
	struct timeval start,end;

	// collect stats about all relevant parameters
	double init_time[num_runs];
	double iter_time[num_runs];
	double total_time[num_runs];
	double init_cost[num_runs];
	double final_cost[num_runs];
	double num_iter[num_runs];

	// read the data into a vector of "vector"
	vector<Point> data(NUM_POINTS);
	ifstream file;
	file.open(string("../data/")+DATA);
	string line;
	double file_read_tmp_value;
	Point file_read_tmp_point;
	gettimeofday(&start,NULL);
	int i,j=0;
	while(getline(file,line)){
		stringstream ss(line);
		j=0;
		while(j < DIMENSION){
			ss>>data[i][j];
			j++;
		}
		i++;
	}
	gettimeofday(&end,NULL);
	ofstream logger;

	for(int run_no = 0; run_no < num_runs ; run_no++){

		double samplingTime_1[NUM_CLUSTER];
		double samplingTime_2[NUM_CLUSTER];
		struct timeval sample_start, sample_end;

		string log ="";
		gettimeofday(&start,NULL);
		// initialize the initial centers
		vector<Point> centers(NUM_CLUSTER);
		if(method == 0){
			// random initialization
			double rand_nos[NUM_CLUSTER];
			// generate k random numbers and bring them to (0,1)
			for(int i = 0; i < NUM_CLUSTER; i++){
				rand_nos[i] = ((double) gen())/max_val;
			}
			// pick points by simply mapping random numbers from (0,1) to (0, NUM_POINTS)
			for(int i = 0; i < NUM_CLUSTER ; i++){
				centers[i] = data[floor(NUM_POINTS * rand_nos[i])];
			}
		} else if(method == 1){ // kmeans++
			// all points have a weight of one. This is an unweighted kmeans++ problem
			vector<int> weights(NUM_POINTS);
			for(int i = 0; i < NUM_POINTS; i++){
				weights[i] = 1;
			}
			for(int i = 0; i < NUM_CLUSTER; i++){
				centers[i] = d2_sample(data,centers,weights,1,i).at(0);
			}
		} else if(method == 2){  // d2-seeding
			// all points have a weight of one. This is an unweighted kmeans++ problem
			vector<int> weights(NUM_POINTS);
			for(int i = 0; i < NUM_POINTS; i++){
				weights[i] = 1;
			}
			vector<double> distances(NUM_POINTS);
			for(int i = 0; i < NUM_CLUSTER; i++){
				gettimeofday(&sample_start,NULL);
				// printf("------------------------start--------------------------------------\n");
				// vector<Point> multiset = d2_sample(data,centers,weights,N,i);
				vector<Point> multiset = d2_sample_2(data,centers,weights,N,i,distances);
				gettimeofday(&sample_end,NULL);
				samplingTime_1[i] = get_time_diff(sample_start,sample_end);
				// printf("-------------------------end---------------------------------------\n");
				gettimeofday(&sample_start,NULL);
				centers[i] = mean_heuristic(multiset);
				gettimeofday(&sample_end,NULL);
				samplingTime_2[i] = get_time_diff(sample_start,sample_end);
			}
		} else if(method == 3){
			vector<Point> C;
			C.push_back(data[NUM_POINTS * (((double) gen())/max_val)]);
			for(int i = 0;i < rounds;i++){
	            vector<Point> tmp = independent_sample(data,C,oversampling);
	            for(int j = 0;j < tmp.size();j++){
	                C.push_back(tmp[j]);
	            }
	        }
	        // find assignments to each point in C in parallel
	        vector<int> weights(C.size());
	        vector<int*> local_weights(num_threads);
	        #pragma omp parallel
	        {
	        	vector<int> local_weight(C.size());
	        	local_weights[omp_get_thread_num()] = local_weight.data();
	        	int index;
    			double min_dist;
    			double current_dist;
    			#pragma omp for schedule(static)
    			for (int i = 0; i < NUM_POINTS; i++) {
    				index = 0;
    				min_dist = DBL_MAX;
    				current_dist = 0;
    				for(int j = 0; j < C.size(); j++){
    					current_dist = distance(data[i],C[j]);
    					if(current_dist < min_dist){
    						index = j;
    						min_dist = current_dist;
    					}
    				}
    				local_weight[index] += 1;
		        }
		        #pragma omp for schedule(static)
		        for(int i = 0; i < C.size(); i++){
		        	for(int j = 0; j < num_threads; j++){
		        		weights[i] = weights[i] + local_weights[j][i];
		        	}
		        }
	        }
	        // now do a weighted kmeans++ initialization 
	        for(int i = 0;i < NUM_CLUSTER;i++){
	            centers[i] = d2_sample(C, centers, weights, 1, i).at(0);
	        }
		}
		gettimeofday(&end,NULL);
		init_time[run_no] = get_time_diff(start,end);

		// now the Lloyd's iterations
		// first we need to figure out the assignments
		gettimeofday(&start,NULL);
		double prev_cost = DBL_MAX;
		int iteration = 0;
		while(true){
			iteration++;
			vector<int> cluster_counts(NUM_CLUSTER); // number of points assigned to each cluster
			vector<Point> cluster_sums(NUM_CLUSTER); // sum of points assigned to each cluster
			vector<int *> cluster_counts_pointers(num_threads); // pointers to local "number of points assigned to each cluster"
    		vector<Point *> cluster_sums_pointers(num_threads); // pointers to local "sum of points assigned to each cluster"

    		// initially, set everything to zero
    		for(int i = 0; i < NUM_CLUSTER; i++){
				cluster_counts[i] = 0;
				for(int j = 0; j < DIMENSION; j++){
					cluster_sums[i][j] = 0;
				}
			}
			// cost according to the current solution
			double current_cost = 0.0;
    		#pragma omp parallel reduction(+: current_cost)
    		{
    			int tid = omp_get_thread_num();
    			vector<int> local_cluster_counts(NUM_CLUSTER); // local "number of points assigned to each cluster"
    			vector<Point> local_cluster_sums(NUM_CLUSTER); // local "sum of points assigned to each cluster"
    			for(int i = 0; i < NUM_CLUSTER; i++){
    				local_cluster_counts[i] = 0;
    				for(int j = 0; j < DIMENSION; j++){
    					local_cluster_sums[i][j] = 0;
    				}
    			}
    			cluster_counts_pointers[tid] = local_cluster_counts.data(); // set the pointer
    			cluster_sums_pointers[tid] = local_cluster_sums.data(); // set the pointer
    			int index;
    			double min_dist;
    			double current_dist;
    			// assign each point to their cluster center in parallel. 
    			// update the cost of current solution and keep updating local counts and sums
    			#pragma omp for schedule(static)
    			for (int i = 0; i < NUM_POINTS; i++) {
    				index = 0;
    				min_dist = DBL_MAX;
    				current_dist = 0;
    				for(int j = 0; j < NUM_CLUSTER; j++){
    					current_dist = distance(data[i],centers[j]);
    					if(current_dist < min_dist){
    						index = j;
    						min_dist = current_dist;
    					}
    				}
    				current_cost += min_dist;
    				local_cluster_counts[index] += 1;
    				for(int j = 0; j < DIMENSION; j++){
    					local_cluster_sums[index][j] = local_cluster_sums[index][j] + data[i][j];
    				}
		        }

		        // aggregate counts and sums across all threads
		        #pragma omp for schedule(static)
		        for(int i = 0; i < NUM_CLUSTER; i++){
		        	for(int j = 0; j < num_threads; j++){
		        		cluster_counts[i] = cluster_counts[i] + cluster_counts_pointers[j][i];
		        		for(int k = 0; k < DIMENSION; k++){
		        			cluster_sums[i][k] = cluster_sums[i][k] + cluster_sums_pointers[j][i][k];
		        		}
		        	}
		        }
		    }
		    if(iteration == 1){
		    	init_cost[run_no] = current_cost;
		    }
		    // now scale all the sums by the number of points at each cluster
		    for(int i = 0; i < NUM_CLUSTER; i++){
		    	int scaler = cluster_counts[i];
		    	for(int j = 0; j < DIMENSION; j++){
		    		centers[i][j] = cluster_sums[i][j]/scaler;
		    	}
		    }
		    // log entry
		    log = log + "Iteration: " + to_string(iteration) + " Cost:" + to_string(current_cost) + "\n";
		    // termination criteria
		    if(1 - current_cost/prev_cost < 0.0001){
		    	prev_cost = current_cost;
		    	break;
		    }
		    prev_cost = current_cost;
		}
		gettimeofday(&end,NULL);
		final_cost[run_no] = prev_cost;
		num_iter[run_no] = iteration;
		iter_time[run_no] = get_time_diff(start,end)/num_iter[run_no];
		total_time[run_no] = iter_time[run_no]*num_iter[run_no] + init_time[run_no];

		logger.open(base_log_file + to_string(run_no) + ".txt");
		log = log + "Number of iterations:" + to_string(num_iter[run_no]) + "\n";
		log = log + "Initialization time:" + to_string(init_time[run_no]) + "\n";
		log = log + "Initialization cost:" + to_string(init_cost[run_no]) + "\n";
		log = log + "Final cost:" + to_string(final_cost[run_no]) + "\n";
		log = log + "Total time:" + to_string(total_time[run_no]) + "\n";
		log = log + "Per iteration time:" + to_string(iter_time[run_no]) + "\n";
		log = log + "Total iteration time:" + to_string(iter_time[run_no]*num_iter[run_no]) + "\n";
		if(method == 2) // d2-seeding
		{
			// int iter;
			// for (iter = 0; iter < NUM_CLUSTER; ++iter)
			// {
			// 	log = log + "samplingTime_1:" + to_string(iter) +"  "+ to_string(samplingTime_1[iter]) + "\n";
			// 	log = log + "samplingTime_2:" + to_string(iter) +"  "+ to_string(samplingTime_2[iter]) + "\n";
			// }
			log = log + "samplingTime_1:" + mean(samplingTime_1,NUM_CLUSTER) + "\n";
			log = log + "samplingTime_2:" + mean(samplingTime_2,NUM_CLUSTER) + "\n";
		}
		logger << log << endl;
		logger.close();
	}


	logger.open(result_file);
	logger << "Initial cost: " << mean(init_cost,num_runs) << " " << sd(init_cost,num_runs) << endl;
	logger << "Final cost: " << mean(final_cost,num_runs) <<  " " << sd(final_cost,num_runs) << endl;
	logger << "Number of iterations: " << mean(num_iter, num_runs) <<  " " << sd(num_iter,num_runs) << endl;
	logger << "Initialization time: " << mean(init_time,num_runs) <<  " " << sd(init_time,num_runs) << endl;
	logger << "Per iteration time: " << mean(iter_time,num_runs) <<  " " << sd(iter_time,num_runs) << endl;

}

// generate num_samples sized multiset from weighted data with weights wrt. centers where the current size of centers is size
vector<Point> d2_sample(vector<Point> &data,vector<Point> &centers,vector<int> &weights,int num_samples, int size){
	int num_pts = data.size();
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,num_pts/num_threads], [num_pts/num_threads+1,...,num_pts*2/num_threads],... and so on. These groups contain cumulative distances.
	vector<double> distances(num_pts);
    vector<double> local_sums(num_threads);   // local sums. first is sum for [0...num_pts/num_threads-1], and so on. This is also a cumulative distribution.
    vector<Point> result(num_samples);
    // we're gonna need 2*num_samples random numbers. 
    vector<double> rnd(2*num_samples);
	for(int i = 0; i < 2*num_samples; i++){
		rnd[i] = ((double) gen())/max_val;
	}
    #pragma omp parallel
    {
    	struct timeval start,end;
    	gettimeofday(&start,NULL);
    	// create blocks of data
        int tid = omp_get_thread_num();
        int per_thread = num_pts / num_threads;
        int lower = tid * per_thread;
        int higher = (tid + 1) * per_thread;
        if(tid == num_threads - 1) higher = num_pts;
        int block_size = higher - lower;
        double min_dist, local_dist;
        Point p;
        int w;
        double prev_val = 0;
        // cost of each block
        double local_sum = 0;
        int center_size = size;
        for(int i = 0;i < block_size;i++){
            w = weights[lower+i];
            if(center_size == 0){
                local_sum += w;
                distances[lower+i] = w + prev_val;
            } else{
                p = data[lower+i];
                min_dist = distance(p,centers[0]);
                for (int j = 1; j < centers.size(); j++) {
                    local_dist = distance(p,centers[j]);
                    min_dist = min (min_dist, local_dist); // calculating minimum distances
                }
                local_sum += w * min_dist * min_dist;
                distances[lower+i] = w * min_dist * min_dist + prev_val; // make cumulative 
            }
            prev_val = distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<num_threads;i++){
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
            // printf("Number of threads::%d\n",omp_get_num_threads());
        }
    	gettimeofday(&end,NULL);
    	float cost_time = get_time_diff(start,end);
        #pragma omp barrier
        gettimeofday(&start,NULL);
        #pragma omp for
        for(int i = 0;i < num_samples;i++){
        	// first pick a block from the local_sums distribution
            int groupNo = sample_from_distribution(local_sums, 0, num_threads, rnd[i*2]*local_sums[num_threads-1]);
            // the start and end index of this block
            int startIndex = groupNo * per_thread;
            int endIndex = (groupNo + 1) * per_thread;
            if(groupNo == num_threads - 1) endIndex = num_pts;
            // now sample from the cumulative distribution of the block
            result[i] = data[sample_from_distribution(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1])];
        }
        gettimeofday(&end,NULL);
        float sample_time = get_time_diff(start,end);
        if (center_size >= 99)
        {
		    printf("Cost computation time 	::%f\n",cost_time);
			printf("Sampling time 		::%f\n",sample_time);	
        } 	
    }
    return result;
}

// This one is optimized for the case when we have to sample just one point, like when used in k-means ++
Point d2_sample_3(vector<Point> &data,vector<Point> &centers,vector<int> &weights,int num_samples, int size){
	int num_pts = data.size();
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,num_pts/num_threads], [num_pts/num_threads+1,...,num_pts*2/num_threads],... and so on. These groups contain cumulative distances.
	vector<double> distances(num_pts);
    vector<double> local_sums(num_threads);   // local sums. first is sum for [0...num_pts/num_threads-1], and so on. This is also a cumulative distribution.
    // vector<Point> result(num_samples);
    // we're gonna need 2*num_samples random numbers. 
    vector<double> rnd(2*num_samples);
	for(int i = 0; i < 2*num_samples; i++){
		rnd[i] = ((double) gen())/max_val;
	}
    #pragma omp parallel
    {
    	// struct timeval start,end;
    	// gettimeofday(&start,NULL);
    	// create blocks of data
        int tid = omp_get_thread_num();
        int per_thread = num_pts / num_threads;
        int lower = tid * per_thread;
        int higher = (tid + 1) * per_thread;
        if(tid == num_threads - 1) higher = num_pts;
        int block_size = higher - lower;
        double min_dist, local_dist;
        Point p;
        int w;
        double prev_val = 0;
        // cost of each block
        double local_sum = 0;
        int center_size = size;
        for(int i = 0;i < block_size;i++){
            w = weights[lower+i];
            if(center_size == 0){
                local_sum += w;
                distances[lower+i] = w + prev_val;
            } else{
                p = data[lower+i];
                min_dist = distance(p,centers[0]);
                for (int j = 1; j < centers.size(); j++) {
                    local_dist = distance(p,centers[j]);
                    min_dist = min (min_dist, local_dist); // calculating minimum distances
                }
                local_sum += w * min_dist * min_dist;
                distances[lower+i] = w * min_dist * min_dist + prev_val; // make cumulative 
            }
            prev_val = distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<num_threads;i++){
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
            // printf("Number of threads::%d\n",omp_get_num_threads());
        }
    }

    	// gettimeofday(&end,NULL);
    	// float cost_time = get_time_diff(start,end);
        // #pragma omp barrier
        // gettimeofday(&start,NULL);
        // #pragma omp for
        // for(int i = 0;i < num_samples;i++){

        	// first pick a block from the local_sums distribution
            int groupNo = sample_from_distribution(local_sums, 0, num_threads, rnd[0]*local_sums[num_threads-1]);
            // the start and end index of this block
	        int per_thread = num_pts / num_threads;
            int startIndex = groupNo * per_thread;
            int endIndex = (groupNo + 1) * per_thread;
            if(groupNo == num_threads - 1) endIndex = num_pts;
            // now sample from the cumulative distribution of the block
            // result[i] = 
            return data[sample_from_distribution(distances, startIndex, endIndex, rnd[1]*distances[endIndex-1])];
        // }
        // gettimeofday(&end,NULL);
   		// float sample_time = get_time_diff(start,end);
        // if (center_size >= 0)
        // {
		    // printf("Cost computation time 	::%f\n",cost_time);
			// printf("Sampling time 			::%f\n",sample_time);	
        // }
    	
    
    // return result;
}


vector<Point> d2_sample_2(vector<Point> &data,vector<Point> &centers,vector<int> &weights,int num_samples, int size,vector<double> &distances){
	int num_pts = data.size();
	// cumulative probability for each group of points
	// the distances are cumulative only for a group. So, [0,...,num_pts/num_threads], [num_pts/num_threads+1,...,num_pts*2/num_threads],... and so on. These groups contain cumulative distances.
	// vector<double> distances(num_pts);
    vector<double> local_sums(num_threads);   // local sums. first is sum for [0...num_pts/num_threads-1], and so on. This is also a cumulative distribution.
    vector<Point> result(num_samples);
    // we're gonna need 2*num_samples random numbers. 
    vector<double> rnd(2*num_samples);
	for(int i = 0; i < 2*num_samples; i++){
		rnd[i] = ((double) gen())/max_val;
	}
	omp_lock_t print_lock;
	omp_init_lock(&print_lock);

    #pragma omp parallel
    {
    	// create blocks of data
    	struct timeval start,end;
    	gettimeofday(&start,NULL);
        int tid = omp_get_thread_num();
        int per_thread = num_pts / num_threads;
        int lower = tid * per_thread;
        int higher = (tid + 1) * per_thread;
        if(tid == num_threads - 1) higher = num_pts;
        int block_size = higher - lower;
        double min_dist, local_dist;
        Point p;
        int w;
        double prev_val = 0;
        double old_prev_val = 0;

        // cost of each block
        double local_sum = 0;
        int center_size = size;
        for(int i = 0;i < block_size;i++)
        {
            w = weights[lower+i];
            if(center_size == 0)
            {
                local_sum += w;
                distances[lower+i] = w + prev_val;
            	prev_val = distances[lower+i];
            } 
            else if (center_size == 1)
            {
                p = data[lower+i];
                min_dist = distance(p,centers[0]);
                // for (int j = 1; j < centers.size(); j++) 
                for (int j = 1; j < center_size; j++) 
                {
                    local_dist = distance(p,centers[j]);
                    min_dist = min (min_dist, local_dist); // calculating minimum distances
                }
                local_sum += w * min_dist * min_dist;
                distances[lower+i] = w * min_dist*min_dist + prev_val; // make cumulative 
            	prev_val = distances[lower+i];
            } 
            else // center_size >= 2
            {
                p = data[lower+i];
                // min_dist = distance(p,centers[0]);
                min_dist   		= distances[lower+i] - old_prev_val;
            	old_prev_val 	= distances[lower+i];  // Important for it to have old value of distance[lower+i] so that min_dist is correct in next iteration
                local_dist 		= distance(p,centers[center_size-1]); // Find distance wrt last added new center;
                local_dist 		= local_dist*local_dist;
                min_dist 		= min(min_dist,local_dist);
                
                local_sum 		+= w * min_dist; // min_dist is already squared here because it is calculated usign cumulative distance
                distances[lower+i] = w * min_dist + prev_val; // make cumulative 
            	prev_val = distances[lower+i];
            }
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i=1;i<num_threads;i++){
                local_sums[i] = local_sums[i] + local_sums[i-1]; // make cumulative
            }
        }
        gettimeofday(&end,NULL);
    	float cost_time = get_time_diff(start,end);
        #pragma omp barrier
        gettimeofday(&start,NULL);
        #pragma omp for
        for(int i = 0;i < num_samples;i++){
        	// first pick a block from the local_sums distribution
            int groupNo = sample_from_distribution_2(local_sums, 0, num_threads, rnd[i*2]*local_sums[num_threads-1]);
            // the start and end index of this block
            int startIndex = groupNo * per_thread;
            int endIndex = (groupNo + 1) * per_thread;
            if(groupNo == num_threads - 1) endIndex = num_pts;
            // now sample from the cumulative distribution of the block
            result[i] = data[sample_from_distribution_2(distances, startIndex, endIndex, rnd[2*i+1]*distances[endIndex-1])];
        }
        gettimeofday(&end,NULL);
        float sample_time = get_time_diff(start,end);
        if (center_size >= 99)
        {
		    printf("Cost computation time 	::%f\n",cost_time);
			printf("Sampling time 		::%f\n",sample_time);	
        }
    }
    return result;
}
// an efficient cumulative distribution sampling procedure which runs in log(n) time
static inline int sample_from_distribution (vector<double> &probabilities, int startIndex, int endIndex, double p) {
    int start=startIndex,end=endIndex-1;
    int mid;
    while(start<=end) {
        mid = (start+end)/2;
        if(p<probabilities[mid-1]) {
            end = mid-1;
        } else if(p > probabilities [mid]) {
            start = mid+1;
        } else {
            break;
        }
    }
    return mid;
}

static inline int sample_from_distribution_2 (vector<double> &probabilities, int startIndex, int endIndex, double p) {
    int start=startIndex,end=endIndex-1;
    int mid;
    while(start<=end) {
        mid = (start+end)/2;
        if(p<probabilities[mid-1]) {
            end = mid-1;
        } else if(p > probabilities [mid]) {
            start = mid+1;
        } else {
            break;
        }
    }
    return mid;
}


// returns a center by running the mean heuristic on the multiset
Point mean_heuristic(vector<Point> & multiset){

	// struct timeval start,end;
	// first do a kmeans++ initialiation on the multiset
	vector<int> weights(multiset.size());
	for(int i = 0; i < multiset.size(); i++){
		weights[i] = 1;
	}
	// gettimeofday(&start,NULL);
	vector<Point> level_2_sample(NUM_CLUSTER);
	for(int i = 0; i < NUM_CLUSTER; i++){
		level_2_sample[i] = d2_sample_3(multiset,level_2_sample,weights,1,i);
	}

	// gettimeofday(&end,NULL);
	// printf("Time taken to choose k centers::%f\n",get_time_diff(start,end));
	// gettimeofday(&start,NULL);
	vector<int> counts(NUM_CLUSTER); // number of points assigned to each kmeans++ center
    vector<Point> cluster_means(NUM_CLUSTER); // for taking the centroid later on. We maintain a sum of all points assigned to a center here.
    for (int i = 0; i < NUM_CLUSTER; i++) {
        counts[i]=0;
        for(int j = 0; j< DIMENSION; j++){
        	cluster_means[i][j]=0;
        }
    }
    // here the heuristic does things in a parallel fashion
    // maintain a local structure for each thread to keep track of cluster sums and counts
    vector<int *> local_tmp_counts (num_threads);
    vector<Point *> local_tmp_cluster_means (num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        vector<int> local_counts (NUM_CLUSTER);
        vector<Point> local_cluster_means(NUM_CLUSTER);
        for (int i = 0; i < NUM_CLUSTER; i++) {
            for(int j = 0; j < DIMENSION; j++) {
            	local_cluster_means[i][j] = 0;
            } 
        }
        local_tmp_counts[tid] = local_counts.data();  // save the pointers to local data structures
        local_tmp_cluster_means[tid] = local_cluster_means.data();
        double min_dist, tmp_dist;
        int index;
        #pragma omp for schedule(static)
        for (int i = 0; i < multiset.size(); i++) {
            min_dist = distance(level_2_sample[0],multiset[i]);  // distance of each kmeans++ center from the points in sampled_set
            index = 0;
            for (int j = 1; j < NUM_CLUSTER; j++) {
                tmp_dist = distance(level_2_sample[j],multiset[i]); // figure out the minimum and assign the point to that kmeans++ center
                if (tmp_dist < min_dist) {
                    min_dist = tmp_dist;
                    index = j;
                }
            }
            for(int j = 0; j < DIMENSION; j++){
            	local_cluster_means[index][j] += multiset[i][j];
            }
            local_counts[index]++;
        }
        // aggregate across all threads
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_CLUSTER; i++) {
            for (int p = 0; p < num_threads ; p++) {
            	for(int j = 0; j < DIMENSION; j++){
            		cluster_means[i][j] += local_tmp_cluster_means[p][i][j];
            	}
                counts[i] += local_tmp_counts[p][i];
            }
        }
    }
    int max = counts[0];
    int index = 0;
    for (int i = 1; i < NUM_CLUSTER; i++) {
        if (counts[i] > max) {
            max = counts[i];
            index = i; // largest cluster with maximum points from sampled_set assigned to it.
        }
    }
    // gettimeofday(&end,NULL);
    // printf("Time for finding partitions::%f\n",get_time_diff(start,end));
    // do the scaling to find the mean
    for(int i = 0; i < DIMENSION; i++){
    	cluster_means[index][i] /= counts[index];
    }
    return cluster_means[index];
}

// pick each point independently based on its probability
vector<Point> independent_sample(vector<Point> &data, vector<Point> &centers, double l){
	int num_pts = data.size();
	vector<double> rand_nos(num_pts);
    for(int i = 0;i < num_pts;i++){
        rand_nos[i] = ((double)gen())/max_val;
    }
    vector<double> distances(num_pts); // weighted distances of all points. Exact values. No cumulative this time.
    vector<Point *> local_samples(num_threads); // samples from each block of data
    vector<int> local_sample_counts(num_threads); // number of samples from each block of data
    vector<double> local_sums(num_threads); // sum of cost of each block
    double global_sum = 0; // total cost
    vector<Point> result;
    #pragma omp parallel
    {
    	// first create blocks
        int tid = omp_get_thread_num();
        int per_thread = num_pts/num_threads;
        int lower = tid*per_thread;
        int higher = (tid+1)*per_thread;
        if(tid == num_threads - 1) higher = num_pts;
        int block_size = higher-lower;
        double min_dist, local_dist;
        Point p;
        int w;
        double local_sum = 0;
        // now, for each block, calculate the total cost and the cost for each point
        for(int i=0;i<block_size;i++){
            p = data[lower+i];
            min_dist = distance(p,centers[0]);
            for (int j = 1; j < centers.size(); j++) {
                local_dist = distance(p,centers[j]);
                min_dist = min (min_dist, local_dist); // calculating minimum distances
            }
            distances[lower+i] = min_dist*min_dist;
            local_sum += distances[lower+i];
        }
        local_sums[tid] = local_sum;
        #pragma omp barrier // everyone is here now
        #pragma omp master
        {
            for(int i = 0;i<num_threads;i++){
                global_sum += local_sums[i]; // find the global sum
            }
        }
        #pragma omp barrier
        vector<Point> local_sample;
        for(int i=0;i<block_size;i++){
            if(rand_nos[lower+i] <= l*distances[lower+i]/global_sum){
                local_sample.push_back(data[lower+i]); // now, for each point, sample with probability e(p,C)/cost
            }
        }
        local_samples[tid] = local_sample.data();
        local_sample_counts[tid] = local_sample.size();
        #pragma omp barrier
        #pragma omp master
        {
            for(int i = 0;i<num_threads;i++){
                for(int j=0;j<local_sample_counts[i];j++){ 
                    result.push_back(local_samples[i][j]); // merge samples from each block
                }
            }
        }
        #pragma omp barrier
    }
    return result;
}

static inline double distance(Point a, Point b){
	double answer = 0.0;
	Point c;
	
	// find the difference of a and b. g++ will vectorize this loop
	for(int counter1 = 0; counter1 < DIMENSION; counter1++){
		c[counter1] = (a[counter1] - b[counter1]) * (a[counter1] - b[counter1]);
	}

	// some simd magic to take "horizontal" sum of a vector
	__m128 mmSum = _mm_setzero_ps();
	int rounded_d = DIMENSION - (DIMENSION%SIMD_WIDTH);
	int i = 0;
	for(i = 0; i < rounded_d; i += SIMD_WIDTH){
		mmSum = _mm_add_ps(mmSum, _mm_loadu_ps(&c[i]));
	}

	for(;i < DIMENSION; i++){
		mmSum = _mm_add_ss(mmSum, _mm_load_ss(&c[i]));
	}

	mmSum = _mm_hadd_ps(mmSum,mmSum);
	mmSum = _mm_hadd_ps(mmSum,mmSum);

	return _mm_cvtss_f32(mmSum);
}

static inline double get_time_diff(struct timeval t1, struct timeval t2){
	return t2.tv_sec - t1.tv_sec + 1e-6 * (t2.tv_usec - t1.tv_usec);
}

static inline string mean(double* a, int n){
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	return to_string(sum/n);
}

static inline string sd(double* a, int n){
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i];
	}
	double mean = sum/n;
	sum = 0;
	for(int i = 0; i < n; i++){
		sum += (a[i] - mean) * (a[i] - mean);
	}
	return to_string(sqrt(sum/n));
}
