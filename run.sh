#PBS -N testRunJob
#PBS -P cse
#PBS -m bea
#PBS -M cs1130242@iitd.ac.in
#PBS -o ~/IS/HPC_outFile.txt
#PBS -e ~/IS/HPC_errorFIle.txt
module load compiler/gcc/4.9.3/compilervars
export OMP_NUM_THREADS=40
 ~/IS/FastKMeansOnGPU/birch1 d2-seeding 2 10 > ~/IS/FastKMeansOnGPU/temp.txt 2>~/IS/FastKMeansOnGPU/tempError.txt