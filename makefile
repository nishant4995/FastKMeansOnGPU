all:
	# g++ -D MNIST  main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o mnist
	# g++ -D THREE_D  main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o 3d_pro
	# g++ -D CIFAR  main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o cifar
	g++ -D BIRCH1 -g main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch1
	# g++ -D BIRCH2 -g main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch2
	# g++ -D BIRCH3 -g main.cpp -std=c++11 -O3 -msse4.2 -fopenmp -o birch3
clean:
	rm birch1 birch2 birch3
	# rm mnist cifar 3d_pro birch1 birch2 birch3