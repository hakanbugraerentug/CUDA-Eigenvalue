all:

	g++ spectra_v3.cpp -std=c++11 -O3  -lnvblas -L/usr/local/cuda/lib64 -larmadillo -fopenmp -I/opt/intel/mkl/include

#old version of make file
#g++ spectra_v3.cpp -std=c++11 -O3 -larmadillo -fopenmp -I/opt/intel/mkl/include/

#run with export MKL_THREADING_LAYER=GNU; export OMP_NUM_THREADS=8; ./a.out
