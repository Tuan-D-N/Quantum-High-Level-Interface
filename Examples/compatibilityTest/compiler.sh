nvq++ -c cudaQ.cpp -o cudaQ.o
nvcc -c -arch=sm_80 cuQuantum.cu -o cuQuantum.o