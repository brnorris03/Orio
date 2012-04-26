
gcc cpu.c -o cpu
nvcc -O3 -arch=sm_20 noroll.cu
mv a.out noroll
nvcc -O3 -arch=sm_20 unrolls2.cu
mv a.out unrolls2
nvcc -O3 -arch=sm_20 unrolls4.cu
mv a.out unrolls4
nvcc -O3 -arch=sm_20 unrolls8.cu
mv a.out unrolls8
nvcc -O3 -arch=sm_20 unrollp2.cu
mv a.out unrollp2
nvcc -O3 -arch=sm_20 unrollp4.cu
mv a.out unrollp4
nvcc -O3 -arch=sm_20 unrollp8.cu
mv a.out unrollp8
