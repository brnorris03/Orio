#!/bin/sh

# runAll.sh
#
# Straightforward script which runs all C executables. 
# 
#
# Alex Kaiser, LBNL, 10/4/10. 

# build codes if not already built
touch Makefile
make

./denseLinearAlgebra/C/denseMain
denseMainPass=$?


./sparseLinearAlgebra/C/sparseMain  
sparseMainPass=$?

	
./structuredGrid/C/structuredGridMain
structuredGridMainPass=$?


./spectralMethods/C/spectralMain
spectralMainPass=$?


cd nBody/C/2D
./nBodyMain
nBodyMainTwoDPass=$?
cd ..


cd 3D
./nBodyMain
nBodyMainThreeDPass=$?
cd ../../..


./monteCarlo/C/monteCarloIntegrate
monteCarloIntegratePass=$?


cd sort
./charArraySort
charArraySortPass=$?
./integerSort
integerSortPass=$?
./spatialSort
spatialSortPass=$?
cd ..


# Output whether each test passed 
if [ $denseMainPass -eq 1 ] 
then 
	echo "Dense Linear Algebra tests passed."
else
	echo "Dense Linear Algebra tests failed."
fi


if [ $sparseMainPass -eq 1 ] 
then 
	echo "Sparse Linear Algebra tests passed."
else
	echo "Sparse Linear Algebra tests failed."
fi


if [ $structuredGridMainPass -eq 1 ] 
then 
	echo "Structured Grid tests passed."
else
	echo "Structured Grid tests failed."
fi


if [ $spectralMainPass -eq 1 ] 
then 
	echo "Spectral Methods tests passed."
else
	echo "Spectral Methods tests failed."
fi


if [ $nBodyMainTwoDPass -eq 1  -a  $nBodyMainThreeDPass -eq 1 ] 
then 
	echo "N Body tests passed."
else
	echo "N Body tests failed."
fi


if [ $monteCarloIntegratePass -eq 1 ] 
then 
	echo "Quasi-Monte Carlo integrate tests passed."
else
	echo "Quasi-Monte Carlo integrate tests failed."
fi

if [ $charArraySortPass -eq 1 -a \
	 $integerSortPass -eq 1 -a \
	 $spatialSortPass -eq 1 ] 
then 
	echo "Sort tests passed."
else
	echo "Sort tests failed."
fi


# Check whether all tests passed
if [ $denseMainPass -eq 1 -a \
	 $sparseMainPass -eq 1 -a \
	 $structuredGridMainPass -eq 1 -a \
	 $spectralMainPass -eq 1 -a \
	 $nBodyMainTwoDPass -eq 1 -a \
	 $nBodyMainThreeDPass -eq 1 -a \
	 $monteCarloIntegratePass -eq 1 -a \
	 $charArraySortPass -eq 1 -a \
	 $integerSortPass -eq 1 -a \
	 $spatialSortPass -eq 1 ] 
then 
	echo "\nAll tests passed.\n"
else
	echo "\nTests failed.\n"
fi