
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>

/*
 32 bit integer sort. 
 
 Alex Kaiser, LBNL, 4/16/10.
 */ 

void swap(unsigned int v[], int i, int j);
void quickSort(unsigned int v[], int left, int right);

void mergeSort(unsigned int x[], int lenX);
void mergeHelper(unsigned int *x, int lenX,unsigned  int *buffer);
void merge(unsigned int x[], int lenX, unsigned int buffer[]);

void radixSort(unsigned int *x, int left, int right, unsigned int bit);

char verifySorted(unsigned int *x, int size);


unsigned int * generateUniformDistribution(int length) ; 
unsigned int * generateAlmostSortedDistribution(int length, int numToSwap) ; 
unsigned int * generateReverseDistribution(int length) ; 

// random number generators
typedef struct {
	double x[2];
}dd_real;

dd_real ddSub(dd_real a, dd_real b) ;
dd_real ddMult(double a, double b);
double expm2(double p, double modulus);
double bcnrand( );

int getRandomInt(int maxVal) ; 
unsigned int getRandomUnsigned() ; 


// timer
double read_timer( );


int main(){

	/*
	 32 bit integer sort. 
	 Three data set generators are available:
		1. Uniformly distributed data. 
		2. Almost sorted distribution, where data is sorted but some percentage of data is out of place. 
		3. Uniformly distributed data, sorted in reverse. 
	 
	 Each are sorted and verified. 
	 Three algorithms may be selected:
		1. Quick sort. 
		2. Merge sort. 
		3. Radix sort. 
	*/ 
	
	printf("Begin integer sort tests.\n");
	
	char allPass = 1; 
	
	// int i; 
	
	double startTime, endTime;
	
	int size = 100000;
	unsigned int *x ; // = (unsigned int *) malloc(size * sizeof(unsigned int));

	printf("Test uniform distribution:\n");
	x = generateUniformDistribution(size) ; 
	
	startTime = read_timer(); 
	
	// sort contents
	// any of three sorts may be selected
	
	quickSort(x, 0, size-1);
	//mergeSort(x, size); 
	//radixSort(x, 0, size-1, 31) ;

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	// output array if desired
	/*
	for(i=0; i<size; i++)
		printf("%u \n", x[i]);
	printf("\n");
	 */ 

	// verify
	if( verifySorted(x, size) )
		printf("Array sorted properly.\nTest passed.\n\n");
	else{
		fprintf(stderr, "Array not in ascending order.\nTest failed.\n\n");
		allPass = 0; 
	}
	
	// new array is allocated in each generator, so free to avoid leaks. 
	free(x); 
	
	printf("Test 'almost sorted' distribution:\n");
	x = generateAlmostSortedDistribution(size, size / 10) ; 
	
	startTime = read_timer();
	
	// sort contents
	// any of three sorts may be selected
	quickSort(x, 0, size-1);
	//mergeSort(x, size); 
	//radixSort(x, 0, size-1, 31) ;
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	// output array if desired
	/*
	 for(i=0; i<size; i++)
		 printf("%u \n", x[i]);
	 printf("\n");
	 */ 
	
	// verify
	if( verifySorted(x, size) )
		printf("Array sorted properly.\nTest passed.\n\n");
	else{
		fprintf(stderr, "Array not in ascending order.\nTest failed.\n\n");
		allPass = 0; 
	}
	
	// new array is allocated in each generator, so free to avoid leaks. 
	free(x); 
	
	printf("Test reverse sorted distribution:\n");
	x = generateReverseDistribution(size) ; 
	
	startTime = read_timer();
	
	// sort contents
	// any of three sorts may be selected
	quickSort(x, 0, size-1);
	//mergeSort(x, size); 
	//radixSort(x, 0, size-1, 31) ;
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	// output array if desired
	/*
	 for(i=0; i<size; i++)
		 printf("%u \n", x[i]);
	 printf("\n");
	 */ 
	
	// verify
	if( verifySorted(x, size) )
		printf("Array sorted properly.\nTest passed.\n\n");
	else{
		fprintf(stderr, "Array not in ascending order.\nTest failed.\n\n");
		allPass = 0; 
	}
	
	if (allPass) 
		printf("All integer sort tests passed.\n\n\n");
	else
		fprintf(stderr, "Integer sort tests failed!\n\n\n");

	
	return allPass;
}



void quickSort(unsigned int v[], int left, int right){
	/* 
	 Simple quick sort 
	 Taken directely from "The C Programming Language", Kernighan and Ritchie.
	 Sort v[left]...v[right] into increasing order. 
	 
	 Input:
		unsigned int v[]	Array to sort
		int left			First index of array to sort
		int right			Last index of array to sort
	 
	 Output:
		unsigned int v[]	Array, sorted in the specified region
	 
	 */ 
	
	int i, last;

	if (left >= right) /* do nothing if array contains */
		return;        /* fewer than two elements */

	swap(v, left, (left + right)/2); /* move partition elem */
	last = left;                     /* to v[0] */

	for (i = left + 1; i <= right; i++)  /* partition */
		if (v[i] < v[left])
			swap(v, ++last, i);

	swap(v, left, last);            /* restore partition  elem */

	quickSort(v, left, last-1);
	quickSort(v, last+1, right);
}


void swap(unsigned int v[], int i, int j) {
	/*   
	 Interchange v[i] and v[j] in place
	 
	 Input:
		unsigned int v[]	Array to sort
		int left			First to swap 
		int right			Second to swap
	 
	 Output:
		unsigned int v[]	Array, sorted in the specified region
	 */ 
	
	int temp;
	temp = v[i];
	v[i] = v[j];
	v[j] = temp;
}


void mergeSort(unsigned int x[], int lenX){
	/* 
	 Wrapper function for merge sort. 
	 Sorts x[0] ... x[lenX - 1]. 
	 Allocates the necessary memory for buffers. 
	 
	 Input: 
		unsigned int x[]	Array to sort
		int lenX			Length of array to sort
	 
	 Output: 
		unsigned int x[]	Array, sorted in the specified region
	 */ 

	unsigned int *buffer = (unsigned int *) malloc( (lenX/ 2) * sizeof(unsigned int) ) ;
	mergeHelper(x, lenX, buffer) ;
}


void mergeHelper(unsigned int *x, int lenX, unsigned  int *buffer){
	/*
	 Main recursives merge sort routine. 
	 Sorts x[0] ... x[lenX - 1] 
	 
	 Input: 
		unsigned int x[]	Array to sort
		int lenX			Length of array to sort
	 
	 Output: 
		unsigned int x[]	Array, sorted in the specified region
	 */

	if( lenX > 1 ){
		mergeHelper(x, lenX/2, buffer);
		mergeHelper(x+(lenX/2), (lenX - lenX/2), buffer);
		merge(x, lenX, buffer);
	}
}


void merge(unsigned int x[], int lenX, unsigned int buffer[]){
	/*
	 Merge two halves of x. 
	 
	 Buffer must be of sufficient length, that is at least floor(lenX/2)	
	 Half of x is initially copied to the buffer. 
	 If length of x is odd, then length of buffer is floor(lenX/2)
	 and ceil(lenX/2) is left in the x array. 
	 
	 Input: 
		unsigned int x[]		Array to sort
		int lenX				Length of array to sort
		unsigned int buffer[]	Buffer array for moving data
	 
	 Output: 
		unsigned int x[]		Array, sorted in the specified region
	 */ 
	
	int j;
	int xIndex, bufIndex, lenBuffer;
	lenBuffer = lenX / 2 ;

	// copy x over
	for(j=0; j < lenBuffer; j++)
		buffer[j] = x[j] ;

	j=0;
	bufIndex = 0;
	xIndex = lenBuffer ;

	while( (xIndex < lenX) && (bufIndex < lenBuffer) ){
		if( x[xIndex] <= buffer[bufIndex] ){
			x[j] = x[xIndex] ;
			xIndex++ ;
			j++;
		}
		else{
			x[j] = buffer[bufIndex] ;
			bufIndex++ ;
			j++ ;
		}
	}

	// copy remaining elements in buffer if needed
	while(bufIndex < lenBuffer){
		x[j] = buffer[bufIndex] ;
		bufIndex++;
		j++;
	}
}


void radixSort(unsigned int *x, int left, int right, unsigned int bit){
	/*	
		Radix 2 integer sort
		Follows a most significant bit scheme
		Sorts array x from index 'left' to index 'right'
			by exchanges based on value at the specified bit.
		Bit index is zero based.
	
	Input: 
		unsigned int *x			Array to sort
		int left				First index of array to sort
		int right				Last index of array to sort
		unsigned int bit		Bit of each element to run comparision with
		
	 Output: 
		unsigned int *x			Array, sorted in the specified region
		
		*/ 

	// shouldn't happen with the way recursion is structured, 
	// but leave for robustness if routine is called to sort one element
	if (left == right)
		return;

	int zerosIndex = left - 1 ;
	int onesIndex = right + 1 ;

	unsigned int mask = 0x1 << bit;

	while(zerosIndex + 1 < onesIndex){
		// skip through the left most zeros
		while ( (!(x[zerosIndex + 1] & mask))  &&  (zerosIndex < right) ) {
			zerosIndex++ ;
		}

		// and skip the right most ones
		while ( (x[onesIndex - 1] & mask)  &&  (left < onesIndex) ) {
			onesIndex-- ;
		}

		if( (zerosIndex + 1 < onesIndex) ){
			swap(x, zerosIndex + 1, onesIndex - 1);
			zerosIndex++;
			onesIndex--;
		}
	}

	// proceed to the next bit if necessary
	if (bit > 0) {
		if(left < zerosIndex) // must find at lease two zeros to recurse on the zeros side
			radixSort(x, left, zerosIndex, bit - 1);
		if(onesIndex < right) // must find at least to ones to recurse on the ones side
			radixSort(x, onesIndex, right, bit - 1);
	}

}



char verifySorted(unsigned int *x, int size){
	/*
	 Simple check that list is sorted. 
	 
	 Input:
		unsigned int *x			Array to check
		int size				Number of elements to check 
	 
	 Output:
		char (returned)		Boolean, whether array is sorted. 
	 
	 */ 
	int i;
	for (i=0; i<size - 1; i++) {
		if(x[i] > x[i+1])
			return 0;
	}
	return 1;
}

unsigned int * generateUniformDistribution(int length){
	/*
	 Generates an array of uniformly distributed unsigned integers. 
 
	 Input:
	 int length						Length of array to generate. 
	 
	 Output:
	 unsigned int *	(returned)		Array of uniformly distributed random number generators. 
	*/ 
	
	int i; 
	unsigned int *x = (unsigned int *) malloc(length * sizeof(unsigned int));
	
	// generate n 32 bit ints
	for(i=0; i<length; i++){
		x[i] = getRandomUnsigned(); 
	}
	
	return x; 
}

unsigned int * generateAlmostSortedDistribution(int length, int numToSwap){
	/*
	 Generates an array of uniformly distributed unsigned integers. 
	 
	 Input:
	 int length						Length of array to generate. 
	 
	 Output:
	 unsigned int *	(returned)		Array of uniformly distributed random number generators. 
	 */ 
	
	int i; 
	unsigned int *x = (unsigned int *) malloc(length * sizeof(unsigned int));
	
	// generate n 32 bit ints
	for(i=0; i<length; i++){
		x[i] = getRandomUnsigned(); 
	}
	
	mergeSort(x, length);
	
	// add swapping
	int firstIndex, secondIndex; 
	for(i=0; i<numToSwap; i++){
		firstIndex = getRandomInt(length) ; 
		secondIndex = getRandomInt(length) ; 
		swap(x, firstIndex, secondIndex) ; 
	}
	
	return x; 
}

unsigned int * generateReverseDistribution(int length){
	/*
	 Generates an array of reverse order unsigned integers. 
	 
	 Input:
	 int length						Length of array to generate. 
	 
	 Output:
	 unsigned int *	(returned)		Array of uniformly distributed random number generators. 
	 */ 
	
	int i; 
	unsigned int *x = (unsigned int *) malloc(length * sizeof(unsigned int));
	
	// generate n 32 bit ints
	for(i=0; i<length; i++){
		x[i] = getRandomUnsigned();
	}
	
	mergeSort(x, length); 
	
	unsigned int *rev = (unsigned int *) malloc(length * sizeof(unsigned int));
	
	for(i=0; i<length; i++)
		rev[length - i - 1] = x[i] ; 
	
	free(x); 
	
	return rev; 
}
	 
int getRandomInt(int maxVal){
	/* 
	 Returns random int in range [0, maxVal]
	 Uses bcnrand.
	 
	 Input: 
	 int maxVal		Maximum value
	 
	 Output:
	 random integer in specified range
	 */ 
	return floor( maxVal * bcnrand() ) ;
}


unsigned int getRandomUnsigned(){
	/* 
	 Returns random int in range [0, UINT_MAX]
	 Uses bcnrand.
	 
	 Output:
	 random integer in specified range
	 */ 
	return (unsigned int) floor( UINT_MAX * bcnrand() ) ;
}


double read_timer( ){
	/*
	 Timer. 
	 Returns elapsed time since initialization as a double. 
	 
	 Output:
	 double (returned)   Elapsed time. 
	 */ 
	
    static char initialized = 0;
    static struct timeval start;
    struct timeval end;
	
    if( !initialized ){
        gettimeofday( &start, NULL );
        initialized = 1;
    }
	
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}


double bcnrand( ){
	/*
	 This routine generates a sequence of IEEE 64-bit floating-point pseudorandom
	 numbers in the range (0, 1), based on the recently discovered class of
	 normal numbers described in the paper "Random Generators and Normal Numbers"
	 by David H. Bailey and Richard Crandall, available at http://www.nersc.gov/~dhbailey.
	 
	 Internal, private variables are allocated as static and should not be modified by the user.
	 User should simply call the routine for each random number desired. 
	 
	 Parameter: 
	 static double startingIndex       Users may modify this parameter to control where in the sequence the generator starts. 
	 This facility is useful for parallel implementations of this generator
	 to ensure that each processor gets the correct portion of the sequence.  
	 */
	
	static char initialized = 0;
	
	// variables
	static double d1, d2, d3;
	static dd_real dd1, dd2, dd3;
	
	// constants
	static double threeToTheThirtyThree ;
	static double reciprocalOfThreeToThirtyThree ;
	static double twoToTheFiftyThree ;
	
	static double startingIndex ;
	
	static unsigned long long int n;
	
	if( !initialized ){
		
		n = 0;
		startingIndex = pow(3.0, 33) + 10000.0 ;
		
		// parameters
		threeToTheThirtyThree = pow(3, 33) ;
		reciprocalOfThreeToThirtyThree = 1.0 / threeToTheThirtyThree ;
		twoToTheFiftyThree = pow(2, 53);
		
		// check inputs
		if( (startingIndex < threeToTheThirtyThree + 100) || (startingIndex > twoToTheFiftyThree) ){
			fprintf(stderr, "bcnrand error: startingIndex must satisfy 3^33 + 100 <= startingIndex <= 2^53\nstartingIndex = %.15f", startingIndex);
			exit(-1) ;
		}
		
		// calculate starting element
		d2 = expm2(startingIndex - threeToTheThirtyThree, threeToTheThirtyThree) ;
		d3 = trunc(0.5 * threeToTheThirtyThree) ;
		dd1 = ddMult(d2, d3) ;
		d1 = trunc(reciprocalOfThreeToThirtyThree * dd1.x[0]) ;
		dd2 = ddMult(d1, threeToTheThirtyThree) ;
		dd3 = ddSub(dd1, dd2) ;
		d1 = dd3.x[0] ;
		
		if(d1 < 0.0)
			d1 += threeToTheThirtyThree ;
		
		initialized = 1;
	}
	
	n++ ;
	if( n > (2.0 * threeToTheThirtyThree / 3.0) ){
		fprintf(stderr, "bcnrand warning: number of elements exceeds period.\nn = %llu\nperiod = %f", n, 2.0 * twoToTheFiftyThree / 3.0) ;
	}
	
	// calculate next element of sequence
	dd1.x[0] = twoToTheFiftyThree * d1 ;
	dd1.x[1] = 0.0;
	d2 = trunc(twoToTheFiftyThree * d1 / threeToTheThirtyThree) ;
	dd2 = ddMult(threeToTheThirtyThree, d2) ;
	dd3 = ddSub(dd1, dd2);
	d1 = dd3.x[0];
	
	if(d1 < 0.0)
		d1 += threeToTheThirtyThree ;
	
	return reciprocalOfThreeToThirtyThree * d1 ;
}

double expm2(double p, double modulus){
	/*
	 Returns 2^p mod am.  This routine uses a left-to-right binary scheme.
	 
	 Input:
	 double p            Exponent.
	 double modulus      Modulus.
	 
	 Output:
	 double (returned) 2^p mod am
	 */
	
	double reciprocalOfModulus;
	double d2;
	dd_real dd1, dd2, dd3, ddModulus;
	double p1, pt1, r;
	
	
	double twoToTheFiftyThree = pow(2, 53);
	
	reciprocalOfModulus = 1.0 / modulus;
	p1 = p ;
	pt1 = twoToTheFiftyThree;
	r = 1.0 ;
	ddModulus.x[0] = modulus;
	ddModulus.x[1] = 0.0;
	
	while(1){
		
		if(p1 >= pt1){
			// r = mod (2.d0 * r, am)
			dd1 = ddMult(2.0, r) ;
			if( dd1.x[0] > modulus){
				dd2 = ddSub(dd1, ddModulus) ;
				dd1 = dd2 ;
			}
			r = dd1.x[0] ;
			p1 -= pt1 ;
		}
		
		pt1 *= 0.5 ;
		
		if(pt1 >= 1.0){
			//r = mod (r * r, am)
			dd1 = ddMult(r, r) ;
			dd2.x[0] = reciprocalOfModulus * dd1.x[0] ;
			d2 = trunc(dd2.x[0]) ;
			dd2 = ddMult(modulus, d2) ;
			dd3 = ddSub(dd1, dd2) ;
			r = dd3.x[0] ;
			
			if(r < 0.0)
				r += modulus ;
		}
		else
			break;
	}
	
	return r;
}


dd_real ddMult(double da, double db){
	/*
	 Returns res = a * b, where res is double double precision
	 
	 Input:
	 double da, db         Values to multiply
	 
	 Output:
	 returned (dd_real)    Their products in double double precision
	 */
	
	dd_real res ;
	
	double split = 134217729.0 ;
	
	double a1, a2, b1, b2, cona, conb ;
	
	cona = da * split ;
	conb = db * split ;
	a1 = cona - (cona - da) ;
	b1 = conb - (conb - db) ;
	a2 = da - a1 ;
	b2 = db - b1 ;
	
	res.x[0] = da * db ;
	res.x[1] = (((a1 * b1 - res.x[0]) + a1 * b2) + a2 * b1) + a2 * b2 ;
	
	return res;
	
}

dd_real ddSub(dd_real a, dd_real b){
	/*
	 Returns res = a - b, where res is double double precision
	 
	 Input:
	 double da, db         Values to subtract
	 
	 Output:
	 returned (dd_real)    Their difference in double double precision
	 */
	
	double e, temp1, temp2;
	dd_real res ;
	
	temp1 = a.x[0] - b.x[0] ;
	e = temp1 - a.x[0] ;
	temp2 = ((-b.x[0] - e) + (a.x[0] - (temp1 - e))) + a.x[1] - b.x[1] ;
	
	res.x[0] = temp1 + temp2 ;
	res.x[1] = temp2 - (res.x[0] - temp1) ;
	
	return res;
}


