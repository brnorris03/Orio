


#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// length of array to sort
#define BYTES 100


/*
 100 byte character array sort. 
 Essentially a reference implementation for the sort benchmark. 
 
 Information on general guidelines and I/O format available at:
 http://sortbenchmark.org/
 Source code to test for dataset generation and output verification at:
 http://www.ordinal.com/gensort.html
 
 
 Alex Kaiser, LBNL, 4/16/10.
 */ 


void quickSort(char **v, int left, int right); 

void mergeSort(char **x, int lenX); 
void mergeHelper(char **x, int lenX, char **buffer); 
void merge(char **x, int lenX, char **buffer); 

void swap(char **v, int i, int j); 
char verifySorted(char **v, int size); 

// timer
double read_timer( );

int main(){
	
	printf("100 byte sort test:\n");
	
	int i; 
	int size = 1000; // number of records to sort
	
	char allPass ; 
		
	char **v = (char **) malloc(size * sizeof(char *)); 
	FILE *in = fopen("input.txt", "r");
	
	if (in == NULL) {
		fprintf(stderr, "File 'input.txt' not found.\nExiting.\n\n"); 
		return -1; 
	}
	
	
	for(i=0; i<size; i++){
		v[i] = (char *) malloc(BYTES + 1) ; // allocate directly as 100 byte arrays
		fgets(v[i], BYTES + 1, in); 
	}
	
	double startTime, endTime; 
	startTime = read_timer(); 
	
	// sort contents
	// quick sort or merge sort may be selected
	quickSort(v, 0, size-1); 
	//mergeSort(v, size); 
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	// output array if desired
	fclose(in); 
	FILE * out = fopen("output.txt", "w"); 
	for(i=0; i<size; i++)
		fprintf(out, "%s", v[i]); 
	
	allPass = verifySorted(v, size) ; 
	 
	if( allPass )
		printf("Array sorted properly. Check with Sort Benchmark verification for completeness.\nTest passed.\n\n");
	else
		fprintf(stderr, "Array not in ascending order. Check with Sort Benchmark verification for completeness.\nTest failed.\n\n");
	
	printf("End of 100 byte sort test.\n\n\n");
	
	return allPass; 
}


void quickSort(char **v, int left, int right){ 
	/* 
	 Simple quick sort 
	 Taken directely from "The C Programming Language", Kernighan and Ritchie.
	 Sort v[left]...v[right] into increasing order. 
	 
	 Input:
			char **v	Array of strings to sort
			int left	First index of array to sort
			int right	Last index of array to sort
	 
	 Output:
			char **v	Array of strings, sorted in the specified region
	
	*/ 
	
	int i, last; 
	
	if (left >= right) /* do nothing if array contains */ 
		return;        /* fewer than two elements */ 
	
	swap(v, left, (left + right)/2); /* move partition elem */ 
	last = left;                     /* to v[0] */ 
	
	for (i = left + 1; i <= right; i++)  /* partition */ 
		if( strncmp(v[i], v[left], 10) < 0 )  //if (v[i] < v[left])
			swap(v, ++last, i); 
	
	swap(v, left, last);            /* restore partition  elem */ 
	
	quickSort(v, left, last-1); 
	quickSort(v, last+1, right); 
} 

void swap(char **v, int i, int j) {
	
	/*   
	 Interchange v[i] and v[j] in place
	 
	 Input:
			char **v	Array of strings to swap
			int i		First value to swap
			int j		Last value to swap
	 
	 Output:
			char **v	Array of strings, with specified values swapped
	 */ 
	
	char *temp; 
	temp = v[i]; 
	v[i] = v[j]; 
	v[j] = temp; 
}


void mergeSort(char **x, int lenX){
	/* 
	 Wrapper function for merge sort. 
	 Sorts x[0] ... x[lenX - 1]. 
	 Allocates the necessary memory for buffers. 
	 
	 Input: 
		char **x		Array of strings to sort
		int lenX		Length of array to sort
	 
	 Output: 
		char **x		Sorted array
	 */ 
	
	char **buffer = (char **) malloc( (lenX / 2) * sizeof(char *) ) ; 
	int j; 
	for(j=0; j<(lenX/2); j++)
		buffer[j] = (char *) malloc(BYTES) ; 
	
	mergeHelper(x, lenX, buffer) ; 	
}


void mergeHelper(char **x, int lenX, char **buffer){
	/*
	 Main recursives merge sort routine. 
	 Sorts x[0] ... x[lenX - 1] 
	 
	 Input: 
		char **x		Array of strings to sort
		int lenX		Length of array to sort
		char **buffer	Buffer array for moving data
	 
	 Output: 
		char **x		Sorted array
	 */ 
	
	if( lenX > 1 ){
		mergeHelper(x, lenX/2, buffer); 
		mergeHelper(x+(lenX/2), (lenX - lenX/2), buffer); 
		merge(x, lenX, buffer); 
	}
}


void merge(char **x, int lenX, char **buffer){
	/*
	Merge two halves of x. 
	
	Buffer must be of sufficient length, that is at least floor(lenX/2)	
	Half of x is initially copied to the buffer. 
	If length of x is odd, then length of buffer is floor(lenX/2)
		and ceil(lenX/2) is left in the x array. 
	 
	 Input: 
		char **x		Array of strings to sort
		int lenX		Length of array to sort
		char **buffer	Buffer array for moving data
		
		Output: 
			char **x		Sorted array
	*/ 
	
	int j; 
	int xIndex, bufIndex, lenBuffer; 
	lenBuffer = lenX / 2 ; 
	
	// copy x to buffer
	for(j=0; j < lenBuffer; j++)
		buffer[j] = x[j] ; 
	
	j=0;
	bufIndex = 0; 
	xIndex = lenBuffer ; 
	
	while( (xIndex < lenX) && (bufIndex < lenBuffer) ){
		
		if( strncmp(x[xIndex], buffer[bufIndex], 10) < 0 ){ 
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


char verifySorted(char **v, int size){ 
	/*
	 Simple check that list is sorted. 
	 
	 Input:
			char **v			Array to check
			int size			Number of elements to check 
	 
	 Output:
			char (returned)		Whether array is in proper lexicographical order
	 
	 */ 
	
	int i; 
	for (i=0; i < (size - 1); i++) {
		if(strncmp(v[i], v[i+1], 10) > 0 ) // if value of first string is larger than second, error. 
			return 0; 
	}
	return 1; 
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


