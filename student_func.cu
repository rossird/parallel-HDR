/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#define MIN_OP 1
#define MAX_OP 2
#define SUM_OP 3

__device__ float min_float(float a, float b) {
    return a < b ? a : b;
}

__device__ float max_float(float a, float b) {
    return a < b ? b : a;
}

//Taken from blackboard
__global__ void global_reduce_kernel(float * d_out, float * d_in, int REDUCE_OP)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            switch(REDUCE_OP) {
                case MIN_OP:
                    d_in[myId] = min_float(d_in[myId], d_in[myId + s]);
                    break;
                case MAX_OP:
                    d_in[myId] = max_float(d_in[myId], d_in[myId + s]);
                    break;
                case SUM_OP:
                    d_in[myId] += d_in[myId + s];
                    break;
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

//Taken from blackboard
//Hillis & Steele: Kernel Function
//Altered by Jake Heath, October 8, 2013 (c)
__global__ 
	 void scanKernel(unsigned int *out_data, unsigned int *in_data, size_t numElements){
	 	//we are creating an extra space for every numElement so the size of the array needs to be 2*numElements
	 	//cuda does not like dynamic array in shared memory so it might be necessary to explicitly state
	 	//the size of this mememory allocation
	 	__shared__ int temp[1024 * 2]; 

	 	//instantiate variables
		int id = threadIdx.x;
		int pout = 0, pin = 1;

		// // load input into shared memory. 
		// // Exclusive scan: shift right by one and set first element to 0
		temp[id] = (id > 0) ? in_data[id - 1] : 0;
		__syncthreads();
		

		//for each thread, loop through each of the steps
		//each step, move the next resultant addition to the thread's 
		//corresponding space to manipulted for the next iteration
		for( int offset = 1; offset < numElements; offset <<= 1 ){
			//these switch so that data can move back and fourth between the extra spaces
			pout = 1 - pout; 
			pin = 1 - pout;

			//IF: the number needs to be added to something, make sure to add those contents with the contents of 
				//the element offset number of elements away, then move it to its corresponding space
			//ELSE: the number only needs to be dropped down, simply move those contents to its corresponding space
			if (id >= offset) {
				//this element needs to be added to something; do that and copy it over
				temp[pout * numElements + id] = temp[pin * numElements + id] + temp[pin * numElements + id - offset]; 
			} else {
				 //this element just drops down, so copy it over
			    temp[pout * numElements + id] = temp[pin * numElements + id];
			}
		 	__syncthreads();
		}
		// write output
		out_data[id] = temp[pout * numElements + id]; 
	}//scanKernel

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in, int REDUCE_OP)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            switch(REDUCE_OP) {
                case MIN_OP:
                    sdata[tid] = min_float(sdata[tid], sdata[tid + s]);
                    break;
                case MAX_OP:
                    sdata[tid] = max_float(sdata[tid], sdata[tid + s]);
                    break;
                case SUM_OP:
                    sdata[tid] += sdata[tid + s];
                    break;
            }
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, bool usesSharedMemory, int REDUCE_OP)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 384;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_in, REDUCE_OP);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_intermediate, d_in, REDUCE_OP);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_out, d_intermediate, REDUCE_OP);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate, REDUCE_OP);
    }
}

//Calculate a histogram
__global__
void histogram(const float* const lum, float lumMin, float lumRange, 
               size_t numBins, unsigned int* histogram) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int bin = (lum[i] - lumMin) / lumRange * numBins;
    //bin = min(bin, static_cast<int>(numBins - 1));
    atomicAdd(&histogram[bin], 1);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    unsigned int* d_histogram;
    float* d_in;
    float* d_intermediate;
    float* d_out;
    float range;
    
    //Allocate memory
    checkCudaErrors(cudaMalloc(&d_histogram, sizeof(int) * numBins));
    checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));
    
    //Initialize memory
    checkCudaErrors(cudaMemset(d_in, 0, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, sizeof(float) * numCols * numRows, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_intermediate, 0, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMemset(d_out, 0, sizeof(float)));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * numBins));
    
    //Min scan to find min_logSum
    reduce(d_out,
           d_intermediate,
           d_in,
           numCols * numRows,
           false,
           MIN_OP);
           
    checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    //Initialize memory
    checkCudaErrors(cudaMemset(d_in, 0, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, sizeof(float) * numCols * numRows, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_intermediate, 0, sizeof(float) * numCols * numRows * 2));
    checkCudaErrors(cudaMemset(d_out, 0, sizeof(float)));
    
    //Max scan to find max_logSum
    reduce(d_out,
           d_intermediate,
           d_in,
           numCols * numRows * 2,
           false,
           MAX_OP);
         
    checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
     
    //Find range
    range = max_logLum - min_logLum;
    
    //Generate histogram
    histogram<<<numCols, numRows>>>(d_logLuminance,
                                    min_logLum,
                                    range,
                                    numBins,
                                    d_histogram);
    
    //Exclusive scan to find cdf
    scanKernel<<<1, numBins>>>(d_cdf, 
                               d_histogram,
                               numBins);

    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
    cudaFree(d_histogram);
}
