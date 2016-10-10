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

#include "utils.h"

__device__ float min_float(float a, float b) {
    return a < b ? a : b;
}

//Calculate a min scan
__global__
void min_scan(float* array, int size, float* min) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int offset = 1; offset < size; offset *= 2) {
        if (i >= offset) {
            array[i] = min_float(array[i - offset], array[i]);
        }
    }
    
    //*min = array[size-1];

}

//Calculate a max scan
__global__
void max_scan(const float* array, size_t size, float* max) {

}

//Calculate a histogram
__global__
void histogram(const float* const lum, float lumMin, float lumRange, size_t numBins, int* histogram) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int bin = (lum[i] - lumMin) / lumRange * numBins;
    atomicAdd(&histogram[bin], 1);
}

//Calculate an exclusive scan
__global__
void exclusive_scan(float* input, size_t size, unsigned int* const cdf) {

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
    
    int* d_histogram;
    float* d_max;
    float* d_min;
    float* d_temp;
    float range;
    
    //Allocate memory
    checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_histogram, sizeof(int) * numBins));
    checkCudaErrors(cudaMalloc(&d_temp, sizeof(float) * numCols * numRows));
    
    //Initialize memory
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * numBins));
    checkCudaErrors(cudaMemcpy(d_temp, d_logLuminance, sizeof(float) * numCols * numRows, cudaMemcpyDeviceToDevice));
    
    //Min scan to find min_logSum
    min_scan<<<numRows, numCols>>>(d_temp, numCols * numRows, d_min);
    //checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    
    //Max scan to find max_logSum
    //max_scan<<<numCols, numRows>>>(d_logLuminance, numCols * numRows, d_max);
    //checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    
    //Find range
    range = max_logLum - min_logLum;
    
    //Generate histogram
    //histogram<<<numCols, numRows>>>(d_logLuminance, min_logLum, range, numBins, d_histogram);
    
    //Exclusive scan to find cdf
    //exclusive_scan<<<numCols, numRows>>>(d_logLuminance, numCols * numRows,  d_cdf);
    
    //free(d_max);
    //free(d_min);
    //free(d_histogram);
}
