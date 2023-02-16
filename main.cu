#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"
#include "handle_image_header.h"


#define RED 0.299
#define GREEN 0.587
#define BLUE 0.114



int temp=0;

typedef unsigned char byte;
typedef unsigned int dword;
typedef unsigned short int word;



int CONVOLUTION_CUDA(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH, FILE *fp,
                     char * new_name, int filterSize, float factor,int NUMBER_BLOCKS,int NUMBER_THREADS_PER_BLOCK);



kernel * getKernel(int filterSize){

    kernel *retKernel= NULL;
    int size = filterSize*2+1;
    float  **filter= NULL;

    cudaMallocManaged(&retKernel,sizeof (kernel));
    cudaMallocManaged(&filter,size*sizeof (float *));

    for (int k = 0; k <size ; ++k)
        cudaMallocManaged(&filter[k],size*sizeof (float ));

    //If filterSize = 1 then return a 3x3 filter for edge detection.
    if (filterSize==1){
        filter[0][0]=-1;filter[0][1]=-1;filter[0][2]=-1;
        filter[1][0]=-1;filter[1][1]=8;filter[1][2]=-1;
        filter[2][0]=-1;filter[2][1]=-1;filter[2][2]=-1;
        retKernel->factor = 1;
        retKernel->bias = 0;
        retKernel->kernel=filter;
    }
        //If filterSize = 2 return a 5x5 Kernel for edge detection.
    else if (filterSize==2){
        filter[0][0]=-1;filter[0][1]=0;filter[0][2]=0;filter[0][3]=0;filter[0][4]=0;
        filter[1][0]=0;filter[1][1]=-2;filter[1][2]=0;filter[1][3]=0;filter[1][4]=0;
        filter[2][0]=0;filter[2][1]=0;filter[2][2]=6;filter[2][3]=0;filter[2][4]=0;
        filter[3][0]=0;filter[3][1]=0;filter[3][2]=0;filter[3][3]=-2;filter[3][4]=0;
        filter[4][0]=0;filter[4][1]=0;filter[4][2]=0;filter[4][3]=0;filter[4][4]=-1;
        retKernel->factor = 1;
        retKernel->bias = 0;
        retKernel->kernel=filter;
    }
        //Otherwise return a Mean filter of Size N*N.
    else {
        for (int i = 0; i <size ; i++) {
            for(int j=0;j<size;j++)
                if(i==j)
                    filter[i][j]=1;
        }
        retKernel->factor = 1.0f/9.0f;
        retKernel->bias = 0;
        retKernel->kernel = filter;
    }
    return retKernel;
}




tbyte **createImage(BITMAP_INFO_HEADER *BIH,int padding){
    tbyte **PIXEL;
    PIXEL= (tbyte **) malloc((BIH->biHeight) * sizeof(tbyte *));
    PIXEL[0]=(tbyte *)malloc((BIH->biHeight)*(BIH->biWidth + padding) * sizeof(tbyte));

    for (int k=1;k<(BIH->biHeight);k++)
        PIXEL[k]=PIXEL[k-1]+(BIH->biWidth + padding);

    return PIXEL;
}

void readImage(BITMAP_INFO_HEADER *BIH,int padding,FILE *fp,tbyte **PIXEL){

    //DIABAZI TA PIXEL APO TO ARXIO KAI TA KATAXORI STON PINAKA.
    for (int i_WIDTH = 0; i_WIDTH < (BIH->biHeight); i_WIDTH++) {
        for (int i_HEIGHT = 0; i_HEIGHT < (BIH->biWidth + padding); i_HEIGHT++) {
            //AN EXI TELIOSI H GRAMI THS EIKONAS DIABAZI ENA ENA TA BYTE TOY PADDING ALLIOS DIABAZI PIXEL.
            if (i_HEIGHT >= BIH->biWidth)
                fread((void *) &(PIXEL[i_WIDTH][i_HEIGHT]), sizeof(byte), 1,fp);
            else
                fread((void *) &(PIXEL[i_WIDTH][i_HEIGHT]), sizeof(tbyte), 1,fp);
        }
    }
}



__device__ void pixelProcessor(tbyte *PIXEL, tbyte *PIXEL_OUT, int i_WIDTH, int i_HEIGHT, int filterSize, float *filter[],float bias,float factor,float multFactor,int one_dimension_index,int cols){
    int i,j;
    PIXEL_OUT[one_dimension_index].R = 0;
    PIXEL_OUT[one_dimension_index].G = 0;
    PIXEL_OUT[one_dimension_index].B = 0;
    float red=0;
    float green=0;
    float blue=0;
    for (j = i_HEIGHT-filterSize;  j<= i_HEIGHT + filterSize; j++)
        for (i = i_WIDTH-filterSize;  i<= i_WIDTH + filterSize; i++){
            int row = j-(i_HEIGHT-filterSize);
            int col = i-(i_WIDTH-filterSize);
            float f1 = filter[col][row];

            red   += (int) ((PIXEL[i*cols+j].R) * f1)+ PIXEL[one_dimension_index].R * multFactor;
            green += (int) ((PIXEL[i*cols+j].G) * f1)+ PIXEL[one_dimension_index].G * multFactor;
            blue  += (int) ((PIXEL[i*cols+j].B) * f1)+ PIXEL[one_dimension_index].B * multFactor;

        }
    PIXEL_OUT[one_dimension_index].R = min(max( (int)(factor * red + bias), 0), 255);
    PIXEL_OUT[one_dimension_index].G = min(max( (int)(factor * green + bias), 0), 255);
    PIXEL_OUT[one_dimension_index].B = min(max( (int)(factor * blue + bias), 0), 255);
}





__global__ void kernelConvolution
        (kernel *kernel1,int filterSize,unsigned int rows,unsigned int cols,
         tbyte *dev_PIXEL, tbyte *dev_PIXEL_OUT,BITMAP_INFO_HEADER BIH,float factor,
         int NUMBER_BLOCKS,int NUMBER_THREADS_PER_BLOCK) {

    /**
     * Here it's worth noticing that the job is assigned to Cuda threads cyclically.
     * The access pattern to global memory can have a huge impact on the performance
     * of a kernel. Here, we detail the concept of coalesced memory accesses. As you many know threads are arranged into
     * groups of 32 known as a warp. Accesses to global memory in CUDA are coalesced such that 32-, 64- and 128-byte
     * accesses are loaded in a single transaction. In my implementation the way pixels are processed is the following:
     *
     * For example if the grid has 3*1 blocks, each block has 4 threads. Then the threads of Block(2,0) will access the pixels cyclically:
     * Block 2, Thread: 0  1  2  3
     *                  8  9  10 11  +=12
     *                  20 21 22 23  +=12
     *                  32 33 34 35  +=12 etc.
     */
    unsigned int n=rows*cols;
    unsigned int threadUniqueID = threadIdx.x + blockIdx.x * blockDim.x;

    //PERNI ENA-ENA TA PIXEL THS EIKONAS.
    for (unsigned int one_dimension_index = threadUniqueID; one_dimension_index<n; one_dimension_index+=blockDim.x*gridDim.x) {

        unsigned int i_WIDTH = one_dimension_index/cols;
        unsigned int i_HEIGHT= one_dimension_index%cols;
        if (i_WIDTH<filterSize || i_HEIGHT<filterSize)
            continue;
        if (i_WIDTH >= (BIH.biHeight)-filterSize)
            continue;
        if(i_HEIGHT >= (BIH.biWidth)-filterSize)
            continue;

        if ((i_WIDTH > filterSize) && (i_WIDTH < (BIH.biHeight) - filterSize)
            && (i_HEIGHT > filterSize) && (i_HEIGHT < (BIH.biWidth) - filterSize)) {
            pixelProcessor(dev_PIXEL, dev_PIXEL_OUT, i_WIDTH, i_HEIGHT,filterSize, kernel1->kernel,kernel1->bias,kernel1->factor,factor,one_dimension_index,cols);
        } else {
            byte NEW_PIXEL;
            NEW_PIXEL = (int) (dev_PIXEL[one_dimension_index].R * RED
                               + dev_PIXEL[one_dimension_index].G * GREEN
                               + dev_PIXEL[one_dimension_index].B * BLUE);
            dev_PIXEL_OUT[one_dimension_index].R = NEW_PIXEL;
            dev_PIXEL_OUT[one_dimension_index].G = NEW_PIXEL;
            dev_PIXEL_OUT[one_dimension_index].B = NEW_PIXEL;
        }
    }
}




// Convolution Function
int CONVOLUTION_CUDA(BITMAP_FILE_HEADER *BFH, BITMAP_INFO_HEADER *BIH, FILE *fp,
                     char * new_name, int filterSize, float factor,int NUMBER_BLOCKS,int NUMBER_THREADS_PER_BLOCK) {

    int i_WIDTH,i_HEIGHT;
    int padding;
    tbyte **PIXEL, **PIXEL_OUT;

    //Calculate padding of the image
    if (((BIH->biWidth * 3) % 4) == 0)
        padding = 0;
    else
        padding = 4 - ((3 * BIH->biWidth) % 4);

    /**
     * Explain how image is created.Important because you can transfer data to GPU using only cudaMemCpy
     * which just copies a number of contigious bits. So I had to make sure that the memory of the image
     * on Host is allocated sequentially.
     */
    PIXEL = createImage(BIH,padding);
    PIXEL_OUT = createImage(BIH,padding);
    readImage(BIH,padding,fp,PIXEL);
    fclose(fp);



    /**
     * Filter that will be applied to the image using convolution.
     * Placed in unified memory for simplicity.
     * This way it can be accessed by both GPU and CPU without cudaMemCpy
     */
    kernel *dev_Filter_Kernel;
    dev_Filter_Kernel = getKernel(filterSize);


    /**
     * dev_PIXEL contains the Pixels of the image that we want to do convolution.
     * Here we allocate memory in the GPU space. And then copy the image from the Host to the Device using memCpy
     *
     * dev_PIXEL_OUT will be the convolved image after the CUDA kernel has finished it's execution. Here we just allocate the memory of the host.
     */
    tbyte *dev_PIXEL, *dev_PIXEL_OUT;
    cudaMalloc(&dev_PIXEL,sizeof (tbyte)*(BIH->biHeight)*(BIH->biWidth + padding));
    cudaMemcpy(dev_PIXEL,PIXEL[0],(BIH->biHeight)*(BIH->biWidth + padding) * sizeof(tbyte) ,cudaMemcpyHostToDevice);
    cudaMalloc(&dev_PIXEL_OUT,sizeof (tbyte)*(BIH->biHeight)*(BIH->biWidth + padding));

    /**
     * Here we allocate the number of blocks and number of threads per block for our grid.
     */
    dim3 h_blockDim(NUMBER_THREADS_PER_BLOCK);
    dim3 h_gridDim(NUMBER_BLOCKS);

    kernelConvolution<<<h_gridDim, h_blockDim>>>(dev_Filter_Kernel,filterSize,(BIH->biHeight),(BIH->biWidth + padding) , dev_PIXEL, dev_PIXEL_OUT,*BIH,factor,NUMBER_BLOCKS,NUMBER_THREADS_PER_BLOCK);
    /**
     * After kernel is done.The convolved image is in dev_PIXEL_OUT and we copy the result from the Device to Host.
     */
    cudaMemcpy(PIXEL_OUT[0],dev_PIXEL_OUT,sizeof (tbyte)*(BIH->biHeight)*(BIH->biWidth + padding),cudaMemcpyDeviceToHost);


#ifdef DEBUG
    FILE *new_fp;
    new_fp=fopen(new_name,"wb");
    //DIMIOYRGA TO NEO FILE EIKONAS ME THN AKOLOYTHI SINARTISI.
    NEW_FILE_HEADER(BFH, BIH, new_fp);

    for(i_WIDTH=0; i_WIDTH<(BIH->biHeight); i_WIDTH++) {
        for(i_HEIGHT=0; i_HEIGHT<(BIH->biWidth + padding); i_HEIGHT++) {

            //AN EXI TELIOSI H GRAMI THS EIKONAS TOPOTHETI ENA ENA TA BYTE TOY PADDING ALLIOS TOPOTHETI PIXEL.
            if(i_HEIGHT>=BIH->biWidth)
                fwrite((void *)&(PIXEL_OUT[i_WIDTH][i_HEIGHT]),sizeof(byte),1,new_fp);
            else
                fwrite((void *)&(PIXEL_OUT[i_WIDTH][i_HEIGHT]),sizeof(tbyte),1,new_fp);
        }
    }

    //KLISIMO ARXIOY KAI APODEZMEYSI MNIMIS.
    fclose(new_fp);
#else
#endif

    free(PIXEL[0]);
    free(PIXEL);
    return (0);
}


#include <stdio.h>
#include <cuda_runtime.h>

void printCudaDeviceProp(int device)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    printf("  CUDA cores: %d\n", deviceProp.multiProcessorCount *
                                 deviceProp.maxThreadsPerMultiProcessor);
    printf("Device %d:\n", device);
    printf("  Name: %s\n", deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %lu bytes\n", deviceProp.totalGlobalMem);
    printf("  Total shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total constant memory: %lu bytes\n", deviceProp.totalConstMem);
    printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum dimension size of a block: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Maximum dimension size of a grid: (%d, %d, %d)\n", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Clock rate: %d kHz\n", deviceProp.clockRate / 1000);
    printf("  Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("  Number of registers per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size: %d\n", deviceProp.warpSize);
}



int main(int argc, char *argv[]) {

    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("Error: Unable to detect any CUDA-capable devices.\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        //printCudaDeviceProp(i);
    }



    BITMAP_FILE_HEADER BFH;
    BITMAP_INFO_HEADER BIH;
    FILE *fp;
    char convolvedImageName[128];

    //Call using:  ./a.out filterSize image.bmp NumBlocks NumThreadsPerBlock
    if (argc != 5){
        printf("Wrong Command Line. Format:./a.out FilterSize BMPimageName.bmp NumBlocks NumThreads\n");
        return 0;
    }

#ifdef DEBUG
    printf("Convolution process started...\n");
#endif
    startTime(0);
    // First Argument is for the Filter Size
    int filterSize = atoi(argv[1]);
    //This factor is going to be added to the filters just to make them a little bit different from each other.
    //This for loop basically perform convolution 10 times on an image with 10 slightly different kernels.
    //The main reason I added this loop is to make the serial program take longer to execute, so we can show that
    //cuda algorithm can make it much much faster. For example if the serial program takes 1 second then speedup of the
    //Cuda algorithm wouldn't be so obvious.
    for (float factor = 0.00; factor <= 0.1; factor += 0.005) {
        fp = fopen(argv[2], "rb");
        //If you can't read image header then terminate program
        if (load_HEADER(&BFH, &BIH, fp) == 1) exit(-1);
        sprintf(convolvedImageName, "output//conv_%dx%d_%3.2f_%s", filterSize, filterSize, factor, argv[2]);
#ifdef DEBUG
        printf("Conversion of %s: Factor:%f, Filter Size %dx%d:\n",argv[1],factor,filterSize,filterSize);
        LIST(&BFH, &BIH);
#else
#endif
        CONVOLUTION_CUDA(&BFH, &BIH, fp, convolvedImageName, filterSize, factor, atoi(argv[3]), atoi(argv[4]));
    }
    stopTime(0);
    elapsedTime(0);
    return (0);

}