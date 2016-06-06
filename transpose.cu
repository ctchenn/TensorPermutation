/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

const int TILE_DIM = 32;
const int BLOCK_ROWS = 2;
const int NUM_REPS = 100;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  /*
  printf("\nreference:\n");
  
  for (int i=0; i<16; i++) {
      for (int j=0; j<16; j++) {
          printf("(");
          for (int k=0; k<2; k++)
              printf("%3.0f ", ref[i*2*16 + j * 2 + k]);
          printf(")");
      }
      printf("\n");
  }
  printf("\n");
  printf("\nresult:\n");
  for (int i=0; i<16; i++) {
      for (int j=0; j<16; j++) {
          printf("(");
          for (int k=0; k<2; k++)
              printf("%3.0f ", res[i*16*2 + j * 2 + k]);
          printf(")");
      }
      printf("\n");
  }
  printf("\n");
  */
  
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
__global__ void copy(float *odata, const float *idata, int nx, int ny)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  if(!(x<nx && y<ny)) return; 

  for (int j = 0; j < TILE_DIM && y+j < ny; j+= BLOCK_ROWS)
    //odata[(y+j)*width + x] = idata[(y+j)*width + x];
    odata[(y+j)*nx + x] = idata[(y+j)*nx + x];
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
__global__ void copySharedMem(float *odata, const float *idata, int nx, int ny)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
  
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  if(!(x<nx && y<ny)) return; 


  for (int j = 0; j < TILE_DIM && y+j < ny; j += BLOCK_ROWS)
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*nx + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM && y+j < ny; j += BLOCK_ROWS)
     odata[(y+j)*nx + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *odata, const float *idata, const int* sizes, const int* perm, const int dim)
{
  int pos0 = blockIdx.x * TILE_DIM + threadIdx.x;
  int pos1 = blockIdx.y * TILE_DIM + threadIdx.y;
  int pos2 = blockIdx.z * 1 + threadIdx.z;
  const int nx = sizes[0];
  const int ny = sizes[1];
  const int nz = sizes[2];

  if (perm[0] == 1 and perm[1] == 2)   // exchange j, k
      if (pos0<nz && pos1<ny && pos2<nx)
          for (int j = 0; j < TILE_DIM && pos1+j<ny; j += BLOCK_ROWS)
              odata[pos2*nz*ny + pos0*ny + pos1+j] = 
                  idata[pos2*ny*nz + (pos1+j)*nz + pos0];
  if (perm[0] == 0 and perm[1] == 2)   // i, k. pos0:z, pos1:x, pos2:y
      if (pos0<nz && pos1<nx && pos2<ny)
          for (int j = 0; j < TILE_DIM && pos1+j<nx; j += BLOCK_ROWS)
              odata[pos0*nx*ny + pos2*nx + (pos1+j)] = 
                  idata[(pos1+j)*ny*nz + pos2*nz + pos0];
  if (perm[0] == 0 and perm[1] == 1)   // i, j
      if (pos0<nz && pos1<ny && pos2<nx)
          for (int j = 0; j < TILE_DIM && pos1+j<ny; j += BLOCK_ROWS)
              odata[(pos1+j)*nx*nz + pos2*nz + pos0] = 
                  idata[pos2*ny*nz + (pos1+j)*nz + pos0];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.

__global__ void transposeCoalesced(float *odata, const float *idata, int nx, int ny)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  if(x<nx && y<ny) { 
     for (int j = 0; j < TILE_DIM && y+j < ny; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*nx + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if(x<ny && y<nx) {
     for (int j = 0; j < TILE_DIM && y+j < nx; j += BLOCK_ROWS)
        odata[(y+j)*ny + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}
  

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata, int nx, int ny)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  if(x<nx && y<ny) { 
     for (int j = 0; j < TILE_DIM && y+j < ny; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*nx + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if(x<ny && y<nx) {
     for (int j = 0; j < TILE_DIM && y+j < nx; j += BLOCK_ROWS)
        odata[(y+j)*ny + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeInplace(float *odata, const float *idata, const int* sizes, const int* perm, const int dim)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int idx0 = threadIdx.x;
  int idx1 = threadIdx.y;
  int idx2 = threadIdx.z;
  int pos0 = blockIdx.x * TILE_DIM + idx0;  // transpose block offset
  int pos1 = blockIdx.y * TILE_DIM + idx1;
  int pos2 = blockIdx.z * 1 + idx2;   
  int postpos0 = blockIdx.y * TILE_DIM + idx0;
  int postpos1 = blockIdx.x * TILE_DIM + idx1;
  assert(threadIdx.z==0);
  const int nx = sizes[0];
  const int ny = sizes[1];
  const int nz = sizes[2];

  // idx0 --> z, idx1 --> y, idx2 --> x 
  if (perm[0] == 1 and perm[1] == 2) {  // exchange j, k
      if (pos2<nx && pos0 < nz) {
          for (int j = 0; j < TILE_DIM && pos1+j<ny; j += BLOCK_ROWS)
              tile[idx1+j][(idx0+idx1+j)%TILE_DIM] = 
                  idata[pos2*ny*nz + (pos1+j)*nz + pos0];
      }
      __syncthreads();
      if (pos2<nx && postpos0<ny) {
          for (int j = 0; j < TILE_DIM && postpos1+j<nz; j += BLOCK_ROWS)
              odata[pos2*nz*ny + (postpos1+j)*ny + postpos0] = 
                  tile[idx0][(idx1+j+idx0)%TILE_DIM];
      }
  }

  // idx0: z, idx1: x, idx2: y
  if (perm[0] == 0 and perm[1] == 2) {  // i, k
      if (pos2<ny && pos0<nz) {
          for (int j = 0; j < TILE_DIM && pos1+j<nx; j += BLOCK_ROWS)
              tile[idx1+j][(idx0+idx1+j)%TILE_DIM] = 
                  idata[(pos1+j)*ny*nz + pos2*nz + pos0];
      }
      __syncthreads();
      if (pos2<ny && postpos0 < nx) {
          for (int j = 0; j < TILE_DIM && postpos1+j<nz; j += BLOCK_ROWS)
              odata[(postpos1+j)*ny*nx + pos2*nx + postpos0] = 
                  tile[idx0][(idx1+j+idx0)%TILE_DIM];
      }
  }

  // idx0: z, idx1: y, idx2: x
  if (perm[0] == 0 && perm[1] == 1)  // i, j
      if (pos2<nx && pos0<nz)
          for (int j = 0; j < TILE_DIM && pos1+j<ny; j += BLOCK_ROWS)
              odata[(pos1+j)*nx*nz + pos2*nz + pos0] = 
                  idata[pos2*ny*nz + (pos1+j)*nz + pos0];
}

int main(int argc, char **argv)
{
  const int dim = 3;
  int sizes[3];
  sizes[0] = atoi(argv[1]);
  sizes[1] = atoi(argv[2]);
  sizes[2] = atoi(argv[3]);
  int perm[2];  // permuted dimensions in ascending order
  perm[0] = atoi(argv[4]);
  perm[1] = atoi(argv[5]);
  int scale = 1;
  for (int i = 0; i < dim; i++)
      scale *= sizes[i];  
  const int mem_size = scale*sizeof(float);
  
  int nx = sizes[0];
  int ny = sizes[1];
  int nz = sizes[2];
  int n1,n2,n0;

  // should revise when dim > 3
  // exchange (0,1) is different from (0,2) and (1,2) 
  // always assign threadIdx.x to z direction, 
  n0 = sizes[2];
  n1 = (perm[1]==1)?sizes[1]:sizes[perm[0]]; 
  n2 = sizes[0]+sizes[1]+sizes[2]-n0-n1;

  dim3 dimGrid((n0-1)/TILE_DIM+1, (n1-1)/TILE_DIM+1, n2);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 2;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("perm order: (%d, %d)\n", perm[0], perm[1]);
  printf("Matrix size: %d %d %d, Block size: %d %d, Tile size: %d %d\n", 
         nx, ny, nz, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  checkCuda( cudaSetDevice(devId) );

  float *h_idata = (float*)malloc(mem_size);
  float *h_cdata = (float*)malloc(mem_size);
  float *h_tdata = (float*)malloc(mem_size);
  float *gold    = (float*)malloc(mem_size);
  
  float *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );
  int *d_sizes, *d_perm;
  checkCuda( cudaMalloc(&d_sizes, 3*sizeof(int)) );
  checkCuda( cudaMalloc(&d_perm, 2*sizeof(int)) );

  // check parameters and calculate execution configuration
  /*
  if (nx % TILE_DIM || ny % TILE_DIM) {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    goto error_exit;
  }

  if (TILE_DIM % BLOCK_ROWS) {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    goto error_exit;
  }
  */
    
  // host
  for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
              h_idata[i*ny*nz + j*nz + k] = i*ny*nz + j*nz + k;

  // correct result for error checking
  for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++) {
              if (perm[0] == 1 && perm[1] == 2)  // exchange j, k
                  gold[i*ny*nz + k*ny + j] = h_idata[i*ny*nz + j*nz + k];
              else if(perm[0] == 0 && perm[1] == 2)  // i, k
                  gold[k*nx*ny + j*nx + i] = h_idata[i*ny*nz + j*nz + k];
              else if(perm[0] == 0 && perm[1] == 1)  // i, j
                  gold[j*nz*nx + i*nz + k] = h_idata[i*ny*nz + j*nz + k];
          } 
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_perm, perm, 2*sizeof(int), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(d_sizes, sizes, 3*sizeof(int), cudaMemcpyHostToDevice) );
  
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  // ------------
  // time kernels
  // ------------
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
 /* 
  // ----
  // copy 
  // ----
  printf("%25s", "copy");
  checkCuda( cudaMemset(d_cdata, 0, mem_size) );
  // warm up
  copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(h_idata, h_cdata, nx*ny, ms);

  // -------------
  // copySharedMem 
  // -------------
  printf("%25s", "shared memory copy");
  checkCuda( cudaMemset(d_cdata, 0, mem_size) );
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(h_idata, h_cdata, nx * ny, ms);
*/

  // --------------
  // transposeNaive 
  // --------------
  printf("%25s", "naive transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata, d_sizes, d_perm, dim);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata, d_sizes, d_perm, dim);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, nx*ny*nz, ms);
/*
  // ------------------
  // transposeCoalesced 
  // ------------------
  printf("%25s", "coalesced transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata,nx,ny);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, nx * ny, ms);
*/
  // ------------------------
  // transposeInplace
  // ------------------------
  printf("%25s", "In-place transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  transposeInplace<<<dimGrid, dimBlock>>>(d_tdata, d_idata, d_sizes, d_perm, dim);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     transposeInplace<<<dimGrid, dimBlock>>>(d_tdata, d_idata, d_sizes, d_perm, dim);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, nx*ny*nz, ms);
  // error_exit:
  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaFree(d_tdata) );
  checkCuda( cudaFree(d_cdata) );
  checkCuda( cudaFree(d_idata) );
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}
