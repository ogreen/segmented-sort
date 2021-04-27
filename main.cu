// -*-c++-*-

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* 
   The segmented sort in this code use the LRB load-balancing method.

   Additional details can be found:

   * J. Fox, A. Tripathy, O. Green, 
   “Improving Scheduling for Irregular Applications with Logarithmic Radix Binning”, 
   IEEE High Performance Extreme Computing Conference (HPEC), 
   Waltham, Massachusetts, 2019 
   * O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, 
   “Logarithmic Radix Binning and Vectorized Triangle Counting”, 
   IEEE High Performance Extreme Computing Conference (HPEC), 
   Waltham, Massachusetts, 2018
*/

using data_t=float;

#include "lrb_sort.cuh"
int main(int argc, char* argv[]) {

    int  num_segments=20000;       // e.g., 3
    int printSegs=10;
    
    int  *h_offsets=new int[num_segments+1];
    h_offsets[0]=0;
    for (int i=1; i<=num_segments; i++){
        h_offsets[i] = h_offsets[i-1]+i;
    }

    data_t *h_values = new data_t[h_offsets[num_segments]];
    int pos=0;
    for (int i=0; i<num_segments; i++){
        for (int s=0; s<=i; s++){
            if(i<printSegs){
                printf("%d,",(i-s));
            }
            h_values[pos++] = i-s;
        }
        if(i<printSegs)
            printf("\n");
    }
    

    int  *d_offsets;         // e.g., [0, 3, 3, 7]
    cudaMalloc(&d_offsets, (num_segments+1)*sizeof(int32_t));
    cudaMemcpy(d_offsets,h_offsets,(num_segments+1)*sizeof(int32_t), cudaMemcpyHostToDevice );
    //int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]    

    data_t *d_values;
    cudaMalloc(&d_values, (h_offsets[num_segments])*sizeof(data_t));
    cudaMemcpy(d_values,h_values,(h_offsets[num_segments])*sizeof(data_t), cudaMemcpyHostToDevice );

    segmentedSort(num_segments,d_offsets, d_offsets+1, d_values);
    cudaDeviceSynchronize();



    cudaMemcpy(h_values,d_values,(h_offsets[num_segments])*sizeof(data_t), cudaMemcpyDeviceToHost );

    pos=0;
    for (int i=0; i<printSegs; i++){
        for (int s=0; s<=i; s++){
            printf("%d,",int(h_values[pos]));
            pos++;
        }
        printf("\n");
    }



    cudaDeviceSynchronize();

    cudaFree(d_values);
    cudaFree(d_offsets);
    delete [] h_offsets;
    delete [] h_values;
}
