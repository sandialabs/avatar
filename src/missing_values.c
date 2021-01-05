/********************************************************************************** 
Avatar Tools 
Copyright (c) 2019, National Technology and Engineering Solutions of Sandia, LLC
All rights reserved. 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer  in the
  documentation and/or other materials provided with the distribution.


3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For questions, comments or contributions contact 
Philip Kegelmeyer, wpk@sandia.gov 
*******************************************************************************/
#include "crossval.h"
#include "missing_values.h"
#include "array.h"

void calculate_missing_values(float **c_atts, int *num_c_exs, int **d_atts, int *num_d_exs, CV_Subset *data) {
    int i;
    int c_count = 0;
    int d_count = 0;
    
    data->meta.Missing = (union data_point_union *)malloc(data->meta.num_attributes * sizeof(union data_point_union));
    
    for (i = 0; i < data->meta.num_attributes; i++) {
        //printf("Calculate MV for 0-based att %d\n", i);
        if (data->meta.attribute_types[i] == CONTINUOUS) {
            data->meta.Missing[i].Continuous = find_median(c_atts[c_count], num_c_exs[c_count]);
            //printf("This is att %d (continuous att %d): %g\n", i+1, c_count+1, data->meta.Missing[i].Continuous);
            c_count++;
        } else if (data->meta.attribute_types[i] == DISCRETE) {
            //printf("Finding post popular from %d %d %d %d ...\n", d_atts[d_count][0], d_atts[d_count][1], d_atts[d_count][2], d_atts[d_count][3]);
            data->meta.Missing[i].Discrete = find_most_popular(d_atts[d_count], num_d_exs[d_count]);
            //printf("This is att %d (discrete att %d): %d\n", i+1, d_count+1, data->meta.Missing[i].Discrete);
            d_count++;
        }
    }
}

int find_most_popular(int *arr, int n) {
    int i;
    int *temp;
    int this_int, num_this_int;
    int most_popular, num_most_popular;
    
    // Make a copy since we're sorting in place
    temp = (int *)calloc(n, sizeof(int));
    for (i = 0; i < n; i++)
        temp[i] = arr[i];
    
    int_array_sort(n, temp-1);
    
    this_int = temp[0];
    most_popular = temp[0];
    num_this_int = 1;
    num_most_popular = 0;
    for (i = 1; i < n; i++) {
        if (temp[i] == this_int) {
            num_this_int++;
        } else {
            if (num_this_int > num_most_popular) {
                num_most_popular = num_this_int;
                most_popular = this_int;
            }
            this_int = temp[i];
            num_this_int = 1;
        }
    }
    if (num_this_int > num_most_popular)
        most_popular = this_int;
    
    free(temp);
    
    return most_popular;
}

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float find_median(float *arr, int n) {
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n-1;
    median = (low + high) / 2;
    
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }
        
        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
        if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
        if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;
        
        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low+1]) ;
        
        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do ll++; while (arr[low] > arr[ll]) ;
            do hh--; while (arr[hh]  > arr[low]) ;
            
            if (hh < ll)
                break;
            
            ELEM_SWAP(arr[ll], arr[hh]) ;
        }
        
        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]) ;
        
        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

