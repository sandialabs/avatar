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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array.h"

int find_int(int key, int n, int *arr)
{
  for(int i = 0; i < n; i++)
  {
    if(arr[i] == key)
      return i + 1;
  }
  return 0;
}

void int_array_sort(int n, int *ra) {
    int l, j, ir, i;
    int rra;
    
    if (n < 2) return;
    
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1)
            rra = ra[--l];
        else {
            rra = ra[ir];
            ra[ir] = ra[1];
            if (--ir == 1) {
                ra[1] = rra;
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && ra[j] < ra[j+1]) ++j;
            if (rra < ra[j]) {
                ra[i] = ra[j];
                j += (i=j);
            }
            else j = ir + 1;
        }
        ra[i] = rra;
    }
}

void float_array_sort(int n, float *ra) {
    int l, j, ir, i;
    float rra;
    
    if (n < 2) return;
    
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1)
            rra = ra[--l];
        else {
            rra = ra[ir];
            ra[ir] = ra[1];
            if (--ir == 1) {
                ra[1] = rra;
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && ra[j] < ra[j+1]) ++j;
            if (rra < ra[j]) {
                ra[i] = ra[j];
                j += (i=j);
            }
            else j = ir + 1;
        }
        ra[i] = rra;
    }
}

/*
   Computes the union of two arrays
*/

int array_union(int *array_a, int size_a, int *array_b, int size_b, int **Union) {

    int count_a = 0;
    int count_b = 0;
    int count_u = 0;

    // The '+ 1' is to allow for the '-1' as the final element so
    // I can get the size later using length_of_uid if necessary
    *Union = (int *)malloc((size_a + size_b + 1) * sizeof(int));
    
    int i;
    
    // Make sure input arrays are sorted
    int_array_sort(size_a, array_a-1);
    int_array_sort(size_b, array_b-1);
    
    while (count_a < size_a && count_b < size_b) {
        if (array_a[count_a] == array_b[count_b]) {
            (*Union)[count_u++] = array_a[count_a];
            count_a++;
            count_b++;
        } else if (array_a[count_a] < array_b[count_b]) {
            (*Union)[count_u++] = array_a[count_a];
            count_a++;
        } else if (array_b[count_b] < array_a[count_a]) {
            (*Union)[count_u++] = array_b[count_b];
            count_b++;
        }
    }
    
    for (i = count_a; i < size_a; i++) {
        (*Union)[count_u++] = array_a[count_a];
        count_a++;
    }
    for (i = count_b; i < size_b; i++) {
        (*Union)[count_u++] = array_b[count_b];
        count_b++;
    }
    
    (*Union)[count_u] = -1;

    return(count_u);

}

int array_intersection(int *array_a, int size_a, int *array_b, int size_b, int **Intersection) {

    // The '+ 1' is to allow for the '-1' as the final element so
    // I can get the size later using length_of_uid if necessary
    *Intersection = (int *)malloc(((size_a > size_b ? size_b : size_a) + 1) * sizeof(int));

    // Make sure input arrays are sorted
    int_array_sort(size_a, array_a-1);
    int_array_sort(size_b, array_b-1);
        
    int count_a = 0;
    int count_b = 0;
    int count_i = 0;
    while (count_a < size_a && count_b < size_b) {
        if (array_a[count_a] == array_b[count_b]) {
            (*Intersection)[count_i++] = array_a[count_a];
            count_a++;
            count_b++;
        } else if (array_a[count_a] < array_b[count_b]) {
            count_a++;
        } else if (array_b[count_b] < array_a[count_a]) {
            count_b++;
        }
    }
    
    (*Intersection)[count_i] = -1;
    
    return(count_i);

}

int array_diff(int *array_a, int size_a, int *array_b, int size_b, int **inAnotB) {

    // The '+ 1' is to allow for the '-1' as the final element so
    // I can get the size later using length_of_uid if necessary
    *inAnotB = (int *)malloc((size_a + 1) * sizeof(int));

    int i;
    
    // Make sure input arrays are sorted
    int_array_sort(size_a, array_a-1);
    int_array_sort(size_b, array_b-1);
    
    int count_a = 0;
    int count_b = 0;
    int count_AnB = 0;
    while (count_a < size_a && count_b < size_b) {
        if (array_a[count_a] == array_b[count_b]) {
            count_a++;
            count_b++;
        } else if (array_a[count_a] < array_b[count_b]) {
            (*inAnotB)[count_AnB++] = array_a[count_a];
            count_a++;
        } else if (array_b[count_b] < array_a[count_a]) {
            count_b++;
        }
    }
    
    for (i = count_a; i < size_a; i++) {
        (*inAnotB)[count_AnB++] = array_a[count_a];
        count_a++;
    }
    for (i = count_b; i < size_b; i++) {
        count_b++;
    }
    
    (*inAnotB)[count_AnB] = -1;
    
    return(count_AnB);

}

int length_of_uid(int *array) {
    int size = 0;
    while (array[size++] > -1) {}
    return(size - 1);
}

void array_print(int *array, int size, char *title) {
    int i;
    printf("%s\n", title);
    for (i = 0; i < size; i++) {
        printf("  array[%d] = %d\n", i, array[i]);
    }
}

