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
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array.h"
#include "util.h"

/*
   Shuffle then sort a float array with a tracking int array
 */
void shuffle_sort_float_int(int size, float *array1, int *array2, int order) {
    int i, temp;
    int *indx, *new2;
    float *new1;
    indx = (int *)malloc(size * sizeof(int));
    new1 = (float *)malloc(size * sizeof(float));
    new2 = (int *)malloc(size * sizeof(int));
    
    // Make up array for shuffling the array pairs
    for (i = 0 ; i < size; i++)
        indx[i] = i;
    _knuth_shuffle(size, indx);
    // Shuffle each of the two arrays in the same way
    for (i = 0; i < size; i++) {
        new1[i] = array1[indx[i]];
        new2[i] = array2[indx[i]];
    }
    free(indx);
    
    // Sort of the shuffled first array but drag second array along for the ride
    float_int_array_sort(size, new1-1, new2-1);
    
    // Reassign the shuffled/sorted versions of the arrays
    for (i = 0; i < size; i++) {
        array1[i] = new1[i];
        array2[i] = new2[i];
    }
    free(new1);
    free(new2);
    
    // Switch if needed. Default is ASCENDING
    if (order == DESCENDING) {
        for (i = 0; i < size/2; i++) {
            temp = array1[i];
            array1[i] = array1[size-i-1];
            array1[size-i-1] = temp;
            temp = array2[i];
            array2[i] = array2[size-i-1];
            array2[size-i-1] = temp;
        }
    }
    
}

void shuffle_sort_int_int(int size, int *array1, int *array2, int order) {
    int i, temp;
    int *indx, *new1, *new2;
    indx = (int *)malloc(size * sizeof(int));
    new1 = (int *)malloc(size * sizeof(int));
    new2 = (int *)malloc(size * sizeof(int));
    
    // Make up array for shuffling the array pairs
    for (i = 0 ; i < size; i++)
        indx[i] = i;
    _knuth_shuffle(size, indx);
    // Shuffle each of the two arrays in the same way
    for (i = 0; i < size; i++) {
        new1[i] = array1[indx[i]];
        new2[i] = array2[indx[i]];
    }
    free(indx);
    
    // Sort of the shuffled first array but drag second array along for the ride
    int_two_array_sort(size, new1-1, new2-1);
    
    // Reassign the shuffled/sorted versions of the arrays
    for (i = 0; i < size; i++) {
        array1[i] = new1[i];
        array2[i] = new2[i];
    }
    free(new1);
    free(new2);
    
    // Switch if needed. Default is ASCENDING
    if (order == DESCENDING) {
        for (i = 0; i < size/2; i++) {
            temp = array1[i];
            array1[i] = array1[size-i-1];
            array1[size-i-1] = temp;
            temp = array2[i];
            array2[i] = array2[size-i-1];
            array2[size-i-1] = temp;
        }
    }
    
}

static int* origArr = NULL;
static int* sortedArr = NULL;
static int origLen = 0;

//Linear search for key in the array of ints. Replaces grep_int().
//Returns 0 if not found, or 1+index (in sorted order) if found.
int find_int(int key, int n, int *arr)
{
  int i;
  //detect if the array changed, and if so, copy and sort it
  if(arr != origArr || n != origLen)
  {
    origArr = arr;
    origLen = n;
    free(sortedArr);
    sortedArr = malloc(n * sizeof(int));
    memcpy(sortedArr, origArr, n * sizeof(int));
    //int_array_sort uses 1-based indices
    int_array_sort(n, sortedArr - 1);
  }
  int ind = 0;
  for(i = 0; i < n; i++)
  {
    if(sortedArr[i] == key)
    {
      ind = i + 1;
      break;
    }
  }
  return ind;
}

void find_int_release()
{
  free(sortedArr);
  sortedArr = NULL;
  origArr = NULL;
  origLen = 0;
}

int remove_dups_int(int num, int *array) {
    if (num == 0)
        return 0;
    int i, j;
    int flag;
    int_array_sort(num, array-1);
    // Need to flag the duplicates. Use the highest value in the array plus one as the flag
    flag = array[num-1]+1;
    for (i = 1; i < num; i++)
        if (array[i] == array[i-1])
            array[i-1] = flag;
    j = 0;
    for (i = 0; i < num; i++)
        if (array[i] != flag)
            array[j++] = array[i];
    return j;
}

int array_to_range(const int *array, int num, char **range) {
    int i;
    if(num == 0 || !array)
    {
      *range = av_strdup("none");
      return 4;
    }
    char sep[] = {'-', 0};
    //allocate the upper bound of space
    //(array printed in decimal, comma-separated)
    //INT_MIN has 11 digits including sign
    int seq_count = 1;
    char* r = malloc(12 * num * sizeof(char));
    sprintf(r, "%d", array[0]);
    size_t rlen = strlen(r);
    for (i = 1; i < num; i++) {
        seq_count = 1;
        while (i < num && array[i] == array[i-1]+1) {
            i++;
            seq_count++;
        }
        if (seq_count > 1) {
            i--;
            sep[0] = ',';
            if (seq_count > 2)
                sep[0] = '-';
        } else {
            sep[0] = ',';
        }
        char temp[16];
        sprintf(temp, "%s%d", sep, array[i]);
        size_t add = strlen(temp);
        memcpy(r + rlen, temp, add + 1);
        rlen += add;
    }
    *range = av_strdup(r);
    free(r);
    return strlen(*range);
}

void parse_int_range(const char *str, int sort, int *num, int **range) {
    char* copy;
    char *token, *ptr_d, *ptr_c;
    int start, end;
    int i;
    int c = 0;
    int cur_alloc = 10;
    *range = (int *)malloc(cur_alloc * sizeof(int));
    *num = 0;

    copy = av_strdup(str);
    token = copy;
    // Handle each string separated by ',' -- but this skips the last one ...
    while ((ptr_c = strchr(token, ','))) {
        *(ptr_c) = '\0';
        if ((ptr_d = strchr(token, '-'))) {
            *(ptr_d) = '\0';
            start = atoi(token);
            token = ++ptr_d;
            end = atoi(token);
            *num += end - start + 1;
        } else {
            start = atoi(token);
            end = start;
            (*num)++;
        }
        
        if (*num > cur_alloc) {
            cur_alloc = *num + 10;
            *range = (int *)realloc(*range, cur_alloc * sizeof(int));
        }
        for (i = start; i <= end; i++) {
            (*range)[c++] = i;
        }
        token = av_strdup(++ptr_c);
    }

    // ... handle the last one
    if ((ptr_d = strchr(token, '-'))) {
        *(ptr_d) = '\0';
        start = atoi(token);
        token = ++ptr_d;
        end = atoi(token);
        *num += end - start + 1;
    } else {
        start = atoi(token);
        end = start;
        (*num)++;
    }

    if (*num > cur_alloc) {
        cur_alloc = *num + 10;
        *range = (int *)realloc(*range, cur_alloc * sizeof(int));
    }
    for (i = start; i <= end; i++) {
        (*range)[c++] = i;
    }
    
    if (sort)
        int_array_sort(*num, (*range)-1);

    free(copy);
}

void parse_float_range(char *str, int sort, int *num, float **range) {
    
    char *token, *ptr_c;
    int c = 0;
    int cur_alloc = 10;
    *range = (float *)malloc(cur_alloc * sizeof(float));
    *num = 0;

    token = av_strdup(str);
    // Handle each string separated by ',' -- but this skips the last one ...
    while ((ptr_c = strchr(token, ','))) {
        *(ptr_c) = '\0';
        (*num)++;
        
        if (*num > cur_alloc) {
            cur_alloc = *num + 10;
            *range = (float *)realloc(*range, cur_alloc * sizeof(float));
        }
        (*range)[c++] = atof(token);
        
        token = av_strdup(++ptr_c);
    }

    // ... handle the last one
    (*num)++;

    if (*num > cur_alloc) {
        cur_alloc = *num + 10;
        *range = (float *)realloc(*range, cur_alloc * sizeof(float));
    }
    (*range)[c++] = atof(token);

    if (sort)
        float_array_sort(*num, (*range)-1);
    
    free(token);
}

/*
 * Not currently used
 *
void _parse_comma_sep_string(char *str, int *num, char ***tokens) {
    
    int i;
    char *token, *ptr_c;
    int c = 0;
    int cur_alloc = 10;
    *tokens = (char **)malloc(cur_alloc * sizeof(char *));
    *num = 0;
    
    token = av_strdup(str);
    // Handle each string separated by ',' -- but this skips the last one ...
    while ((ptr_c = strchr(token, ','))) {
        *ptr_c = '\0';
        (*num)++;
        
        if (*num > cur_alloc) {
            cur_alloc = *num + 10;
            *tokens = (char **)realloc(*tokens, cur_alloc * sizeof(char *));
        }
        (*tokens)[c++] = av_strdup(token);
        token = ++ptr_c;
    }
    
    // ... handle the last one
    (*num)++;
    
    if (*num > cur_alloc) {
        cur_alloc = *num + 10;
        *tokens = (char **)realloc(*tokens, cur_alloc * sizeof(char *));
    }
    (*tokens)[c++] = av_strdup(token);
    
    for (i = 0; i < *num; i++)
        strip_lt_whitespace((*tokens)[i]);
}
 *
 */

void parse_delimited_string(char delimiter, char *str, int *num, char ***tokens) {
    //count the number of tokens there will be
    int i,n = 1;
    size_t len = 0;
    char * it;
    for(it = str; *it; it++)
    {
      if(*it == delimiter)
        n++;
      len++;
    }
    *num = n;
    *tokens = malloc(n * sizeof(char*));
    //copy out the tokens, stripping whitespace
    char* temp = malloc(len + 1);
    char* tokStart = str;
    for(i = 0; i < n; i++)
    {
      size_t tokLen = 0;
      while(tokStart[tokLen] && tokStart[tokLen] != delimiter)
      {
        tokLen++;
      }
      memcpy(temp, tokStart, tokLen);
      temp[tokLen] = 0;
      strip_lt_whitespace(temp);
      (*tokens)[i] = av_strdup(temp);
      tokStart += (tokLen + 1);
    }
    free(temp);
}

// Strip leading/trailing whitespace
void strip_lt_whitespace(char* s)
{
    assert(s != NULL);

    // Find first non-whitespace character.
    char* first = s;
    while (*first != '\0' && isspace(*first)) {
        ++first;
    }

    // If s is nothing but whitespace, return empty string.
    if (*first == '\0') {
        *s = '\0';
        return;
    }

    // Find last non-whitespace character.
    size_t length = strlen(first);
    char* last = first + length - 1;
    while (last >= first && isspace(*last)) {
        --last;
    }
    assert(last >= first);  // empty string case should have already been handled above
    // Make last point to where null-termination should be.
    ++last;
    *last = '\0';
    length = last - first;

    // Shift the non-space characters so that s points to where they begin.
    if (s != first) {
        memmove(s, first, length+1); // +1 for null termination
    }
}



/*
 * ToDo: trailing white space is not removed from last token
 */
void parse_space_sep_string(char *str, int *num, char ***tokens) {
    
    int j, k;
    
    int wait_for_dquote = 0;
    int wait_for_squote = 0;
    int last_space = -1;
    int this_space = -1;
    
    // Initialize
    *num = 0;
    *tokens = (char **)malloc(sizeof(char *));
    
    // Start at beginning and end at end ...
    int j_1 = 0;
    int j_2 = strlen(str);
    // ... unless the whole thing is wrapped in double quotes, in which case, ignore them
    if (str[0] == '"' && str[j_2-1] == '"') {
        j_1++;
        j_2--;
    }
    
    // Build up space-separated arguments allowing arguments to be single- or double-quote delimited
    for (j = j_1; j < j_2; j++) {
        if ( str[j] == '"' && wait_for_dquote == 0 )
            wait_for_dquote = 1;
        else if ( str[j] == '"' && wait_for_dquote == 1 )
            wait_for_dquote = 0;
        else if ( str[j] == 39 && wait_for_squote == 0 )    // 39 = '
            wait_for_squote = 1;
        else if ( str[j] == 39 && wait_for_squote == 1 )
            wait_for_squote = 0;
        else if ( ( str[j] == ' ' || str[j] == '\t' || j == j_2 - 1 ) &&
                    wait_for_dquote == 0 && wait_for_squote == 0 ) {
            this_space = j;
            // Skip over multiple spaces in a row
            if (this_space == last_space + 1) {
                last_space = this_space;
                continue;
            }
            // Adjust if we're at the end of the string ...
            if (j == j_2 - 1)
                this_space++;

            (*num)++;
            *tokens = (char **)realloc(*tokens, (*num) * sizeof(char *));

            // Malloc room for the characters between the spaces and copy them one by one
            int str_size = this_space - last_space - 1;
            if (str_size < 1)
                fprintf(stderr, "This arg is only %d characters wide!\n", str_size);
            (*tokens)[(*num)-1] = (char *)malloc((str_size + 1) * sizeof(char));
            for (k = 0; k < str_size; k++)
                (*tokens)[(*num) - 1][k] = str[last_space + 1 + k];
            // Manually NUL terminate
            (*tokens)[(*num) - 1][str_size] = '\0';
            last_space = this_space;
        }
    }
}

/*
 * Not currently used
 *
// Index Table from Numerical Recipes in C
// Uses 1-based arrays.
// So, cheating by creating copies that are 1-based from the 0-based arrays passed int/out
void float_index_table(int n, float *arrin, int **indx, int order) {
    int l, j, ir, indxt, i;
    float q;
    // 1-based copies
    int *indx_1;
    float *arrin_1;
    
    arrin_1 = (float *)calloc(n + 1, sizeof(float)); // 1-based
    indx_1 = (int *)calloc(n + 1, sizeof(int));      // 1-based
    (*indx) = (int *)malloc(n * sizeof(int));        // 0-based
    
    for (j = 1; j <= n; j++) {
        indx_1[j] = j;           // Initialize
        arrin_1[j] = arrin[j-1]; // Create 1-based copy
    }
    if (n == 1) {
        // Map back to 0-based, free(), and return
        for (i = 0; i < n; i++) {
            if (order == DESCENDING)
                (*indx)[i] = indx_1[n-i]-1;
            else
                (*indx)[i] = indx_1[i+1]-1;
        }
        free(indx_1);
        free(arrin_1);
        return;
    }
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1)
            q = arrin_1[(indxt=indx_1[--l])];
        else {
            q = arrin_1[(indxt=indx_1[ir])];
            indx_1[ir] = indx_1[1];
            if (--ir == 1) {
                indx_1[1] = indxt;
                // Map back to 0-based, free(), and return
                for (i = 0; i < n; i++) {
                    if (order == DESCENDING)
                        (*indx)[i] = indx_1[n-i]-1;
                    else
                        (*indx)[i] = indx_1[i+1]-1;
                }
                free(indx_1);
                free(arrin_1);
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && arrin_1[indx_1[j]] < arrin_1[indx_1[j+1]])
                j++;
            if (q < arrin_1[indx_1[j]]) {
                indx_1[i] = indx_1[j];
                j += (i=j);
            }
            else
                j = ir + 1;
        }
        indx_1[i] = indxt;
    }
}
 *
 */

/*
 * Not currently used
 *
void int_index_table(int n, int *arrin, int **indx, int order) {
    int l, j, ir, indxt, i;
    int q;
    // 1-based copies
    int *indx_1;
    int *arrin_1;
    
    arrin_1 = (int *)calloc(n + 1, sizeof(int)); // 1-based
    indx_1 = (int *)calloc(n + 1, sizeof(int));      // 1-based
    (*indx) = (int *)malloc(n * sizeof(int));        // 0-based
    
    for (j = 1; j <= n; j++) {
        indx_1[j] = j;           // Initialize
        arrin_1[j] = arrin[j-1]; // Create 1-based copy
    }
    if (n == 1) {
        // Map back to 0-based, free(), and return
        for (i = 0; i < n; i++) {
            if (order == DESCENDING)
                (*indx)[i] = indx_1[n-i]-1;
            else
                (*indx)[i] = indx_1[i+1]-1;
        }
        free(indx_1);
        free(arrin_1);
        return;
    }
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1)
            q = arrin_1[(indxt=indx_1[--l])];
        else {
            q = arrin_1[(indxt=indx_1[ir])];
            indx_1[ir] = indx_1[1];
            if (--ir == 1) {
                indx_1[1] = indxt;
                // Map back to 0-based, free(), and return
                for (i = 0; i < n; i++) {
                    if (order == DESCENDING)
                        (*indx)[i] = indx_1[n-i]-1;
                    else
                        (*indx)[i] = indx_1[i+1]-1;
                }
                free(indx_1);
                free(arrin_1);
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && arrin_1[indx_1[j]] < arrin_1[indx_1[j+1]])
                j++;
            if (q < arrin_1[indx_1[j]]) {
                indx_1[i] = indx_1[j];
                j += (i=j);
            }
            else
                j = ir + 1;
        }
        indx_1[i] = indxt;
    }
}
 *
 */

// Heapsort: Numerical Recipes in C
// This and float_array_sort use 1-based arrays.
// Pass in array-1 if int *array is 0-based
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

// Heapsort: Numerical Recipes in C
// This and float_array_sort use 1-based arrays.
// Pass in array-1 if int *array is 0-based
//
// Modified to sort on ra[] but modifiy rb[] similarly
void int_two_array_sort(int n, int *ra, int *rb) {
    int l, j, ir, i;
    int rra, rrb;
    
    if (n < 2) return;
    
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1) {
            rra = ra[--l];
            rrb = rb[l];
        } else {
            rra = ra[ir];
            rrb = rb[ir];
            ra[ir] = ra[1];
            rb[ir] = rb[1];
            if (--ir == 1) {
                ra[1] = rra;
                rb[1] = rrb;
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && ra[j] < ra[j+1]) ++j;
            if (rra < ra[j]) {
                ra[i] = ra[j];
                rb[i] = rb[j];
                j += (i=j);
            }
            else j = ir + 1;
        }
        ra[i] = rra;
        rb[i] = rrb;
    }
}

// Heapsort: Numerical Recipes in C
// This and float_array_sort use 1-based arrays.
// Pass in array-1 if int *array is 0-based
//
// Modified to sort on ra[] (float) but modifiy rb[] (int) similarly
void float_int_array_sort(int n, float *ra, int *rb) {
    int l, j, ir, i;
    float rra;
    int rrb;
    
    if (n < 2) return;
    
    l = (n >> 1) + 1;
    ir = n;
    for (;;) {
        if (l > 1) {
            rra = ra[--l];
            rrb = rb[l];
        } else {
            rra = ra[ir];
            rrb = rb[ir];
            ra[ir] = ra[1];
            rb[ir] = rb[1];
            if (--ir == 1) {
                ra[1] = rra;
                rb[1] = rrb;
                return;
            }
        }
        i = l;
        j = l << 1;
        while (j <= ir) {
            if (j < ir && ra[j] < ra[j+1]) ++j;
            if (rra < ra[j]) {
                ra[i] = ra[j];
                rb[i] = rb[j];
                j += (i=j);
            }
            else j = ir + 1;
        }
        ra[i] = rra;
        rb[i] = rrb;
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

/*
 * Not currently used
 *
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
 *
 */

/*
 * Not currently used
 *
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
 *
 */

/*
 * Not currently used
 *
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
 *
 */

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

int int_find_max(int *array, int size, int *max) {
    int i;
    *max = array[0];
    int max_i = 0;
    for (i = 1; i < size; i++) {
        if (array[i] > *max) {
            *max = array[i];
            max_i = i;
        }
    }
    return max_i;
}

float int_average(int *array, int start, int end) {
    int sum = 0;
    int i;
    for (i = start; i <= end; i++)
        sum += array[i];
    return ((float)sum/(float)(end-start+1));
}

/*
 * Not currently used
 *
float float_average(float *array, int start, int end) {
    float sum = 0.0;
    int i;
    for (i = start; i <= end; i++)
        sum += array[i];
    return (sum/(float)(end-start+1));
}
 *
 */

float int_stddev(int *array, int start, int end) {
    float sum = 0.0;
    int i;
    float avg = int_average(array, start, end);
    for (i = start; i <= end; i++)
        sum += pow((float)array[i] - avg, 2.0);
    return sqrt(sum/(float)(end-start));
}

/*
 * Not currently used
 *
float float_stddev(float *array, int start, int end) {
    float sum = 0.0;
    int i;
    float avg = float_average(array, start, end);
    for (i = start; i <= end; i++)
        sum += pow(array[i] - avg, 2.0);
    return sqrt(sum/(float)(end-start));
}
 *
 */

