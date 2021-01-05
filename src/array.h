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
#ifndef __ARRAY__
#define __ARRAY__

typedef enum {
    ASCENDING,
    DESCENDING
} Sort_Order;

void shuffle_sort_float_int(int size, float *array1, int *array2, int order);
void shuffle_sort_int_int(int size, int *array1, int *array2, int order);
void int_array_sort(int n, int *array);
void int_two_array_sort(int n, int *ra, int *rb);
//void float_index_table(int n, float *arrin, int **indx, int order);
void float_int_array_sort(int n, float *ra, int *rb);
//void int_index_table(int n, int *arrin, int **indx, int order);
void float_array_sort(int n, float *array);
//int array_union(int *array_a, int size_a, int *array_b, int size_b, int **Union);
//int array_intersection(int *array_a, int size_a, int *array_b, int size_b, int **Intersection);
//int array_diff(int *array_a, int size_a, int *array_b, int size_b, int **InANotB);
int length_of_uid(int *array);
void array_print(int *array, int size, char *title);
int find_int(int value, int size, int *array);
void find_int_release();
int remove_dups_int(int num, int *array);
void parse_int_range(const char *str, int sort, int *num, int **range);
void parse_delimited_string(char delimiter, char *str, int *num, char ***tokens);
void _parse_comma_sep_string(char *str, int *num, char ***tokens);
void parse_space_sep_string(char *str, int *num, char ***tokens);
void parse_float_range(char *str, int sort, int *num, float **range);
int int_find_max(int *array, int size, int *max);
int array_to_range(const int *array, int num, char **range);

/*
 * Delete leading and trailing whitespace from str.
 *
 * Pre: str != NULL
 * Post: Any whitespace that was prefixed or suffixed on str is removed.
 *       The pointer str points to beginning of remaining content.
 *
 * NB: this function modifies the given string in place.
 */
void strip_lt_whitespace(char* str);

float int_average(int *array, int start, int end);
float float_average(float *array, int start, int end);
float int_stddev(int *array, int start, int end);
float float_stddev(float *array, int start, int end);

#endif //  __ARRAY__
