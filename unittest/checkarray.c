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
/**
 * \file checkarrayutils.c
 * \brief Unit tests for array module.
 *
 * $Source: /home/Repositories/avatar/avatar/src-redesign/unittest/checkarray.c,v $
 * $Revision: 1.11 $ 
 * $Date: 2007/09/01 05:52:15 $
 *
 * \modifications
 *    
 */

#include <stdlib.h>
#include <string.h>
#include <check.h>
#include <time.h>
#include <unistd.h>
#include "../src/array.h"
#include "checkall.h"

// *********************************************
// ***** General library interface tests
// *********************************************

START_TEST(sorting)
{
    int A[] = { 1, 5, 4, 6, 7, 10, 34, 100, 254653, 0, -4, -2354, -5 };
    int A_truth[] = { -2354, -5, -4, 0, 1, 4, 5, 6, 7, 10, 34, 100, 254653 };
    float B[] = { 1.2, 5.45, 4.023, 6.345, 7.31, 10.3, 34.0, 100, 254653, 0, -4.99, -2354.2, -4.991 };
    float B_truth[] = { -2354.2, -4.991, -4.99, 0, 1.2, 4.023, 5.45, 6.345, 7.31, 10.3, 34.0, 100, 254653 };
    
    // *_array_sort uses one-offset arrays. So, passing in A-1 "converts" our zero-offset arrays "on the fly"
    int_array_sort(13, A-1);
    float_array_sort(13, B-1);
    
    fail_unless( memcmp(A, A_truth, 13 * sizeof(int)) == 0, "int sort failed" );
    fail_unless( memcmp(B, B_truth, 13 * sizeof(float)) == 0, "float sort failed" );
}
END_TEST
/*
START_TEST(uid)
{
    
    int A[] = { 1, 2, 3, 5, 8, 13, 21, 34, 22 };
    int B[] = { 1, 3, 5, 7, 9, 11, 13, 15 };

    int *Union;
    int *Intersect;
    int *InAnotB;
    int *InBnotA;

    int U_truth1[] = { 1, 2, 3, 5, 7, 8, 9, 11, 13, 15, 21, 22, 34, -1 };
    int I_truth1[] = { 1, 3, 5, 13, -1 };
    int InA_truth[] = { 2, 8, 21, 22, 34, -1 };
    int InB_truth[] = { 7, 9, 11, 15, -1 };
    
    int size_u = array_union(A, 9, B, 8, &Union);
    int size_i = array_intersection(A, 9, B, 8, &Intersect);
    int size_da = array_diff(A, 9, B, 8, &InAnotB);
    int size_db = array_diff(B, 8, A, 9, &InBnotA);
    fail_unless( size_u == 13, "returned size not correct for union" );
    fail_unless( size_i == 4, "returned size not correct for intersection" );
    fail_unless( size_da == 5, "returned size not correct for in1" );
    fail_unless( size_db == 4, "returned size not correct for in2" );
    fail_unless( length_of_uid(Union) == 13, "length_of_uid not correct for union" );
    fail_unless( length_of_uid(Intersect) == 4, "length_of_uid not correct for intersection" );
    fail_unless( length_of_uid(InAnotB) == 5, "length_of_uid not correct for in1" );
    fail_unless( length_of_uid(InBnotA) == 4, "length_of_uid not correct for in2" );
    fail_unless( memcmp(Union, U_truth1, 14 * sizeof(int)) == 0, "union not correct" );
    fail_unless( memcmp(Intersect, I_truth1, 5 * sizeof(int)) == 0, "intersection not correct" );
    fail_unless( memcmp(InAnotB, InA_truth, 6 * sizeof(int)) == 0, "inAnotB not correct" );
    fail_unless( memcmp(InBnotA, InB_truth, 5 * sizeof(int)) == 0, "inBnotA not correct" );
    
    free(Union);
    free(Intersect);
    free(InAnotB);
    free(InBnotA);

}
END_TEST
*/
START_TEST(uid_length)
{
    //int A[] = { };
    //int B[] = { 0 };
    int C[] = { 0, -1 };
    int D[] = { -1 };
    fail_unless( length_of_uid(C) == 1, "length incorrect for array of length 1");
    fail_unless( length_of_uid(D) == 0, "length incorrect for array of length 0");
}
END_TEST

START_TEST(int_searching)
{
    int A[] = { 1, 2, 33, 15, 48, 123, 221, 34, 22, 100 };
    fail_unless( find_int(33, 9, A) == 5, "failed to grep existing value from int array (1)\n");
    fail_unless( find_int(221, 9, A) == 9, "failed to grep existing value from int array (2)\n");
    fail_unless( find_int(100, 10, A) == 8, "failed to grep existing value from int array (3)\n");
    fail_unless( find_int(-1, 10, A) == 0, "found non-existing value in int array (4)\n");
    fail_unless( find_int(-1, 10, A) == 0, "found non-existing value in int array (5)\n");
    find_int_release();
}
END_TEST

START_TEST(find_max)
{
    int A[] = { 1, 2, 33, 15, 48, 123, 221, 34, 22 };
    int high_pop;
    fail_unless( int_find_max(A, 9, &high_pop) == 6 && high_pop == 221, "failed to find int max\n");
    A[0] = 222;
    fail_unless( int_find_max(A, 9, &high_pop) == 0 && high_pop == 222, "failed to find int max at front\n");
    A[8] = 223;
    fail_unless( int_find_max(A, 9, &high_pop) == 8 && high_pop == 223, "failed to find int max at end\n");
}
END_TEST
/*
START_TEST(index_table)
{
    float A[] = { 14.01, 8.4, 32.8, 7.06, 3.1, 15.3 };
    int B[] = { 14, 8, 32, 7, 3, 15 };
    int truth_up[] = { 4, 3, 1, 0, 5, 2 };
    int truth_down[] = { 2, 5, 0, 1, 3, 4 };
    int *testA, *testB;
    
    // Check ascending order
    float_index_table(6, A, &testA, 0);
    int_index_table(6, B, &testB, 0);
    fail_unless(memcmp(truth_up, testA, 6 * sizeof(int)) == 0, "float index_table failed for ascending");
    fail_unless(memcmp(truth_up, testB, 6 * sizeof(int)) == 0, "int index_table failed for ascending");
    free(testA);
    free(testB);
    
    // Check descending order
    float_index_table(6, A, &testA, 1);
    int_index_table(6, B, &testB, 1);
    fail_unless(memcmp(truth_down, testA, 6 * sizeof(int)) == 0, "float index_table failed for descending");
    fail_unless(memcmp(truth_down, testB, 6 * sizeof(int)) == 0, "int index_table failed for descending");
    free(testA);
    free(testB);
    
    // Check for an array of one
    
    // Check ascending order
    float_index_table(1, A, &testA, 0);
    int_index_table(1, B, &testB, 0);
    fail_unless(*testA == 0, "float index_table failed for ascending size-1 array");
    fail_unless(*testB == 0, "int index_table failed for ascending size-1 array");
    free(testA);
    free(testB);
    
    // Check descending order
    float_index_table(1, A, &testA, 1);
    int_index_table(1, B, &testB, 1);
    fail_unless(*testA == 0, "float index_table failed for descending size-1 array");
    fail_unless(*testB == 0, "int index_table failed for descending size-1 array");
    free(testA);
    free(testB);
    
}
END_TEST
*/
START_TEST(array2range)
{
    int array1[] = { 2 };
    int array2[] = { 2, 3 };
    int array3[] = { 2, 4 };
    int array4[] = { 2, 3, 4, 5 };
    int array5[] = { 2, 3, 4, 5, 7, 8, 9 };
    int array6[] = { 2, 4, 5, 6, 8 };
    char *truth1 = "2";
    char *truth2 = "2,3";
    char *truth3 = "2,4";
    char *truth4 = "2-5";
    char *truth5 = "2-5,7-9";
    char *truth6 = "2,4-6,8";
    char *range;
    
    array_to_range(array1, 1, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth1), "failed (1)");
    free(range);
    array_to_range(array2, 2, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth2), "failed (2)");
    free(range);
    array_to_range(array3, 2, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth3), "failed (3)");
    free(range);
    array_to_range(array4, 4, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth4), "failed (4)");
    free(range);
    array_to_range(array5, 7, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth5), "failed (5)");
    free(range);
    array_to_range(array6, 5, &range);
    //printf("%s\n", range);
    fail_unless(! strcmp(range, truth6), "failed (6)");
    free(range);
}
END_TEST

START_TEST(parse_spaces)
{
    int i;
    char *string1 = "foo : bar one   two three     four  \" five six   seven\" eight  ";
    char **elements;
    int num_elements;
    parse_space_sep_string(string1, &num_elements, &elements);
    
    fail_unless(num_elements == 9, "(1) did not get 9 elements in list");
    fail_unless(! strcmp(elements[0], "foo"), "(1) did not get 'foo'");
    fail_unless(! strcmp(elements[1], ":"), "(1) did not get ':'");
    fail_unless(! strcmp(elements[2], "bar"), "(1) did not get 'bar'");
    fail_unless(! strcmp(elements[3], "one"), "(1) did not get 'one'");
    fail_unless(! strcmp(elements[4], "two"), "(1) did not get 'two'");
    fail_unless(! strcmp(elements[5], "three"), "(1) did not get 'three'");
    fail_unless(! strcmp(elements[6], "four"), "(1) did not get 'four'");
    fail_unless(! strcmp(elements[7], "\" five six   seven\""), "(1) did not get '\" five six   seven\"'");
    fail_unless(! strcmp(elements[8], "eight"), "(1) did not get 'eight'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
    
    char *string2 = "foo : bar one   two three     four  \" five six   seven\" eight";
    parse_space_sep_string(string2, &num_elements, &elements);
    
    fail_unless(num_elements == 9, "(2) did not get 9 elements in list");
    fail_unless(! strcmp(elements[0], "foo"), "(2) did not get 'foo'");
    fail_unless(! strcmp(elements[1], ":"), "(2) did not get ':'");
    fail_unless(! strcmp(elements[2], "bar"), "(2) did not get 'bar'");
    fail_unless(! strcmp(elements[3], "one"), "(2) did not get 'one'");
    fail_unless(! strcmp(elements[4], "two"), "(2) did not get 'two'");
    fail_unless(! strcmp(elements[5], "three"), "(2) did not get 'three'");
    fail_unless(! strcmp(elements[6], "four"), "(2) did not get 'four'");
    fail_unless(! strcmp(elements[7], "\" five six   seven\""), "(2) did not get '\" five six   seven\"'");
    fail_unless(! strcmp(elements[8], "eight"), "(2) did not get 'eight'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
}
END_TEST

START_TEST(parse_commas)
{
    int i;
    char *string1 = "foo,:,bar,one,two,three,four,five six   seven,eight";
    char **elements;
    int num_elements;
    //parse_comma_sep_string(string1, &num_elements, &elements);
    parse_delimited_string(',', string1, &num_elements, &elements);
    
    fail_unless(num_elements == 9, "(1) did not get 9 elements in list");
    fail_unless(! strcmp(elements[0], "foo"), "(1) did not get 'foo'");
    fail_unless(! strcmp(elements[1], ":"), "(1) did not get ':'");
    fail_unless(! strcmp(elements[2], "bar"), "(1) did not get 'bar'");
    fail_unless(! strcmp(elements[3], "one"), "(1) did not get 'one'");
    fail_unless(! strcmp(elements[4], "two"), "(1) did not get 'two'");
    fail_unless(! strcmp(elements[5], "three"), "(1) did not get 'three'");
    fail_unless(! strcmp(elements[6], "four"), "(1) did not get 'four'");
    fail_unless(! strcmp(elements[7], "five six   seven"), "(1) did not get 'five six   seven'");
    fail_unless(! strcmp(elements[8], "eight"), "(1) did not get 'eight'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
    
    char *string2 = "foo, :, bar  ,one, two  ,    three,four,\"  five six   seven\" , eight  ";
    //parse_comma_sep_string(string2, &num_elements, &elements);
    parse_delimited_string(',', string2, &num_elements, &elements);
    
    fail_unless(num_elements == 9, "(2) did not get 9 elements in list");
    fail_unless(! strcmp(elements[0], "foo"), "(2) did not get 'foo'");
    fail_unless(! strcmp(elements[1], ":"), "(2) did not get ':'");
    fail_unless(! strcmp(elements[2], "bar"), "(2) did not get 'bar'");
    fail_unless(! strcmp(elements[3], "one"), "(2) did not get 'one'");
    fail_unless(! strcmp(elements[4], "two"), "(2) did not get 'two'");
    fail_unless(! strcmp(elements[5], "three"), "(2) did not get 'three'");
    fail_unless(! strcmp(elements[6], "four"), "(2) did not get 'four'");
    fail_unless(! strcmp(elements[7], "\"  five six   seven\""), "(2) did not get '\"  five six   seven\"'");
    fail_unless(! strcmp(elements[8], "eight"), "(2) did not get 'eight'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
}
END_TEST

START_TEST(parse_colons)
{
    int i;
    char *string1 = ":one two, three    four  ";
    char **elements;
    int num_elements;
    //parse_comma_sep_string(string1, &num_elements, &elements);
    parse_delimited_string(':', string1, &num_elements, &elements);
    
    fail_unless(num_elements == 2, "(1) did not get 2 elements in list");
    fail_unless(strlen(elements[0]) == 0, "(1) did not get ''");
    fail_unless(! strcmp(elements[1], "one two, three    four"), "(1) did not get 'one two, three    four'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
    
    char *string2 = "       :          one two, three    four  ";
    //parse_comma_sep_string(string2, &num_elements, &elements);
    parse_delimited_string(':', string2, &num_elements, &elements);
    
    fail_unless(num_elements == 2, "(2) did not get 2 elements in list");
    fail_unless(strlen(elements[0]) == 0, "(2) did not get ''");
    fail_unless(! strcmp(elements[1], "one two, three    four"), "(2) did not get 'one two, three    four'");
    
    for (i = 0; i < num_elements; i++)
        free(elements[i]);
    free(elements);
}
END_TEST


START_TEST(test_shuffle_sort)
{
    int size = 16;
    int votes[]     = {  1, 4, 4, 3, 4,  2, 3, 1,  1,  4,  3,  2,  2,  2,  1,  5 };
    int classes_v[] = {  0, 1, 2, 3, 4,  5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15 };

    float weights[] = {  1, 4, 4, 3, 4,  2, 3, 1,  1,  4,  3,  2,  2,  2,  1,  5 };
    int classes_w[] = {  0, 1, 2, 3, 4,  5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15 };

    int truth_v[]   = {  5, 4, 4, 4, 4,  3, 3, 3,  2,  2,  2,  2,  1,  1,  1,  1 };
    float truth_w[] = {  5, 4, 4, 4, 4,  3, 3, 3,  2,  2,  2,  2,  1,  1,  1,  1 };
    int truth_c[]   = { 15, 9, 1, 4, 2, 10, 6, 3, 12, 11, 13,  5, 14,  7,  0,  8 };
    
    srand48(123);
    shuffle_sort_int_int(size, votes, classes_v, DESCENDING);
    fail_unless(memcmp(votes, truth_v, size * sizeof(int)) == 0, "Votes array didn't match");
    fail_unless(memcmp(classes_v, truth_c, size * sizeof(int)) == 0, "Classes(1) array didn't match");
    
    srand48(123);
    shuffle_sort_float_int(size, weights, classes_w, DESCENDING);
    fail_unless(memcmp(weights, truth_w, size * sizeof(float)) == 0, "Weights array didn't match");
    fail_unless(memcmp(classes_w, truth_c, size * sizeof(int)) == 0, "Classes(2) array didn't match");
}
END_TEST

START_TEST(test_dup_removal)
{
    int dup_size = 16;
    int truth_size = 8;
    int size;
    int dups[] = { -1, 0, 5, 7, 9, 11, 13, 17, -1, 0, 5, 7, 9, 11, 13, 17 };
    int truth[] = { -1, 0, 5, 7, 9, 11, 13, 17 };
    size = remove_dups_int(dup_size, dups);
    fail_unless(size == truth_size, "Should have gotten 8 unique values");
    fail_unless(memcmp(dups, truth, truth_size * sizeof(float)) == 0, "All dups were not removed");
}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *array_suite(void)
{
    Suite *suite = suite_create("ArrayUtils");
    
    TCase *tc_array = tcase_create(" - ArrayUtils Interface ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_array);
    tcase_add_test(tc_array, sorting);
    //tcase_add_test(tc_array, uid);
    tcase_add_test(tc_array, uid_length);
    tcase_add_test(tc_array, int_searching);
    tcase_add_test(tc_array, find_max);
    //tcase_add_test(tc_array, index_table);
    tcase_add_test(tc_array, array2range);
    tcase_add_test(tc_array, parse_spaces);
    tcase_add_test(tc_array, parse_commas);
    tcase_add_test(tc_array, parse_colons);
    tcase_add_test(tc_array, test_shuffle_sort);
    tcase_add_test(tc_array, test_dup_removal);
    
    return suite;
}
