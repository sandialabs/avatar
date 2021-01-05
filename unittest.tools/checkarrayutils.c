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
 * $Source: /home/Repositories/avatar/avatar/unittest/checkarrayutils.c,v $
 * $Revision: 1.4 $ 
 * $Date: 2006/02/04 04:59:32 $
 *
 * \modifications
 *    
 */

#include <stdlib.h>
#include <string.h>
#include <check.h>
#include "fc.h"
#include "array.h"
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

START_TEST(int_searching))
{
    int A[] = { 1, 2, 33, 15, 48, 123, 221, 34, 22 };
    fail_unless( find_int(33, 9, A), "failed to grep existing value from int array\n");
    fail_unless( find_int(221, 9, A), "failed to grep existing value from int array\n");
    fail_unless( ! find_int(-1, 9, A), "found non-existing value in int array\n");
    fail_unless( ! find_int(-1, 9, A), "found non-existing value in int array\n");
}
END_TEST
  
// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *arrayutils_suite(void)
{
    Suite *suite = suite_create("ArrayUtils");
    
    TCase *tc_arrayutils = tcase_create(" - ArrayUtils Interface ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_arrayutils);
    tcase_add_test(tc_arrayutils, sorting);
    tcase_add_test(tc_arrayutils, uid);
    tcase_add_test(tc_arrayutils, uid_length);
    tcase_add_test(tc_arrayutils, grep);
    
    return suite;
}
