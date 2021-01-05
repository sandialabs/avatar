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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/missing_values.h"


START_TEST(check_median)
{
    float values1[] = { 1 };
    fail_unless(find_median(values1, 1) == 1, "median (1) should be 1");
    
    float values2[] = { 1, 2 };
    fail_unless(find_median(values2, 2) == 1, "median (2) should be 1");
    
    // Always picks the lower of the two middle ones if there are an even number of values
    float values3[] = { 2, 1 };
    fail_unless(find_median(values3, 2) == 1, "median (3) should be 1");
    
    float values4[] = { 11, 7, 3, 4, 10, 6, 2, 9, 8, 5, 1 };
    fail_unless(find_median(values4, 11) == 6, "median (4) should be 6");
    
    // Always picks the lower of the two middle ones if there are an even number of values
    float values5[] = { 11, 7, 3, 4, 10, 6, 12, 2, 9, 8, 5, 1 };
    fail_unless(find_median(values5, 12) == 6, "median (5) should be 6");
}
END_TEST

START_TEST(check_most_popular)
{
    int values1[] = { 1 };
    fail_unless(find_most_popular(values1, 1) == 1, "most popular (1) should be 1");
    
    int values2[] = { 1, 2 };
    fail_unless(find_most_popular(values2, 2) == 1, "most popular (2) should be 1");
    
    int values3[] = { 2, 1 };
    fail_unless(find_most_popular(values3, 2) == 1, "most popular (3) should be 1");
    
    int values4[] = { 2, 1, 1, 1, 2 };
    fail_unless(find_most_popular(values4, 5) == 1, "most popular (4) should be 1");
    
    int values5[] = { 2, 1, 1, 1, 2, 2, 2 };
    fail_unless(find_most_popular(values5, 7) == 2, "most popular (5) should be 2");
    
    int values6[] = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1 };
    fail_unless(find_most_popular(values6, 13) == 1, "most popular(6) should be 1");
}
END_TEST

Suite *missing_suite(void)
{
    Suite *suite = suite_create("MissingValues");
    
    TCase *tc_missing_values = tcase_create(" Check MissingValues ");
    
    suite_add_tcase(suite, tc_missing_values);
    tcase_add_test(tc_missing_values, check_median);
    tcase_add_test(tc_missing_values, check_most_popular);
        
    return suite;
}
