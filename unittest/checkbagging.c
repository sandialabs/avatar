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
 * \file checkbagging.c
 * \brief Unit tests for bagging.
 *
 * $Source: /home/Repositories/avatar/avatar/src-redesign/unittest/checkbagging.c,v $
 * $Revision: 1.7 $ 
 * $Date: 2007/06/30 06:09:55 $
 *
 * \modifications
 *    
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "../src/crossval.h"
#include "check.h"
#include "checkall.h"
#include "util.h"
#include "../src/bagging.h"
#include "../src/util.h"

void _gen_bag_data(int num_examples, CV_Subset *data, Args_Opts *args);
void _free_bag_data(CV_Subset data);

void _gen_bag_data(int num_examples, CV_Subset *data, Args_Opts *args) {
    int i;
    
    args->debug = FALSE;
    args->use_opendt_shuffle = FALSE;
    args->majority_bagging = FALSE;
    args->random_seed = 1;
    args->mpi_rank = 0;
    data->meta.num_classes = 0;
    data->meta.num_attributes = 0;
    data->meta.num_examples = num_examples;
    data->examples = (CV_Example *)malloc(num_examples * sizeof(CV_Example));
    data->meta.exo_data.num_seq_meshes = 0;
    for (i = 0; i < num_examples; i++)
        data->examples[i].global_id_num = i;
}

void _free_bag_data(CV_Subset data) {
    free(data.examples);
}

// *********************************************
// ***** General library interface tests
// *********************************************

START_TEST(bagging100)
{
    int i;
    int num = 1000000;
    Args_Opts Args = {0};
    CV_Subset Data = {0}, Bag = {0};
    long **tuple_counts;
    int num_it = 3;
    
    double fudge_factor[] = { 1.005, 1.005, 1.005, 1.015, 1.03 };
    
    tuple_counts = (long **)malloc(num_it * sizeof(long*));
    
    _gen_bag_data(num, &Data, &Args);
    Args.bag_size = 100.0;
    
    // Check that we have one of every number in the original data
    _count_xlicates(Data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    for (i = 0; i < num_it; i++) {
        
      make_bag(&Data, &Bag, Args, 0);
        _count_xlicates(Bag, 0, &tuple_counts[i]);
        
        //__print_xlicates(tuple_counts[i]);
        //__print_scaled_xlicates(tuple_counts[i], num);
        _free_bag_data(Bag);
    }
    _free_bag_data(Data);
    
    // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
    //printf("Checking for %d out of %d\n", num, num);
    fail_unless(! _check_xlicates(tuple_counts, num_it, num, num, fudge_factor), "distribution incorrect for 100% bagging");
    
    for (i = 0; i < num_it; i++)
        free(tuple_counts[i]);
    free(tuple_counts);
}
END_TEST

START_TEST(bagging71)
{
    int i;
    int num = 1000000;
    Args_Opts Args = {0};
    CV_Subset Data = {0}, Bag = {0};
    long **tuple_counts;
    int num_it = 3;
    
    double fudge_factor[] = { 1.005, 1.005, 1.01, 1.015, 1.03 };
    
    tuple_counts = (long **)malloc(num_it * sizeof(long*));
    
    _gen_bag_data(num, &Data, &Args);
    Args.bag_size = 71;
    
    // Check that we have one of every number in the original data
    _count_xlicates(Data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    for (i = 0; i < num_it; i++) {
        // Explicity set seed for first two only
        if (i < 2)
            srand48(i+Args.bag_size);
        
        make_bag(&Data, &Bag, Args, 0);
        _count_xlicates(Bag, num, &tuple_counts[i]);
        
        //__print_xlicates(tuple_counts[i]);
        //__print_scaled_xlicates(tuple_counts[i], num);
        _free_bag_data(Bag);
    }
    _free_bag_data(Data);
    
    // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
    //printf("Checking for %d out of %d\n", (int)(Args.bag_size*num/100), num);
    fail_unless(! _check_xlicates(tuple_counts, num_it, Args.bag_size*num/100, num, fudge_factor),
                "distribution incorrect for 71% bagging");
    
    for (i = 0; i < num_it; i++)
        free(tuple_counts[i]);
    free(tuple_counts);
}
END_TEST

START_TEST(bagging20)
{
    int i;
    int num = 1000000;
    Args_Opts Args = {0};
    CV_Subset Data = {0}, Bag = {0};
    long **tuple_counts;
    int num_it = 3;
    
    double fudge_factor[] = { 1.005, 1.01, 1.08, 1.2, 1.5 };
    
    tuple_counts = (long **)malloc(num_it * sizeof(long*));
    
    _gen_bag_data(num, &Data, &Args);
    Args.bag_size = 20;
    
    // Check that we have one of every number in the original data
    _count_xlicates(Data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    for (i = 0; i < num_it; i++) {
        // Explicity set seed for first two only
        if (i < 2)
            srand48(i+Args.bag_size);
        
        make_bag(&Data, &Bag, Args, 0);
        _count_xlicates(Bag, num, &tuple_counts[i]);
        
        //__print_xlicates(tuple_counts[i]);
        //__print_scaled_xlicates(tuple_counts[i], num);
        _free_bag_data(Bag);
    }
    _free_bag_data(Data);
    
    // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
    //printf("Checking for %d out of %d\n", (int)(Args.bag_size*num/100), num);
    fail_unless(! _check_xlicates(tuple_counts, num_it, Args.bag_size*num/100, num, fudge_factor),
                "distribution incorrect for 5% bagging");
    
    for (i = 0; i < num_it; i++)
        free(tuple_counts[i]);
    free(tuple_counts);
}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *bagging_suite(void)
{
    Suite *suite = suite_create("Bagging");
    
    TCase *tc_bagging = tcase_create(" - Bagging ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_bagging);
    tcase_add_test(tc_bagging, bagging100);
    tcase_add_test(tc_bagging, bagging71);
    tcase_add_test(tc_bagging, bagging20);
    
    return suite;
}
