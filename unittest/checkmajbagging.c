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
#include <time.h>
#include <unistd.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/bagging.h"
#include "../src/util.h"
#include "../src/skew.h"
#include "../src/memory.h"
#include "../src/array.h"
#include "util.h"

void _gen_majbag_data(int num_examples, float min_prop, CV_Subset *data, Args_Opts *args, AV_SortedBlobArray *blob);

void _gen_majbag_data(int num_examples, float min_prop, CV_Subset *data, Args_Opts *args, AV_SortedBlobArray *blob) {
    int init_proportions[] = { 0, 2, 29, 59, 46, 35, 29 };
    int *num_per_class;
    int i;
    
    args->debug = FALSE;
    args->use_opendt_shuffle = FALSE;
    args->majority_bagging = TRUE;
    args->do_balanced_learning = FALSE;
    args->num_minority_classes = 1;
    args->minority_classes = (int *)malloc(args->num_minority_classes * sizeof(int));
    args->minority_classes[0] = 1;
    args->proportions = (float *)malloc(args->num_minority_classes * sizeof(float));
    args->proportions[0] = min_prop;
    args->random_seed = 3;
    args->mpi_rank = 0;
    
    data->meta.num_classes = 7;
    data->meta.num_attributes = 0;
    data->meta.num_examples = num_examples;
    data->meta.num_examples_per_class = (int *)calloc(data->meta.num_classes, sizeof(int));
    data->examples = (CV_Example *)malloc(num_examples * sizeof(CV_Example));
    data->meta.exo_data.num_seq_meshes = 0;
    
    num_per_class = (int *)malloc(data->meta.num_classes * sizeof(int));
    int init_num_examples = 0;
    int new_num_examples = 0;
    for (i = 0; i < data->meta.num_classes; i++)
        init_num_examples += init_proportions[i];
    for (i = 0; i < data->meta.num_classes; i++) {
        num_per_class[i] = (int)((float)init_proportions[i] * (float)num_examples / (float)init_num_examples);
        new_num_examples += num_per_class[i];
    }
    fail_unless(new_num_examples >= init_num_examples, "Initial number of examples too small");
    
    int current_class = 0;
    int current_class_count = 0;
    av_exitIfError(av_initSortedBlobArray(blob));
    for (i = 0; i < num_examples; i++) {
        data->examples[i].global_id_num = i;
        data->examples[i].fclib_id_num = i;
        data->examples[i].fclib_seq_num = 0;
        if (current_class_count < num_per_class[current_class]) {
            data->examples[i].containing_class_num = current_class;
            data->meta.num_examples_per_class[current_class]++;
            current_class_count++;
        } else {
            current_class++;
            data->examples[i].containing_class_num = current_class;
            data->meta.num_examples_per_class[current_class]++;
            current_class_count = 1;
        }
        av_addBlobToSortedBlobArray(blob, &data->examples[i], cv_example_compare_by_seq_id);
    }
    
}

START_TEST(check_sample_selection)
{
    int i, j;
    int num = 1500000;
    CV_Subset data = {0}, bag = {0};
    long **tuple_counts;
    int **class_map;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_majbag_data(num, 0.04, &data, &args, &sorted_blob);
    
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    // Just doing a single iteration here
    tuple_counts = (long **)malloc(sizeof(long*));
    
    // Check that we have one of every number in the original data
    _count_xlicates(data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts[0]);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    _map_classes(data, &class_map);
    for (i = 0; i < 1; i++) {
      make_bag(&data, &bag, args, 0);
        // Check xlicates for each class
        for (j = 0; j < bag.meta.num_classes; j++) {
            if (data.meta.num_examples_per_class[j] > 0) {
                //printf("Class %d has %d examples (%f%%) in data\n", j, data.meta.num_examples_per_class[j], 100.0*(float)data.meta.num_examples_per_class[j]/(float)data.meta.num_examples);
                //printf("Class %d has %d examples (%f%%) in bag\n", j, bag.meta.num_examples_per_class[j], 100.0*(float)bag.meta.num_examples_per_class[j]/(float)bag.meta.num_examples);
                // Create mapping from examples in each class to ordered list to check for xlicates
                _count_class_xlicates(data, bag, j, class_map, &tuple_counts[0]);
                //__print_xlicates(tuple_counts[0]);
                // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
                char *error_msg;
                error_msg = (char *)malloc(128*sizeof(char));
                sprintf(error_msg, "distribution incorrect for class %d in iteration %d", j, i);
                //printf("Class %d has %d samples and bag is picked from %d\n", j, data.meta.num_examples_per_class[j], _num_per_class_per_clump(j, num_clumps, data.meta, args));
                if (find_int(j, args.num_minority_classes, args.minority_classes))
                    fail_unless(! _check_for_0_and_1(tuple_counts, 1,
                                                     _num_per_class_per_clump(j, num_clumps, data.meta, args),
                                                     data.meta.num_examples_per_class[j]), error_msg);
                else
                    fail_unless(! _check_class_xlicates(tuple_counts, 1,
                                                        _num_per_class_per_clump(j, num_clumps, data.meta, args),
                                                        data.meta.num_examples_per_class[j]), error_msg);
                free(error_msg);
                free(tuple_counts[0]);
            } else {
                //printf("Skipping class %d because no samples\n", j);
            }
        }
        free_CV_Subset_inter(&bag, args, TRAIN_MODE);
    }
    free(tuple_counts);
}
END_TEST

START_TEST(check_clump_stats_01)
{
    int i, j;
    int num = 200000;
    CV_Subset data = {0}, bag = {0};
    int *iteration_seen;
    int *integral_seen;
    int iteration_total_seen;
    int integral_total_seen = 0;
    int last_integral_total_seen;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_majbag_data(num, 0.01, &data, &args, &sorted_blob);
    
    iteration_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
    integral_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
    
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    
    srand48(time(NULL)+getpid());
    for (i = 0; i < 10; i++) {
        free(iteration_seen);
        iteration_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
        last_integral_total_seen = integral_total_seen;
        make_bag(&data, &bag, args, 0);
        // Accumulate stats for samples seen
        for (j = 0; j < num_ex_per_clump; j++) {
            int id = bag.examples[j].global_id_num;
            iteration_seen[id] = 1;
            integral_seen[id] = 1;
        }
        // Compute percentage of points seen
        iteration_total_seen = 0;
        integral_total_seen = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            if (iteration_seen[j] == 1)
                iteration_total_seen++;
            if (integral_seen[j] == 1)
                integral_total_seen++;
        }
        //printf("Iteration and Integral percentages after %d iterations: %.4f/%.4f\n", i,
        //       (float)iteration_total_seen/(float)data.meta.num_examples,
        //       (float)integral_total_seen/(float)data.meta.num_examples);
        
        /*
           NOTE: target values for iteration sample selection were computed using the formula
        
                 1.0-(N-1/N)**N
        
                 which gives the percentage of samples picked from a population of N
                 when picking N values with replacement. Actual numbers are approximations
                 because each sample in the minority class is ALWAYS picked one time only.
        */
        float iter = (float)iteration_total_seen/(float)data.meta.num_examples;
        fail_unless(iter > 0.63 && iter < 0.64, "iteration selection failed");
        fail_unless(integral_total_seen > last_integral_total_seen, "sample selection not increasing");
        free_CV_Subset_inter(&bag, args, TRAIN_MODE);
    }
    fail_unless(integral_total_seen > 0.999, "Ending total sample selection too low");
}
END_TEST

START_TEST(check_clump_stats_12)
{
    int i, j;
    int num = 200000;
    CV_Subset data = {0}, bag = {0};
    int *iteration_seen;
    int *integral_seen;
    int iteration_total_seen;
    int integral_total_seen = 0;
    int last_integral_total_seen;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_majbag_data(num, 0.12, &data, &args, &sorted_blob);
    
    iteration_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
    integral_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
    
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    
    srand48(time(NULL)+getpid());
    for (i = 0; i < 100; i++) {
        free(iteration_seen);
        iteration_seen = (int *)calloc(data.meta.num_examples, sizeof(int));
        last_integral_total_seen = integral_total_seen;
        make_bag(&data, &bag, args, 0);
        // Accumulate stats for samples seen
        for (j = 0; j < num_ex_per_clump; j++) {
            int id = bag.examples[j].global_id_num;
            iteration_seen[id] = 1;
            integral_seen[id] = 1;
        }
        // Compute percentage of points seen
        iteration_total_seen = 0;
        integral_total_seen = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            if (iteration_seen[j] == 1)
                iteration_total_seen++;
            if (integral_seen[j] == 1)
                integral_total_seen++;
        }
        //printf("Iteration and Integral percentages after %d iterations: %.4f/%.4f\n", i,
        //       (float)iteration_total_seen/(float)data.meta.num_examples,
        //       (float)integral_total_seen/(float)data.meta.num_examples);
        
        /*
           NOTE: target values for iteration sample selection were computed using the formula
        
                 1.0-(N-1/N)**M
        
                 which gives the percentage of samples picked from a population of N
                 when picking M values with replacement. Actual numbers are approximations
                 because each sample in the minority class is ALWAYS picked one time only.
        */
        float iter = (float)iteration_total_seen/(float)data.meta.num_examples;
        fail_unless(iter > 0.0775 && iter < 0.079, "iteration selection failed");
        fail_unless(integral_total_seen > last_integral_total_seen, "sample selection not increasing");
        free_CV_Subset_inter(&bag, args, TRAIN_MODE);
    }
    fail_unless(integral_total_seen > 0.998, "Ending total sample selection too low");
}
END_TEST

Suite *majbag_suite(void)
{
    Suite *suite = suite_create("MajorityBagging");
    
    TCase *tc_majority_bagging = tcase_create(" Check Majority Bagging ");
    
    suite_add_tcase(suite, tc_majority_bagging);
    tcase_add_test(tc_majority_bagging, check_sample_selection);
    tcase_add_test(tc_majority_bagging, check_clump_stats_01);
    tcase_add_test(tc_majority_bagging, check_clump_stats_12);
        
    return suite;
}
