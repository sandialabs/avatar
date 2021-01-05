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
#include <string.h>
#include "../src/crossval.h"
#include "check.h"
#include "checkall.h"
#include "util.h"
#include "../src/balanced_learning.h"
#include "../src/skew.h"
#include "../src/util.h"
#include "../src/memory.h"
#include "../src/array.h"

void _gen_clump_data(int num_examples, float min_prop, CV_Subset *data, Args_Opts *args, AV_SortedBlobArray *blob);
void _free_clump_data(CV_Subset *data);
void _map_clump_classes(CV_Subset data, CV_Subset clump, int class, int **class_map, long **tuple_counts);

void _gen_clump_data(int num_examples, float min_prop, CV_Subset *data, Args_Opts *args, AV_SortedBlobArray *blob) {
    int init_proportions[] = { 0, 2, 29, 59, 46, 35, 29 };
    int *num_per_class;
    int i;
    
    args->debug = FALSE;
    args->use_opendt_shuffle = FALSE;
    args->majority_bagging = FALSE;
    args->majority_ivoting = FALSE;
    args->do_balanced_learning = TRUE;
    args->do_ivote = FALSE;
    args->do_bagging = FALSE;
    args->num_minority_classes = 1;
    args->minority_classes = (int *)malloc(args->num_minority_classes * sizeof(int));
    args->minority_classes[0] = 1;
    args->proportions = (float *)malloc(args->num_minority_classes * sizeof(float));
    //args->proportions[0] = 0.12;
    args->proportions[0] = min_prop;
    
    data->meta.num_classes = 7;
    data->meta.num_attributes = 0;
    data->meta.num_examples = num_examples;
    data->meta.num_examples_per_class = (int *)calloc(data->meta.num_classes, sizeof(int));
    data->examples = (CV_Example *)calloc(num_examples, sizeof(CV_Example));
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

void _free_clump_data(CV_Subset *data) {
    free(data->examples);
}

START_TEST(check_comp_num_parts_2exact)
{
    int i;
    int k, s;
    
    CV_Metadata meta = {0};
    Args_Opts args = {0};
    
    int nepc[] = {10000,100};
    float props[] = {0.5};
    int minc[] = {1};

    meta.num_classes = 2;
    meta.num_examples = 0;
    for (i = 0; i < meta.num_classes; i++)
        meta.num_examples += nepc[i];
    args.num_minority_classes = 1;
    meta.num_examples_per_class = nepc;
    args.majority_ivoting = FALSE;
    args.majority_bagging = FALSE;
    args.do_balanced_learning = FALSE;
    args.proportions = props;
    args.minority_classes = minc;
    
    compute_number_of_clumps(meta, &args, &k, &s);
    
    fail_unless(k == 100, "Number of partitions not correct");
    fail_unless(s == 200, "Number of samples/partition not correct");
}
END_TEST

START_TEST(check_comp_num_parts_2inexact)
{
    int i;
    int k, s;
    
    CV_Metadata meta = {0};
    Args_Opts args = {0};
    
    int nepc[] = {10001,100};
    float props[] = {0.5};
    int minc[] = {1};

    meta.num_classes = 2;
    meta.num_examples = 0;
    for (i = 0; i < meta.num_classes; i++)
        meta.num_examples += nepc[i];
    args.num_minority_classes = 1;
    meta.num_examples_per_class = nepc;
    args.majority_ivoting = FALSE;
    args.majority_bagging = FALSE;
    args.do_balanced_learning = FALSE;
    args.proportions = props;
    args.minority_classes = minc;
    
    compute_number_of_clumps(meta, &args, &k, &s);
    
    fail_unless(k == 101, "Number of partitions not correct");
    fail_unless(s == 200, "Number of samples/partition not correct");
}
END_TEST

START_TEST(check_comp_num_parts_10)
{
    int i;
    int k, s;
    
    CV_Metadata meta = {0};
    Args_Opts args = {0};
    
    int nepc[] = {120,100,1000,5000,110,100,4950,10300,7490,6080};
    float props[] = {0.1,0.1,0.1,0.1};
    int minc[] = {0,1,4,5};

    meta.num_classes = 10;
    meta.num_examples = 0;
    for (i = 0; i < meta.num_classes; i++)
        meta.num_examples += nepc[i];
    args.num_minority_classes = 4;
    meta.num_examples_per_class = nepc;
    args.majority_ivoting = FALSE;
    args.majority_bagging = FALSE;
    args.do_balanced_learning = FALSE;
    args.proportions = props;
    args.minority_classes = minc;
    
    compute_number_of_clumps(meta, &args, &k, &s);
    
    fail_unless(k == 54, "Number of partitions not correct");
    fail_unless(s == 1077, "Number of samples/partition not correct");
}
END_TEST

START_TEST(check_sample_selection_exact)
{
    int i, j;
    int num = 200; // The exact number of samples as in _gen_clump_data so each sample is picked exactly once
    CV_Subset data = {0}, clump = {0};
    long **tuple_counts;
    int **class_map;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob = {0};
    int num_clumps, num_ex_per_clump;
    _gen_clump_data(num, 0.01, &data, &args, &sorted_blob);
    
    assign_bl_clump_numbers(data.meta, sorted_blob, args);
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    tuple_counts = (long **)malloc(sizeof(long*));
    
    // Check that we have one of every number in the original data
    _count_xlicates(data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    _map_classes(data, &class_map);
    for (i = 0; i < num_clumps; i++) {
        get_next_balanced_set(i, &data, &clump, args);
        // Check xlicates for each class
        for (j = 0; j < clump.meta.num_classes; j++) {
            if (data.meta.num_examples_per_class[j] > 0) {
                // Create mapping from examples in each class to ordered list to check for xlicates
                _count_class_xlicates(data, clump, j, class_map, &tuple_counts[0]);
                // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
                char *error_msg;
                error_msg = (char *)malloc(128*sizeof(char));
                sprintf(error_msg, "distribution incorrect for class %d in clump %d\n", j, i);
                fail_unless(! _check_no_xlicates(tuple_counts, 1, data.meta.num_examples_per_class[j]), error_msg);
                free(error_msg);
                free(tuple_counts[0]);
            } else {
                //printf("Skipping class %d because no samples\n", j);
            }
        }
        free_CV_Subset_inter(&clump, args, TRAIN_MODE);
    }
    free(tuple_counts);
}
END_TEST

START_TEST(check_sample_selection)
{
    int i, j;
    int num = 200;
    CV_Subset data = {0}, clump = {0};
    long **tuple_counts;
    int **class_map;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_clump_data(num, 0.12, &data, &args, &sorted_blob);
    
    assign_bl_clump_numbers(data.meta, sorted_blob, args);
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    tuple_counts = (long **)malloc(sizeof(long*));
    
    // Check that we have one of every number in the original data
    _count_xlicates(data, 0, &tuple_counts[0]);
    //__print_xlicates(tuple_counts);
    fail_unless(tuple_counts[0][0] == 0 && tuple_counts[0][1] == num, "missing data");
    free(tuple_counts[0]);
    
    _map_classes(data, &class_map);
    for (i = 0; i < num_clumps; i++) {
        get_next_balanced_set(i, &data, &clump, args);
        // Check xlicates for each class
        for (j = 0; j < clump.meta.num_classes; j++) {
            if (data.meta.num_examples_per_class[j] > 0) {
                // Create mapping from examples in each class to ordered list to check for xlicates
                _count_class_xlicates(data, clump, j, class_map, &tuple_counts[0]);
                // Make sure misses, singles, dup-/trip-/quadrip-licates are in range
                char *error_msg;
                error_msg = (char *)malloc(128*sizeof(char));
                sprintf(error_msg, "distribution incorrect for class %d in clump %d\n", j, i);
                fail_unless(! _check_for_0_and_1(tuple_counts, 1, clump.meta.num_examples_per_class[j],
                                                 data.meta.num_examples_per_class[j]), error_msg);
                free(error_msg);
                free(tuple_counts[0]);
            } else {
                //printf("Skipping class %d because no samples\n", j);
            }
        }
        free_CV_Subset_inter(&clump, args, TRAIN_MODE);
    }
    free(tuple_counts);
}
END_TEST

START_TEST(check_sample_overlap)
{
    int i, j;
    int num = 200;
    CV_Subset data = {0}, clump = {0};
    int *sample_counts;
    int *sample_classes;
    int *dups_per_class;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_clump_data(num, 0.12, &data, &args, &sorted_blob);
    
    assign_bl_clump_numbers(data.meta, sorted_blob, args);
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    sample_counts = (int *)calloc(data.meta.num_examples, sizeof(int));
    sample_classes = (int *)calloc(data.meta.num_examples, sizeof(int));
    dups_per_class = (int *)calloc(data.meta.num_classes, sizeof(int));
    for (i = 0; i < num_clumps; i++) {
        get_next_balanced_set(i, &data, &clump, args);
        for (j = 0; j < clump.meta.num_examples; j++) {
            sample_counts[clump.examples[j].global_id_num]++;
            sample_classes[clump.examples[j].global_id_num] = clump.examples[j].containing_class_num;
        }
    }
    for (i = 0; i < data.meta.num_examples; i++) {
        if (sample_classes[i] == args.minority_classes[0]) {
            fail_unless(sample_counts[i] == num_clumps, "wrong number for minority class");
        } else {
            if (sample_counts[i] == 2)
                dups_per_class[sample_classes[i]]++;
            else
                fail_unless(sample_counts[i] == 1, "wrong number for majority class");
        }
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        if (i == args.minority_classes[0])
            continue;
        char *error_msg;
        error_msg = (char *)malloc(128 * sizeof(char));
        sprintf(error_msg, "wrong number for class %d\n", i);
        fail_unless(dups_per_class[i] == (_num_per_class_per_clump(i, num_clumps, data.meta, args) * num_clumps) -
                                         data.meta.num_examples_per_class[i], error_msg);
        free(error_msg);
    }
}
END_TEST

START_TEST(check_clump_overlap)
{
    int i, j;
    int num = 200;
    CV_Subset data = {0}, clump = {0};
    int *sample_counts;
    int *sample_clumps;
    int num_dups = 0;
    Args_Opts args = {0};
    AV_SortedBlobArray sorted_blob;
    int num_clumps, num_ex_per_clump;
    _gen_clump_data(num, 0.12, &data, &args, &sorted_blob);
    
    assign_bl_clump_numbers(data.meta, sorted_blob, args);
    compute_number_of_clumps(data.meta, &args, &num_clumps, &num_ex_per_clump);
    //printf("Using %d clumps and %d examples per\n", num_clumps, num_ex_per_clump);
    sample_counts = (int *)calloc(data.meta.num_examples, sizeof(int));
    sample_clumps = (int *)calloc(data.meta.num_examples, sizeof(int));
    for (i = 0; i < num_clumps; i++) {
        get_next_balanced_set(i, &data, &clump, args);
        for (j = 0; j < num_ex_per_clump; j++) {
            sample_counts[clump.examples[j].global_id_num]++;
            if (i > 0)
                fail_unless(i != sample_clumps[clump.examples[j].global_id_num], "dup in one clump");
            sample_clumps[clump.examples[j].global_id_num] = i;
        }
    }
    for (i = 0; i < data.meta.num_examples; i++) {
        //printf("Sample %d seen %d times\n", i, sample_counts[i]);
        if (data.examples[i].containing_class_num == args.minority_classes[0]) {
            fail_unless(sample_counts[i] == num_clumps, "minority class not seen in all clumps");
        } else if (sample_counts[i] == 2) {
            //printf("Dup for sample %d found in %d\n", i, sample_clumps[i]);
            //fail_unless(sample_clumps[i] == 0, "dups found in other than first clump");
            num_dups++;
        } else if (sample_counts[i] != 1) {
            fail_unless(sample_counts[i] == 1, "sample missed or used more than twice");
        }
    }
    int num1 = num_clumps * num_ex_per_clump;
    int num2 = data.meta.num_examples + num_dups;
    // Add minority classes replicated in clumps
    for (i = 0; i < num_clumps; i++)
      if (find_int(i, args.num_minority_classes, args.minority_classes))
            num2 += (num_clumps - 1) * data.meta.num_examples_per_class[i];
    fail_unless(num1 == num2, "wrong number of dups");
}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *blearning_suite(void)
{
    Suite *suite = suite_create("BalancedLearning");
    
    TCase *tc_blearning = tcase_create(" - Balanced Learning ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_blearning);
    tcase_add_test(tc_blearning, check_comp_num_parts_2exact);
    tcase_add_test(tc_blearning, check_comp_num_parts_2inexact);
    tcase_add_test(tc_blearning, check_comp_num_parts_10);
    tcase_add_test(tc_blearning, check_sample_selection_exact);
    tcase_add_test(tc_blearning, check_sample_selection);
    // Make sure each sample occurs once throughout all clumps and that the proper number of examples
    // occur twice (to account for wrapping around to the beginning when the numbers don't come out even)
    tcase_add_test(tc_blearning, check_sample_overlap);
    // Make sure each sample occurs once in each clump
    tcase_add_test(tc_blearning, check_clump_overlap);
    return suite;
}
