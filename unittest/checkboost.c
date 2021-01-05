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
#include <math.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/boost.h"

void _boost_tree_set_up(DT_Node **nodes, CV_Subset *data);
void _boost_tree_clean_up(DT_Node **nodes, CV_Subset *data);
void _gen_boost_data(int num_examples, CV_Subset *data, Args_Opts *args);
void _free_boost_data(CV_Subset data);

void _boost_tree_set_up(DT_Node **nodes, CV_Subset *data) {
    int i;
    
    *nodes = (DT_Node *)calloc(9, sizeof(DT_Node));

    (*nodes)[0].branch_type = BRANCH;
    (*nodes)[0].attribute = 0;
    (*nodes)[0].num_branches = 2;
    (*nodes)[0].branch_threshold = 10.0;
    (*nodes)[0].attribute_type = CONTINUOUS;
    (*nodes)[0].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    (*nodes)[0].Node_Value.branch[0] = 1;
    (*nodes)[0].Node_Value.branch[1] = 6;
    
    (*nodes)[1].branch_type = BRANCH;
    (*nodes)[1].attribute = 1;
    (*nodes)[1].num_branches = 2;
    (*nodes)[1].branch_threshold = 8.0;
    (*nodes)[1].attribute_type = CONTINUOUS;
    (*nodes)[1].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    (*nodes)[1].Node_Value.branch[0] = 2;
    (*nodes)[1].Node_Value.branch[1] = 5;
    
    (*nodes)[2].branch_type = BRANCH;
    (*nodes)[2].attribute = 2;
    (*nodes)[2].num_branches = 2;
    (*nodes)[2].branch_threshold = 6.0;
    (*nodes)[2].attribute_type = CONTINUOUS;
    (*nodes)[2].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    (*nodes)[2].Node_Value.branch[0] = 3;
    (*nodes)[2].Node_Value.branch[1] = 4;
    
    (*nodes)[3].branch_type = LEAF;
    (*nodes)[3].Node_Value.class_label = 0;
    
    (*nodes)[4].branch_type = LEAF;
    (*nodes)[4].Node_Value.class_label = 1;
    
    (*nodes)[5].branch_type = LEAF;
    (*nodes)[5].Node_Value.class_label = 2;

    (*nodes)[6].branch_type = BRANCH;
    (*nodes)[6].attribute = 2;
    (*nodes)[6].num_branches = 2;
    (*nodes)[6].branch_threshold = 15.0;
    (*nodes)[6].attribute_type = CONTINUOUS;
    (*nodes)[6].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    (*nodes)[6].Node_Value.branch[0] = 7;
    (*nodes)[6].Node_Value.branch[1] = 8;

    (*nodes)[7].branch_type = LEAF;
    (*nodes)[7].Node_Value.class_label = 1;
    
    (*nodes)[8].branch_type = LEAF;
    (*nodes)[8].Node_Value.class_label = 0;

    data->meta.num_classes = 3;
    data->meta.num_attributes = 3;
    data->meta.num_examples = 7;
    data->meta.class_names = (char **)malloc(data->meta.num_classes * sizeof(char *));
    for (i = 0; i < data->meta.num_classes; i++) {
        data->meta.class_names[i] = (char *)malloc(10 * sizeof(char));
        sprintf(data->meta.class_names[i], "Class%02d", i);
    }

    data->float_data = (float **)malloc(data->meta.num_attributes * sizeof(float *));
    data->float_data[0] = (float *)malloc(data->meta.num_examples * sizeof(float));
    data->float_data[1] = (float *)malloc(data->meta.num_examples * sizeof(float));
    data->float_data[2] = (float *)malloc(data->meta.num_examples * sizeof(float));
    data->float_data[0][0] = data->float_data[1][0] = data->float_data[2][0] =  4.0;
    data->float_data[0][1] = data->float_data[1][1] = data->float_data[2][1] =  6.0;
    data->float_data[0][2] = data->float_data[1][2] = data->float_data[2][2] =  7.0;
    data->float_data[0][3] = data->float_data[1][3] = data->float_data[2][3] =  8.0;
    data->float_data[0][4] = data->float_data[1][4] = data->float_data[2][4] = 11.0;
    data->float_data[0][5] = data->float_data[1][5] = data->float_data[2][5] = 15.0;
    data->float_data[0][6] = data->float_data[1][6] = data->float_data[2][6] = 16.0;
    
    data->examples = (CV_Example *)calloc(data->meta.num_examples, sizeof(CV_Example));
    for (i = 0; i < data->meta.num_examples; i++) {
        data->examples[i].distinct_attribute_values = (int *)malloc(data->meta.num_attributes * sizeof(int));
        data->examples[i].distinct_attribute_values[0] = i;
        data->examples[i].distinct_attribute_values[1] = i;
        data->examples[i].distinct_attribute_values[2] = i;
    }
    data->examples[0].containing_class_num = 0;
    data->examples[1].containing_class_num = 0;
    data->examples[2].containing_class_num = 1;
    data->examples[3].containing_class_num = 2;
    data->examples[4].containing_class_num = 2;
    data->examples[5].containing_class_num = 1;
    data->examples[6].containing_class_num = 0;
}

void _boost_tree_clean_up(DT_Node **nodes, CV_Subset *data) {
    int i;
    for (i = 0; i < 9; i++)
        if ((*nodes)[i].branch_type == BRANCH)
            free((*nodes)[i].Node_Value.branch);
    free(*nodes);
    
    for (i = 0; i < 3; i++)
        free(data->float_data[i]);
    free(data->float_data);
    for (i = 0; i < data->meta.num_classes; i++)
        free(data->meta.class_names[i]);
    free(data->meta.class_names);
    for (i = 0; i < data->meta.num_examples; i++)
        free(data->examples[i].distinct_attribute_values);
    free(data->examples);
}

void _gen_boost_data(int num_examples, CV_Subset *data, Args_Opts *args) {
    int i;
    
    args->debug = FALSE;
    args->majority_ivoting = FALSE;
    data->meta.num_attributes = 0;
    data->meta.num_classes = 0;
    data->meta.num_examples = num_examples;
    data->examples = (CV_Example *)malloc(num_examples * sizeof(CV_Example));
    data->meta.exo_data.num_seq_meshes = 0;
    for (i = 0; i < num_examples; i++) {
        data->examples[i].global_id_num = i;
        data->examples[i].containing_class_num = 0;
        data->examples[i].in_bag = FALSE;
    }
}

void _free_boost_data(CV_Subset data) {
    free(data.examples);
}

START_TEST(check_weighted_samples)
{
    int i;
    double weights[] = { 0.1, 0.2, 0.3, 0.4 };
    Args_Opts Args = {0};
    Args.random_seed = 1;
    int *result;
    int new_freq[] = { 0, 0, 0, 0 };
    
    init_weighted_rng(4, weights, Args);
    
    int num = 1000000;
    result = (int *)malloc(num * sizeof(int));
    for (i = 0; i < num; i++) {
        result[i] = get_next_weighted_sample();
        new_freq[result[i]]++;
    }
    
    free_weighted_rng();
    
    for (i = 0; i < 4; i++) {
        // printf("%d is off by %f%%\n", i, fabsf((float)new_freq[i]/(float)num - weights[i]) * 100.0 / weights[i]);
        fail_unless((float)new_freq[i]/(float)num < 1.004 * weights[i] &&
                    (float)new_freq[i]/(float)num > 0.996 * weights[i],
                    "Sampled proportions are incorrect");
    }
}
END_TEST

START_TEST(check_weighted_samples_with_reset)
{
    int i;
    double weights[] = { 0.044, 0.1, 0.101, 0.755 };
    Args_Opts Args = {0};
    Args.random_seed = 3;
    int *result;
    int *new_freq;
    new_freq = (int *)calloc(4, sizeof(int));
    
    init_weighted_rng(4, weights, Args);
    
    int num = 10000000;
    result = (int *)malloc(num * sizeof(int));
    for (i = 0; i < num; i++) {
        result[i] = get_next_weighted_sample();
        new_freq[result[i]]++;
    }
    
    for (i = 0; i < 4; i++) {
        // printf("%d is off by %f%%\n", i, fabsf((float)new_freq[i]/(float)num - weights[i]) * 100.0 / weights[i]);
        // Adjusted from 1.0004 and 0.9996
        fail_unless((float)new_freq[i]/(float)num < 1.0007 * weights[i] &&
                    (float)new_freq[i]/(float)num > 0.9900 * weights[i],
                    "Sampled proportions are incorrect");
    }
    
    // Set up new weights
    weights[0] = 0.755;
    weights[1] = 0.101;
    weights[2] = 0.1;
    weights[3] = 0.044;
    
    free(new_freq);
    new_freq = (int *)calloc(4, sizeof(int));
    reset_weights(4, weights);
    for (i = 0; i < num; i++) {
        result[i] = get_next_weighted_sample();
        new_freq[result[i]]++;
    }
    
    for (i = 0; i < 4; i++) {
        //printf("%d is off by %f%%\n", i, fabsf((float)new_freq[i]/(float)num - weights[i]) * 100.0 / weights[i]);
        fail_unless((float)new_freq[i]/(float)num < 1.0015 * weights[i] &&
                    (float)new_freq[i]/(float)num > 0.9985 * weights[i],
                    "Sampled proportions are incorrect");
    }
    
    free_weighted_rng();
}
END_TEST

START_TEST(check_update_weights)
{
    int i;
    double *weights;
    double *updated_weights_truth;
    DT_Node *Tree = NULL;
    CV_Subset Data = {0};
    
    _boost_tree_set_up(&Tree, &Data);
    weights = (double *)malloc(Data.meta.num_examples * sizeof(double));
    updated_weights_truth = (double *)malloc(Data.meta.num_examples * sizeof(double));
    for (i = 0; i < Data.meta.num_examples; i++) {
        weights[i] = 1.0/(double)Data.meta.num_examples;
        if (i == 1 || i == 4 || i == 5)
            updated_weights_truth[i] = 1.0/6.0; // 1/num_examples normalizes to 1/6
        else
            updated_weights_truth[i] = 1.0/8.0; // 0.75*1/num_examples normalizes to 1/8;
    }
    
    double beta = update_weights(&weights, Tree, Data);
    
    fail_unless(av_eqf(beta, 0.75), "Value for beta is incorrect");
    fail_unless(! memcmp(weights, updated_weights_truth, Data.meta.num_examples*sizeof(double)), "New weights are incorrect");

    // Memory problems with this
    //_boost_tree_clean_up(&Tree, &Data);
}
END_TEST

Suite *boost_suite(void)
{
    Suite *suite = suite_create("Boost");
    
    TCase *tc_boost = tcase_create(" Check Boost ");
    
    suite_add_tcase(suite, tc_boost);
    tcase_add_test(tc_boost, check_weighted_samples);
    tcase_add_test(tc_boost, check_weighted_samples_with_reset);
    tcase_add_test(tc_boost, check_update_weights);
        
    return suite;
}
