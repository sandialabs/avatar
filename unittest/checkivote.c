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
#include <math.h>
#include <string.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/ivote.h"
#include "../src/util.h"
#include "../src/tree.h"

void _ivote_oob_set_up(DT_Node **nodes, CV_Subset *data, Vote_Cache *cache);
void _ivote_oob_clean_up(DT_Node *nodes, CV_Subset data, Vote_Cache cache);
void _gen_bite_data(int num_examples, CV_Subset *data, Vote_Cache *cache, Args_Opts *args);
void _free_bite_data(CV_Subset data);
void _free_bite_cache(Vote_Cache cache);
void _count_bite_xlicates(CV_Subset data, int size, long **xlicates);
int _check_bite_xlicates(long *xlicates, int pick, int outof);

int _check_bite_xlicates(long *xlicates, int pick, int outof) {
    int i;
    int num_errors = 0;
    double truth;
    double M = pick;
    double N = outof;

    // We have a match if the values agree to the third decimal place.
    // This seems to work well for picking numbers out of 10e6 one time.
    // Picking numbers repeatedly and averaging will get closer but takes more time.
    // My assumption is that it will either be right or way off so this will catch the later.
    for (i = 0; i <= 4; i++) {
        truth = pow((N-1)/N, M-i) * pow(M/N, i+1) / (double)factorial(i);
        if ((double)xlicates[i]/N < truth-0.001 || (double)xlicates[i]/N > truth+0.001)
            num_errors++;
    }
    
    return num_errors;
}

void _count_bite_xlicates(CV_Subset data, int size, long **xlicates) {
    long i;
    long *hits;

    if (size == 0)
        hits = (long *)calloc(data.meta.num_examples, sizeof(long));
    else
        hits = (long *)calloc(size, sizeof(long));
    
    *xlicates = (long *)calloc(10, sizeof(long));
    // Count number of times each example number shows up
    for (i = 0; i < data.meta.num_examples; i++)
        hits[data.examples[i].global_id_num]++;
    // Compute number of misses, single occurences, duplicates, triplicates, ...
    for (i = 0; i < data.meta.num_examples; i++)
        if (hits[i] <= 9)
            (*xlicates)[hits[i]]++;
    free(hits);
}

void _ivote_oob_set_up(DT_Node **nodes, CV_Subset *data, Vote_Cache *cache) {
    int i, j;
    
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
    (*nodes)[3].class_count = (int *)malloc(3 * sizeof(int));
    for (i = 0; i < 3; i++)
        (*nodes)[3].class_count[i] = -1;
    (*nodes)[3].class_probs = (float *)calloc(3, sizeof(float));
    (*nodes)[3].class_probs[0] = 1.0;
    
    (*nodes)[4].branch_type = LEAF;
    (*nodes)[4].Node_Value.class_label = 1;
    (*nodes)[4].class_count = (int *)malloc(3 * sizeof(int));
    for (i = 0; i < 3; i++)
        (*nodes)[4].class_count[i] = -1;
    (*nodes)[4].class_probs = (float *)calloc(3, sizeof(float));
    (*nodes)[4].class_probs[1] = 1.0;
    
    (*nodes)[5].branch_type = LEAF;
    (*nodes)[5].Node_Value.class_label = 2;
    (*nodes)[5].class_count = (int *)malloc(3 * sizeof(int));
    for (i = 0; i < 3; i++)
        (*nodes)[5].class_count[i] = -1;
    (*nodes)[5].class_probs = (float *)calloc(3, sizeof(float));
    (*nodes)[5].class_probs[2] = 1.0;

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
    (*nodes)[7].class_count = (int *)malloc(3 * sizeof(int));
    for (i = 0; i < 3; i++)
        (*nodes)[7].class_count[i] = -1;
    (*nodes)[7].class_probs = (float *)calloc(3, sizeof(float));
    (*nodes)[7].class_probs[1] = 1.0;
    
    (*nodes)[8].branch_type = LEAF;
    (*nodes)[8].Node_Value.class_label = 0;
    (*nodes)[8].class_count = (int *)malloc(3 * sizeof(int));
    for (i = 0; i < 3; i++)
        (*nodes)[8].class_count[i] = -1;
    (*nodes)[8].class_probs = (float *)calloc(3, sizeof(float));
    (*nodes)[8].class_probs[0] = 1.0;
    
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
    
    data->examples = (CV_Example *)malloc(data->meta.num_examples * sizeof(CV_Example));
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
    data->examples[4].containing_class_num = 1;
    data->examples[5].containing_class_num = 0;
    data->examples[6].containing_class_num = 0;
    data->examples[0].in_bag = FALSE;
    data->examples[1].in_bag = FALSE;
    data->examples[2].in_bag = FALSE;
    data->examples[3].in_bag = FALSE;
    data->examples[4].in_bag = FALSE;
    data->examples[5].in_bag = FALSE;
    data->examples[6].in_bag = FALSE;
    
    cache->num_classes = data->meta.num_classes;
    cache->num_train_examples = data->meta.num_examples;
    cache->num_test_examples = data->meta.num_examples;
    cache->oob_error = 0.0;
    cache->best_train_class = (int *)malloc(cache->num_train_examples * sizeof(int));
    // Initialize to -1 so all examples are wrongly classed the first time through
    for (j = 0; j < cache->num_train_examples; j++)
        cache->best_train_class[j] = -1;
    cache->best_test_class = (int *)malloc(cache->num_test_examples * sizeof(int));
    for (j = 0; j < cache->num_test_examples; j++)
        cache->best_test_class[j] = -1;
    
    cache->oob_class_votes = (int **)malloc(cache->num_train_examples * sizeof(int *));
    for (j = 0; j < cache->num_train_examples; j++)
        cache->oob_class_votes[j] = (int *)calloc(cache->num_classes, sizeof(int));
    cache->class_votes_test = (int **)malloc(cache->num_test_examples * sizeof(int *));
    for (j = 0; j < cache->num_test_examples; j++)
        cache->class_votes_test[j] = (int *)calloc(cache->num_classes, sizeof(int));
    
    cache->oob_class_weighted_votes = (float **)malloc(cache->num_train_examples * sizeof(float*));
    for (j = 0; j < cache->num_train_examples; j++)
        cache->oob_class_weighted_votes[j] = (float *)calloc(cache->num_classes, sizeof(float));
    cache->class_weighted_votes_test = (float **)malloc(cache->num_test_examples * sizeof(float*));
    for (j = 0; j < cache->num_test_examples; j++)
        cache->class_weighted_votes_test[j] = (float *)calloc(cache->num_classes, sizeof(float));
}

void _ivote_oob_clean_up(DT_Node *nodes, CV_Subset data, Vote_Cache cache) {
    int i;
    for (i = 0; i < 9; i++)
        if (nodes[i].branch_type == BRANCH)
            free(nodes[i].Node_Value.branch);
    free(nodes);
    
    for (i = 0; i < 3; i++)
        free(data.float_data[i]);
    free(data.float_data);
    for (i = 0; i < data.meta.num_classes; i++)
        free(data.meta.class_names[i]);
    free(data.meta.class_names);
    for (i = 0; i < data.meta.num_examples; i++)
        free(data.examples[i].distinct_attribute_values);
    free(data.examples);

    free(cache.best_train_class);
    free(cache.best_test_class);
    for (i = 0; i < cache.num_train_examples; i++)
        free(cache.oob_class_votes[i]);
    free(cache.oob_class_votes);
    for (i = 0; i < cache.num_test_examples; i++)
        free(cache.class_votes_test[i]);
    free(cache.class_votes_test);
    
}

void _gen_bite_data(int num_examples, CV_Subset *data, Vote_Cache *cache, Args_Opts *args) {
    int i;
    
    args->debug = FALSE;
    args->majority_ivoting = FALSE;
    cache->num_classifiers = 0;
    cache->oob_error = 0.5; // Puts correctly classed examples in with probability 1
    cache->num_train_examples = num_examples;
    cache->best_train_class = (int *)calloc(num_examples, sizeof(int));
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

void _free_bite_data(CV_Subset data) {
    free(data.examples);
}

void _free_bite_cache(Vote_Cache cache) {
    free(cache.best_train_class);
}

START_TEST(check_oob_error_rate)
{
    DT_Node *Nodes = NULL;
    CV_Subset Data = {0};
    Vote_Cache Cache = {0};
    Args_Opts Args = {0};
    
    Args.break_ties_randomly = FALSE;
    _ivote_oob_set_up(&Nodes, &Data, &Cache);
    
    int best_class_truth[][7] = { { -1, 1, -1, 2, -1, 0, 0 }, { 0, 1, 1, 2, -1, 0, 0 }, { 0, 1, 1, 2, 1, 0, 0 } };
    int oob_class_votes_truth[][3] = { { 2, 0, 0 }, { 0, 2, 0 }, { 0, 2, 0 }, { 0, 0, 2 },
                                       { 0, 1, 0 }, { 3, 0, 0 }, { 2, 0, 0 } };
    
    Data.examples[0].in_bag = TRUE;
    Data.examples[2].in_bag = TRUE;
    Data.examples[4].in_bag = TRUE;
    fail_unless(compute_oob_error_rate(Nodes, Data, &Cache, Args) == 0.25, "wrong oob error rate (1)");
    fail_unless(! memcmp(best_class_truth[0], Cache.best_train_class, Data.meta.num_examples*sizeof(int)),
                "best_train_class doesn't match (1)");
    
    Data.examples[0].in_bag = FALSE;
    Data.examples[2].in_bag = FALSE;
    Data.examples[4].in_bag = FALSE;
    Data.examples[3].in_bag = TRUE;
    Data.examples[4].in_bag = TRUE;
    Data.examples[6].in_bag = TRUE;
    fail_unless(compute_oob_error_rate(Nodes, Data, &Cache, Args) == 0.25, "wrong oob error rate (2)");
    fail_unless(! memcmp(best_class_truth[1], Cache.best_train_class, Data.meta.num_examples*sizeof(int)),
                "best_train_class doesn't match (2)");

    Data.examples[3].in_bag = FALSE;
    Data.examples[4].in_bag = FALSE;
    Data.examples[6].in_bag = FALSE;
    Data.examples[1].in_bag = TRUE;
    fail_unless(compute_oob_error_rate(Nodes, Data, &Cache, Args) == 0, "wrong oob error rate (3)");
    fail_unless(! memcmp(best_class_truth[2], Cache.best_train_class, Data.meta.num_examples*sizeof(int)),
                "best_train_class doesn't match (3)");

    fail_unless(! memcmp(oob_class_votes_truth[0], Cache.oob_class_votes[0], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[1], Cache.oob_class_votes[1], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[2], Cache.oob_class_votes[2], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[3], Cache.oob_class_votes[3], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[4], Cache.oob_class_votes[4], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[5], Cache.oob_class_votes[5], Data.meta.num_classes*sizeof(int)) &&
                ! memcmp(oob_class_votes_truth[6], Cache.oob_class_votes[6], Data.meta.num_classes*sizeof(int)),
                "oob_class_votes doesn't match");
    
    _ivote_oob_clean_up(Nodes, Data, Cache);
}
END_TEST

START_TEST(check_bite_creation)
{
    int i;
    int num = 1000000;
    Args_Opts Args = {0};
    CV_Subset Data = {0}, Bite = {0};
    Vote_Cache Cache = {0};
    long **xlicate_counts;
    int num_it = 3;
    
    xlicate_counts = (long **)malloc(num_it * sizeof(long *));
    
    _gen_bite_data(num, &Data, &Cache, &Args);
    
    for (i = 0; i < num_it; i++) {
        Args.bite_size = (int)pow(10.0, (double)(i+3));
        if (i < 2)
            srand48(i+Args.bite_size);
        make_bite(&Data, &Bite, &Cache, Args);
        _count_bite_xlicates(Bite, num, &xlicate_counts[i]);
        fail_unless(! _check_bite_xlicates(xlicate_counts[i], Args.bite_size, num), "distribution incorrect");
        _free_bite_data(Bite);
    }
    _free_bite_cache(Cache);
    _free_bite_data(Data);
    for (i = 0; i < num_it; i++)
        free(xlicate_counts[i]);
    free(xlicate_counts);
}
END_TEST

Suite *ivote_suite(void)
{
    Suite *suite = suite_create("IVote");
    
    TCase *tc_ivote = tcase_create(" Check IVote ");
    suite_add_tcase(suite, tc_ivote);
    tcase_add_test(tc_ivote, check_oob_error_rate);
    tcase_add_test(tc_ivote, check_bite_creation);
    
    return suite;
}
