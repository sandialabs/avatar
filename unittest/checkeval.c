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
#include <time.h>
#include <sys/types.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/evaluate.h"
#include "../src/util.h"
#include "../src/gain.h"

void _set_up(DT_Ensemble *ensemble, CV_Subset *data);
void _set_up_matrix(CV_Matrix *truth_matrix);
void _set_up_boost_matrix(CV_Matrix *truth_matrix);
void _clean_up(DT_Ensemble *ensemble, CV_Subset data);
void _clean_up_matrix(CV_Matrix *matrix);

void _set_up(DT_Ensemble *ensemble, CV_Subset *data) {
    int i;

    ensemble->num_trees = 3;
    ensemble->Trees = (DT_Node **)malloc(ensemble->num_trees * sizeof(DT_Node *));
    for (i = 0; i < ensemble->num_trees; i++)
        ensemble->Trees[i] = (DT_Node *)malloc(9 * sizeof(DT_Node));
    
    ensemble->Trees[0][0].branch_type = BRANCH;
    ensemble->Trees[0][0].attribute = 0;
    ensemble->Trees[0][0].num_branches = 2;
    ensemble->Trees[0][0].branch_threshold = 10.0;
    ensemble->Trees[0][0].attribute_type = CONTINUOUS;
    ensemble->Trees[0][0].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[0][0].Node_Value.branch[0] = 1;
    ensemble->Trees[0][0].Node_Value.branch[1] = 6;
    
    ensemble->Trees[0][1].branch_type = BRANCH;
    ensemble->Trees[0][1].attribute = 1;
    ensemble->Trees[0][1].num_branches = 2;
    ensemble->Trees[0][1].branch_threshold = 8.0;
    ensemble->Trees[0][1].attribute_type = CONTINUOUS;
    ensemble->Trees[0][1].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[0][1].Node_Value.branch[0] = 2;
    ensemble->Trees[0][1].Node_Value.branch[1] = 5;
    
    ensemble->Trees[0][2].branch_type = BRANCH;
    ensemble->Trees[0][2].attribute = 2;
    ensemble->Trees[0][2].num_branches = 2;
    ensemble->Trees[0][2].branch_threshold = 6.0;
    ensemble->Trees[0][2].attribute_type = CONTINUOUS;
    ensemble->Trees[0][2].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[0][2].Node_Value.branch[0] = 3;
    ensemble->Trees[0][2].Node_Value.branch[1] = 4;
    
    ensemble->Trees[0][3].branch_type = LEAF;
    ensemble->Trees[0][3].Node_Value.class_label = 0;
    
    ensemble->Trees[0][4].branch_type = LEAF;
    ensemble->Trees[0][4].Node_Value.class_label = 1;
    
    ensemble->Trees[0][5].branch_type = LEAF;
    ensemble->Trees[0][5].Node_Value.class_label = 2;

    ensemble->Trees[0][6].branch_type = BRANCH;
    ensemble->Trees[0][6].attribute = 2;
    ensemble->Trees[0][6].num_branches = 2;
    ensemble->Trees[0][6].branch_threshold = 15.0;
    ensemble->Trees[0][6].attribute_type = CONTINUOUS;
    ensemble->Trees[0][6].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[0][6].Node_Value.branch[0] = 7;
    ensemble->Trees[0][6].Node_Value.branch[1] = 8;

    ensemble->Trees[0][7].branch_type = LEAF;
    ensemble->Trees[0][7].Node_Value.class_label = 1;
    
    ensemble->Trees[0][8].branch_type = LEAF;
    ensemble->Trees[0][8].Node_Value.class_label = 0;
        
    ensemble->Trees[1][0].branch_type = BRANCH;
    ensemble->Trees[1][0].attribute = 2;
    ensemble->Trees[1][0].num_branches = 2;
    ensemble->Trees[1][0].branch_threshold = 15.0;
    ensemble->Trees[1][0].attribute_type = CONTINUOUS;
    ensemble->Trees[1][0].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[1][0].Node_Value.branch[0] = 1;
    ensemble->Trees[1][0].Node_Value.branch[1] = 8;
    
    ensemble->Trees[1][1].branch_type = BRANCH;
    ensemble->Trees[1][1].attribute = 2;
    ensemble->Trees[1][1].num_branches = 2;
    ensemble->Trees[1][1].branch_threshold = 6.0001;
    ensemble->Trees[1][1].attribute_type = CONTINUOUS;
    ensemble->Trees[1][1].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[1][1].Node_Value.branch[0] = 2;
    ensemble->Trees[1][1].Node_Value.branch[1] = 3;
    
    ensemble->Trees[1][2].branch_type = LEAF;
    ensemble->Trees[1][2].Node_Value.class_label = 0;
    
    ensemble->Trees[1][3].branch_type = BRANCH;
    ensemble->Trees[1][3].attribute = 1;
    ensemble->Trees[1][3].num_branches = 2;
    ensemble->Trees[1][3].branch_threshold = 8.0;
    ensemble->Trees[1][3].attribute_type = CONTINUOUS;
    ensemble->Trees[1][3].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[1][3].Node_Value.branch[0] = 4;
    ensemble->Trees[1][3].Node_Value.branch[1] = 5;
    
    ensemble->Trees[1][4].branch_type = LEAF;
    ensemble->Trees[1][4].Node_Value.class_label = 1;
    
    ensemble->Trees[1][5].branch_type = BRANCH;
    ensemble->Trees[1][5].attribute = 0;
    ensemble->Trees[1][5].num_branches = 2;
    ensemble->Trees[1][5].branch_threshold = 10.0;
    ensemble->Trees[1][5].attribute_type = CONTINUOUS;
    ensemble->Trees[1][5].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[1][5].Node_Value.branch[0] = 6;
    ensemble->Trees[1][5].Node_Value.branch[1] = 7;

    ensemble->Trees[1][6].branch_type = LEAF;
    ensemble->Trees[1][6].Node_Value.class_label = 2;

    ensemble->Trees[1][7].branch_type = LEAF;
    ensemble->Trees[1][7].Node_Value.class_label = 0;
    
    ensemble->Trees[1][8].branch_type = LEAF;
    ensemble->Trees[1][8].Node_Value.class_label = 0;
    
    ensemble->Trees[2][0].branch_type = BRANCH;
    ensemble->Trees[2][0].attribute = 0;
    ensemble->Trees[2][0].num_branches = 2;
    ensemble->Trees[2][0].branch_threshold = 10.0;
    ensemble->Trees[2][0].attribute_type = CONTINUOUS;
    ensemble->Trees[2][0].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[2][0].Node_Value.branch[0] = 1;
    ensemble->Trees[2][0].Node_Value.branch[1] = 6;
    
    ensemble->Trees[2][1].branch_type = BRANCH;
    ensemble->Trees[2][1].attribute = 1;
    ensemble->Trees[2][1].num_branches = 2;
    ensemble->Trees[2][1].branch_threshold = 8.0;
    ensemble->Trees[2][1].attribute_type = CONTINUOUS;
    ensemble->Trees[2][1].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[2][1].Node_Value.branch[0] = 2;
    ensemble->Trees[2][1].Node_Value.branch[1] = 5;
    
    ensemble->Trees[2][2].branch_type = BRANCH;
    ensemble->Trees[2][2].attribute = 2;
    ensemble->Trees[2][2].num_branches = 2;
    ensemble->Trees[2][2].branch_threshold = 6.0;
    ensemble->Trees[2][2].attribute_type = CONTINUOUS;
    ensemble->Trees[2][2].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[2][2].Node_Value.branch[0] = 3;
    ensemble->Trees[2][2].Node_Value.branch[1] = 4;
    
    ensemble->Trees[2][3].branch_type = LEAF;
    ensemble->Trees[2][3].Node_Value.class_label = 0;
    
    ensemble->Trees[2][4].branch_type = LEAF;
    ensemble->Trees[2][4].Node_Value.class_label = 1;
    
    ensemble->Trees[2][5].branch_type = LEAF;
    ensemble->Trees[2][5].Node_Value.class_label = 2;

    ensemble->Trees[2][6].branch_type = BRANCH;
    ensemble->Trees[2][6].attribute = 2;
    ensemble->Trees[2][6].num_branches = 2;
    ensemble->Trees[2][6].branch_threshold = 15.0;
    ensemble->Trees[2][6].attribute_type = CONTINUOUS;
    ensemble->Trees[2][6].Node_Value.branch = (int *)malloc(2 * sizeof(int));
    ensemble->Trees[2][6].Node_Value.branch[0] = 7;
    ensemble->Trees[2][6].Node_Value.branch[1] = 8;

    ensemble->Trees[2][7].branch_type = LEAF;
    ensemble->Trees[2][7].Node_Value.class_label = 1;
    
    ensemble->Trees[2][8].branch_type = LEAF;
    ensemble->Trees[2][8].Node_Value.class_label = 0;
        
    ensemble->boosting_betas = (double *)malloc(ensemble->num_trees * sizeof(double));
    ensemble->boosting_betas[0] = 0.15;
    ensemble->boosting_betas[1] = 0.0224;
    ensemble->boosting_betas[2] = 0.15;

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
}

void _set_up_matrix(CV_Matrix *matrix) {
    int i, j;
    matrix->num_examples = 7;
    matrix->num_classifiers = 3;
    matrix->additional_cols = 1;
    matrix->num_classes = 3;
    int matrix_data[7][4] = {
                              { 0, 0, 0, 0 },
                              { 0, 1, 0, 1 },
                              { 1, 1, 1, 1 },
                              { 2, 2, 2, 2 },
                              { 1, 1, 0, 1 },
                              { 0, 0, 0, 0 },
                              { 0, 0, 0, 0 }
                            };
    matrix->data = (union data_type_union **)malloc(matrix->num_examples * sizeof(union data_type_union *));
    for (i = 0; i < matrix->num_examples; i++) {
        matrix->data[i] = (union data_type_union *)malloc((matrix->num_classifiers + matrix->additional_cols) * sizeof(union data_type_union));
        for (j = 0; j < matrix->num_classifiers; j++) {
            matrix->data[i][j].Integer = matrix_data[i][j];
        }
    }
}


void _set_up_boost_matrix(CV_Matrix *matrix) {
    int i, j;
    matrix->num_examples = 7;
    matrix->num_classifiers = 3;
    matrix->num_classes = 3;
    int matrix_classes[7] = { 0, 0, 1, 2, 1, 0, 0 };
    float matrix_data[7][3] = {
                                { 10.95428865,  0,           0          },
                                {  5.48035746,  5.47393119,  0          },
                                {  0,          10.95428865,  0          },
                                {  0,           0,          10.95428865 },
                                {  5.48035746,  5.47393119,  0          },
                                { 10.95428865,  0,           0          },
                                { 10.95428865,  0,           0          },
                              };
    matrix->data = (union data_type_union **)malloc(matrix->num_examples * sizeof(union data_type_union *));
    matrix->classes = (int *)malloc(matrix->num_examples * sizeof(int));
    for (i = 0; i < matrix->num_examples; i++) {
        matrix->data[i] = (union data_type_union *)malloc(matrix->num_classes * sizeof(union data_type_union));
        matrix->classes[i] = matrix_classes[i];
        for (j = 0; j < matrix->num_classes; j++) {
            matrix->data[i][j].Real = matrix_data[i][j];
        }
    }
}

void _clean_up(DT_Ensemble *ensemble, CV_Subset data) {
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 9; j++)
            if (ensemble->Trees[i][j].branch_type == BRANCH)
                free(ensemble->Trees[i][j].Node_Value.branch);
        free(ensemble->Trees[i]);
    }
    free(ensemble->Trees);
    free(ensemble->boosting_betas);
    
    for (i = 0; i < 3; i++)
        free(data.float_data[i]);
    free(data.float_data);
    for (i = 0; i < data.meta.num_classes; i++)
        free(data.meta.class_names[i]);
    free(data.meta.class_names);
    for (i = 0; i < data.meta.num_examples; i++)
        free(data.examples[i].distinct_attribute_values);
    free(data.examples);
}

void _clean_up_matrix(CV_Matrix *matrix) {
    int i;
    // This does not seem to actually free the matrix.data stuff. Still shows up as lost in valgrind
    for (i = 0; i < matrix->num_examples; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
}

START_TEST(run_cont_tree)
{
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0};
    int leaf_node;
    
    _set_up(&ensemble, &data);
    _set_up_matrix(&matrix);

    fail_unless(count_nodes(ensemble.Trees[0]) == 9, "got wrong node count for tree 2");
    fail_unless(count_nodes(ensemble.Trees[1]) == 9, "got wrong node count for tree 1");

    fail_unless(classify_example(ensemble.Trees[0], data.examples[0], data.float_data, &leaf_node) == 0, "failed to classify example 0 as 0");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[1], data.float_data, &leaf_node) == 1, "failed to classify example 1 as 1");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[2], data.float_data, &leaf_node) == 1, "failed to classify example 2 as 1");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[3], data.float_data, &leaf_node) == 2, "failed to classify example 3 as 2");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[4], data.float_data, &leaf_node) == 1, "failed to classify example 4 as 1");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[5], data.float_data, &leaf_node) == 0, "failed to classify example 5 as 0");
    fail_unless(classify_example(ensemble.Trees[0], data.examples[6], data.float_data, &leaf_node) == 0, "failed to classify example 6 as 0");
    
    fail_unless(classify_example(ensemble.Trees[1], data.examples[0], data.float_data, &leaf_node) == 0, "failed to classify example 0 as 0");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[1], data.float_data, &leaf_node) == 0, "failed to classify example 1 as 0");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[2], data.float_data, &leaf_node) == 1, "failed to classify example 2 as 1");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[3], data.float_data, &leaf_node) == 2, "failed to classify example 3 as 2");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[4], data.float_data, &leaf_node) == 0, "failed to classify example 4 as 0");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[5], data.float_data, &leaf_node) == 0, "failed to classify example 5 as 0");
    fail_unless(classify_example(ensemble.Trees[1], data.examples[6], data.float_data, &leaf_node) == 0, "failed to classify example 6 as 0");
    
    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

START_TEST(run_build_matrix)
{
    int i, j;
    
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0}, truth = {0};
    
    _set_up(&ensemble, &data);
    _set_up_matrix(&truth);
    
    build_prediction_matrix(data, ensemble, &matrix);

    int num_errors = 0;
    if (truth.num_examples != matrix.num_examples)
        num_errors++;
    if (truth.num_classifiers != matrix.num_classifiers)
        num_errors++;
    if (truth.num_classes != matrix.num_classes)
        num_errors++;
    for (i = 0; i < truth.num_examples; i++)
        for (j = 0; j < truth.num_classifiers; j++)
            if (truth.data[i][j].Integer != matrix.data[i][j].Integer)
                num_errors++;

    fail_unless(num_errors == 0, "failed to build matrix correctly");

    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

START_TEST(run_build_boost_matrix)
{
    int i, j;
    
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0}, truth = {0};
    
    _set_up(&ensemble, &data);
    _set_up_boost_matrix(&truth);
    
    build_boost_prediction_matrix(data, ensemble, &matrix);

    int num_errors = 0;
    if (truth.num_examples != matrix.num_examples)
        num_errors++;
    if (truth.num_classifiers != matrix.num_classifiers)
        num_errors++;
    if (truth.num_classes != matrix.num_classes)
        num_errors++;
    for (i = 0; i < truth.num_examples; i++) {
        if (truth.classes[i] != matrix.classes[i])
            num_errors++;
        for (j = 0; j < truth.num_classifiers; j++)
            if (! av_eqf(truth.data[i][j].Real, matrix.data[i][j].Real))
                num_errors++;
    }

    fail_unless(num_errors == 0, "failed to build matrix correctly");

    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

START_TEST(run_conf_matrix)
{
    int i, j;
    CV_Matrix matrix = {0};
    Args_Opts args = {0};
    int **confusion_matrix;
    int **truth;
    
    matrix.num_examples = 10;
    matrix.num_classifiers = 5;
    matrix.additional_cols = 1;
    matrix.num_classes = 3;
    
    args.do_boosting = FALSE;
    args.break_ties_randomly = FALSE;
    /* WARNING WARNING WARNING
       
       It is CRUCIAL that THIS TEST be the FIRST one to call compute_voted_accuracy().
       It is CRUCIAL that check_accuracies be run immediately after this test, or at least no intervening
           test call compute_voted_accuracy().
       It is CRUCIAL that the random_seed be set to 7 here as this ensures that check_accuracies PASSES
       
       WARNING WARNING WARNING
     */
    args.random_seed = 7;

    int matrix_data[10][6] = {
                              { 0, 0, 0, 0, 2, 1 },
                              { 0, 1, 0, 1, 1, 2 },
                              { 1, 1, 1, 2, 1, 2 },
                              { 0, 0, 0, 1, 2, 0 },
                              { 0, 1, 0, 1, 1, 0 },
                              { 1, 1, 1, 2, 2, 2 },
                              { 2, 2, 1, 0, 1, 1 },
                              { 1, 1, 2, 2, 2, 0 },
                              { 0, 0, 2, 0, 0, 1 },
                              { 0, 0, 0, 1, 1, 1 }
                            };
    matrix.data = (union data_type_union **)malloc(matrix.num_examples * sizeof(union data_type_union *));
    for (i = 0; i < matrix.num_examples; i++) {
        matrix.data[i] = (union data_type_union *)malloc((matrix.num_classifiers+matrix.additional_cols) * sizeof(union data_type_union));
        for (j = 0; j < matrix.num_classifiers+matrix.additional_cols; j++)
            matrix.data[i][j].Integer = matrix_data[i][j];
    }
                            
    char *class_names[matrix.num_classes];
    class_names[0] = "Audi";
    class_names[1] = "Maserati";
    class_names[2] = "Land Rover";
    
    truth = (int **)calloc(matrix.num_classes, sizeof(int *));
    for (i = 0; i < matrix.num_classes; i++)
        truth[i] = (int *)calloc(matrix.num_classes, sizeof(int));
    truth[0][0] = 3;
    truth[1][0] = 3;
    truth[1][1] = 1;
    truth[1][2] = 1;
    truth[2][1] = 2;

    //_generate_confusion_matrix(matrix, &confusion_matrix, args);
    compute_voted_accuracy(matrix, &confusion_matrix, args);
    
    // ????? running memcmp on the 2-D array did not work. Are the 'stripes' not contiguous???
    for (i = 0; i < matrix.num_classes; i++)
        fail_unless(! memcmp(truth[i], confusion_matrix[i], matrix.num_classes*sizeof(int)),
                    "confusion matrix incorrect");
                
    for (i = 0; i < matrix.num_classes; i++) {
        free(confusion_matrix[i]);
        free(truth[i]);
    }
    free(confusion_matrix);
    free(truth);
    for (i = 0; i < matrix.num_examples; i++)
        free(matrix.data[i]);
    free(matrix.data);

}
END_TEST

START_TEST(check_accuracies)
{    
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0}, truth = {0};
    int **Confusion;
    Args_Opts args = {0};
    
    _set_up(&ensemble, &data);
    _set_up_matrix(&truth);
    // Only consider the first two trees. This tests random-tie-breaks, too, since there will
    // now be some ties.
    ensemble.num_trees = 2;
    args.break_ties_randomly = TRUE;
    args.random_seed = 5; // This has no effect because the RNG is initiated in the
                          // previous test with the args.random_seed value defined
                          // there and there is no mechanism for re-initializing the
                          // RNG to a user-defined seed.
    
    build_prediction_matrix(data, ensemble, &matrix);
    
    // This is a bit of a weird test because I can't reseed the tie-breaking RNG with a user-defined value
    // once it's been initialized by the first call. Therefore, to get all possible combinations of tie-breaks
    // (i.e. all correct, #1 wrong, #4 wrong, #1,4 wrong) I need to recompute the voted accuracy over and over
    // until the next new seed produces the desired result. This is only a hassle for unit tests and does not,
    // that I can forsee, affect the algorithm.
    
    /* With an initial seed in the previous test of 7, this gets sample #4 wrong */
    float va = compute_voted_accuracy(matrix, &Confusion, args);
    float aa = compute_average_accuracy(matrix);
    //printf("va = %.6f\n", va);
    fail_unless(av_eqf(va, 6.0/7.0), "voted accuracy is incorrect (1)");
    fail_unless(av_eqf(aa, 12.0/14.0), "average accuracy is incorrect (1)");
    va = compute_voted_accuracy(matrix, &Confusion, args);
    /* This gets all samples correct */
    va = compute_voted_accuracy(matrix, &Confusion, args);
    //printf("va = %.6f\n", va);
    fail_unless(av_eqf(va, 7.0/7.0), "voted accuracy is incorrect (3)");
    va = compute_voted_accuracy(matrix, &Confusion, args);
    va = compute_voted_accuracy(matrix, &Confusion, args);
    /* This gets sample #1 wrong */
    va = compute_voted_accuracy(matrix, &Confusion, args);
    //printf("va = %.6f\n", va);
    fail_unless(av_eqf(va, 6.0/7.0), "voted accuracy is incorrect (4)");
    va = compute_voted_accuracy(matrix, &Confusion, args);
    va = compute_voted_accuracy(matrix, &Confusion, args);
    va = compute_voted_accuracy(matrix, &Confusion, args);
    va = compute_voted_accuracy(matrix, &Confusion, args);
    va = compute_voted_accuracy(matrix, &Confusion, args);
    /* This gets samples #1, 4 wrong */
    va = compute_voted_accuracy(matrix, &Confusion, args);
    //printf("va = %.6f, looking for %.6f\n", va, 6.0/7.0);
    fail_unless(av_eqf(va, 6.0/7.0), "voted accuracy is incorrect (4)");
    
    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

START_TEST(check_boost_accuracies)
{
    int i;
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0};
    int **Confusion;
    Args_Opts args = {0};
    
    int truth_conf[3][3] = {
                             { 4, 1, 0 },
                             { 0, 1, 0 },
                             { 0, 0, 1 }
                           };
    
    _set_up(&ensemble, &data);
    args.break_ties_randomly = TRUE;
    
    build_boost_prediction_matrix(data, ensemble, &matrix);
    fail_unless(av_eqf(compute_boosting_accuracy(matrix, &Confusion), 6.0/7.0), "voted accuracy is incorrect");
    //fail_unless(av_eqf(compute_average_accuracy(matrix), 12.0/14.0), "average accuracy is incorrect");
    for (i = 0; i < matrix.num_classes; i++)
        fail_unless(! memcmp(truth_conf[i], Confusion[i], matrix.num_classes*sizeof(int)),
                    "confusion matrix incorrect");
    
    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

START_TEST(check_best_class)
{
}
END_TEST

START_TEST(check_count_votes)
{
    DT_Ensemble ensemble = {0};
    CV_Subset data = {0};
    CV_Matrix matrix = {0}, truth = {0};
    int *votes;
    
    _set_up(&ensemble, &data);
    _set_up_matrix(&truth);
    ensemble.num_trees = 2;
    
    build_prediction_matrix(data, ensemble, &matrix);
    count_class_votes_from_matrix(0, matrix, &votes);
    fail_unless(votes[0] == 2 && votes[1] == 0 && votes[2] == 0, "wrong vote count for example 0");
    count_class_votes_from_matrix(1, matrix, &votes);
    fail_unless(votes[0] == 1 && votes[1] == 1 && votes[2] == 0, "wrong vote count for example 1");
    count_class_votes_from_matrix(2, matrix, &votes);
    fail_unless(votes[0] == 0 && votes[1] == 2 && votes[2] == 0, "wrong vote count for example 2");
    count_class_votes_from_matrix(3, matrix, &votes);
    fail_unless(votes[0] == 0 && votes[1] == 0 && votes[2] == 2, "wrong vote count for example 3");
    count_class_votes_from_matrix(4, matrix, &votes);
    fail_unless(votes[0] == 1 && votes[1] == 1 && votes[2] == 0, "wrong vote count for example 4");
    count_class_votes_from_matrix(5, matrix, &votes);
    fail_unless(votes[0] == 2 && votes[1] == 0 && votes[2] == 0, "wrong vote count for example 5");
    count_class_votes_from_matrix(6, matrix, &votes);
    fail_unless(votes[0] == 2 && votes[1] == 0 && votes[2] == 0, "wrong vote count for example 6");
    _clean_up(&ensemble, data);
    _clean_up_matrix(&matrix);
}
END_TEST

Suite *eval_suite(void)
{
    Suite *suite = suite_create("Evaluate");
    
    TCase *tc_classify = tcase_create(" Check Classify ");
    suite_add_tcase(suite, tc_classify);
    tcase_add_test(tc_classify, run_cont_tree);
    
    TCase *tc_matrix = tcase_create(" Check Matrix Ops ");
    suite_add_tcase(suite, tc_matrix);
    tcase_add_test(tc_matrix, run_build_matrix);
    tcase_add_test(tc_matrix, run_conf_matrix);
    tcase_add_test(tc_matrix, check_accuracies);
    //tcase_add_test(tc_matrix, check_best_class);
    tcase_add_test(tc_matrix, check_count_votes);
    tcase_add_test(tc_matrix, run_build_boost_matrix);
    tcase_add_test(tc_matrix, check_boost_accuracies);
    
    return suite;
}
