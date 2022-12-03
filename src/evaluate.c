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
#include "crossval.h"
#include "evaluate.h"
#include "util.h"
#include "gain.h"
#include "av_rng.h"
#include "tree.h"

typedef struct sortstore {
  double value;
  int class;
} continuous_sort;


int count_nodes(DT_Node *tree) {
    int count = 1;
    _count_nodes(tree, 0, &count);
    return count;
}

void _count_nodes(DT_Node *tree, int node, int *count) {
    int i;
    if (tree[node].branch_type != LEAF) {
        for (i = 0; i < tree[node].num_branches; i++) {
            (*count)++;
            _count_nodes(tree, tree[node].Node_Value.branch[i], count);
        }
    }
}

//Added by DACIESL June-05-08: Laplacean Estimates
//external call to find where the example falls in the tree and returns the class probabilities
float *find_example_probabilities(DT_Node *tree, CV_Example example, float **xlate, int *leaf_node) {
    //printf("START ...\n");
    return _find_example_probabilities(tree, 0, example, xlate, leaf_node);
}

//Added by DACIESL June-05-08: Laplacean Estimates
//recurses through tree to find where the example falls in the tree and returns the class probabilities
float * _find_example_probabilities(DT_Node *tree, int node, CV_Example example, float **xlate, int *leaf_node) {
  //int class_label = -1;
    float *class_probs=NULL;
    //printf("HERE with %d node %d\n", tree[node].branch_type, node);
    if (tree[node].branch_type == LEAF) {
        *leaf_node = node;
        //printf("LEAF %d PROB 0/1: %f/%f\n", node, tree[node].class_probs[0], tree[node].class_probs[1]);
        return tree[node].class_probs;
    } else {
        if (tree[node].attribute_type == CONTINUOUS) {
            if (xlate[tree[node].attribute][example.distinct_attribute_values[tree[node].attribute]] <
                                                                                        tree[node].branch_threshold)
                class_probs = _find_example_probabilities(tree, tree[node].Node_Value.branch[0], example, xlate, leaf_node);
            else
                class_probs = _find_example_probabilities(tree, tree[node].Node_Value.branch[1], example, xlate, leaf_node);
        } else if (tree[node].attribute_type == DISCRETE) {
            int b = example.distinct_attribute_values[tree[node].attribute];
            class_probs = _find_example_probabilities(tree, tree[node].Node_Value.branch[b], example, xlate, leaf_node);
        }
    }
    
    return class_probs;
}

int classify_example(DT_Node *tree, CV_Example example, float **xlate, int *leaf_node) {
    return _classify_example(tree, 0, example, xlate, leaf_node);
}

int _classify_example(DT_Node *tree, int node, CV_Example example, float **xlate, int *leaf_node) {
    int class_label = -1;
    
    if (tree[node].branch_type == LEAF) {
        *leaf_node = node;
        return tree[node].Node_Value.class_label;
    } else {
        if (tree[node].attribute_type == CONTINUOUS) {
            if (xlate[tree[node].attribute][example.distinct_attribute_values[tree[node].attribute]] <
                                                                                        tree[node].branch_threshold)
                class_label = _classify_example(tree, tree[node].Node_Value.branch[0], example, xlate, leaf_node);
            else
                class_label = _classify_example(tree, tree[node].Node_Value.branch[1], example, xlate, leaf_node);
        } else if (tree[node].attribute_type == DISCRETE) {
            int b = example.distinct_attribute_values[tree[node].attribute];
            class_label = _classify_example(tree, tree[node].Node_Value.branch[b], example, xlate, leaf_node);
        }
    }
    
    return class_label;
}

void print_pred_matrix(char *pre, CV_Matrix matrix) {
    int i, j;
    for (i = 0; i < matrix.num_examples; i++)
        for (j = 0; j < matrix.num_classifiers + matrix.additional_cols; j++)
            printf("%s:matrix[%d][%d] = %d\n", pre, i, j, matrix.data[i][j].Integer);
}

void print_boosting_pred_matrix(char *pre, CV_Matrix matrix) {
    int i, j;
    for (i = 0; i < matrix.num_examples; i++)
        for (j = 0; j < matrix.num_classifiers + matrix.additional_cols; j++)
            printf("%s:matrix[%d][%d] = %f\n", pre, i, j, matrix.data[i][j].Real);
}

void build_prediction_matrix_for_ivote(CV_Subset data, Vote_Cache cache, CV_Matrix *matrix) {
    int i, j;
    
    matrix->data = (union data_type_union **)malloc(cache.num_test_examples * sizeof(union data_type_union *));
    for (i = 0; i < cache.num_test_examples; i++)
        matrix->data[i] = (union data_type_union *)malloc((cache.num_classifiers + 1) * sizeof(union data_type_union));
    matrix->num_examples = cache.num_test_examples;
    matrix->num_classifiers = cache.num_classifiers;
    matrix->additional_cols = 1;
    matrix->num_classes = cache.num_classes;
    
    for (i = 0; i < cache.num_test_examples; i++) {
        matrix->data[i][0].Integer = data.examples[i].containing_class_num;
        for (j = 0; j < cache.num_classifiers; j++)
            matrix->data[i][j+1].Integer = cache.best_test_class[i];
    }
}

//Added by DACIESL June-11-08: Laplacean Estimates
//constructs a matrix of Laplacean probability estimates
void build_probability_matrix_for_ivote(Vote_Cache cache, CV_Prob_Matrix *matrix) {
    int i, j;
    float tmp;

    matrix->data = (float **)malloc(cache.num_test_examples * sizeof(float*));
    for (i = 0; i < cache.num_test_examples; i++)
        matrix->data[i] = (float *)malloc((cache.num_classifiers) * sizeof(float));
    matrix->num_examples = cache.num_test_examples;
    matrix->num_classes = cache.num_classes;
    
    for (i = 0; i < cache.num_test_examples; i++) {
        tmp = 0.0;
        for (j = 0; j < cache.num_classes; j++) {
            tmp += cache.class_weighted_votes_test[i][j];
	}
        for (j = 0; j < cache.num_classes; j++) {
	  matrix->data[i][j] = cache.class_weighted_votes_test[i][j]/tmp;
	}
    }
}

//Added by DACIESL June-05-08: Laplacean Estimates
//constructs a matrix of Laplacean probability estimates
void build_probability_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Prob_Matrix *matrix) {
    int i, j, k;
    int leaf_node;
    float *class_probs;

    matrix->data = (float **)malloc(data.meta.num_examples * sizeof(float *));
    for (i = 0; i < data.meta.num_examples; i++)
        matrix->data[i] = (float *)calloc((ensemble.num_classes), sizeof(float));
    matrix->num_examples = data.meta.num_examples;
    matrix->num_classes = data.meta.num_classes;

    for (i = 0; i < data.meta.num_examples; i++) {
        for (j = 0; j < ensemble.num_trees; j++) {
	    leaf_node=0;
            class_probs = find_example_probabilities(ensemble.Trees[j], data.examples[i], data.float_data, &leaf_node);
            for (k = 0; k < data.meta.num_classes; k++) {
	        matrix->data[i][k] += class_probs[k]/(double)ensemble.num_trees;
		//printf("%f%s",class_probs[k],(k==data.meta.num_classes-1)?"\n":" ");
            }
        }
    }
    //print_pred_matrix("other", *matrix);
}

void build_prediction_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Matrix *matrix) {
    int i, j;
    int leaf_node;

    matrix->data = (union data_type_union **)malloc(data.meta.num_examples * sizeof(union data_type_union*));
    for (i = 0; i < data.meta.num_examples; i++)
        matrix->data[i] = (union data_type_union *)malloc((ensemble.num_trees + 1) * sizeof(union data_type_union));
    matrix->num_examples = data.meta.num_examples;
    matrix->num_classifiers = ensemble.num_trees;
    matrix->additional_cols = 1;
    matrix->num_classes = data.meta.num_classes;
    
    for (i = 0; i < data.meta.num_examples; i++) {
        matrix->data[i][0].Integer = data.examples[i].containing_class_num;
        for (j = 0; j < ensemble.num_trees; j++)
            matrix->data[i][j+1].Integer = classify_example(ensemble.Trees[j], data.examples[i], data.float_data, &leaf_node);
    }
    //print_pred_matrix("other", *matrix);
}


void build_boost_prediction_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Matrix *matrix) {
    int i, j;
    int leaf_node;

    matrix->data = (union data_type_union **)malloc(data.meta.num_examples * sizeof(union data_type_union *));
    for (i = 0; i < data.meta.num_examples; i++)
        matrix->data[i] = (union data_type_union *)calloc(data.meta.num_classes, sizeof(union data_type_union));
    matrix->classes = (int *)malloc(data.meta.num_examples * sizeof(int));
    matrix->num_examples = data.meta.num_examples;
    matrix->num_classifiers = ensemble.num_trees;
    matrix->additional_cols = 0;
    matrix->num_classes = data.meta.num_classes;
    
    for (i = 0; i < data.meta.num_examples; i++) {
        matrix->classes[i] = data.examples[i].containing_class_num;
        for (j = 0; j < ensemble.num_trees; j++) {
            int this_class = classify_example(ensemble.Trees[j], data.examples[i], data.float_data, &leaf_node);
            matrix->data[i][this_class].Real += (float)dlog_2(1.0/ensemble.boosting_betas[j]);
        }
    }
    //print_pred_matrix("other", *matrix);
}

//Added by DACIESL June-05-08: Laplacean Estimates
//constructs a matrix of Laplacean probability estimates
void build_boost_probability_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Prob_Matrix *matrix) {
    int i, j, k;
    int leaf_node;
    float *class_probs;
    matrix->data = (float **)malloc(data.meta.num_examples * sizeof(float*));
    for (i = 0; i < data.meta.num_examples; i++)
        matrix->data[i] = (float *)calloc((data.meta.num_classes), sizeof(float));
    matrix->num_examples = data.meta.num_examples;
    matrix->num_classes = data.meta.num_classes;
    
    double sum_betas = 0.0;
    for (j = 0; j < ensemble.num_trees; j++)
        sum_betas += ensemble.boosting_betas[j];
    for (i = 0; i < data.meta.num_examples; i++) {
        for (j = 0; j < ensemble.num_trees; j++) {
            class_probs = find_example_probabilities(ensemble.Trees[j], data.examples[i], data.float_data, &leaf_node);
            for (k = 0; k < data.meta.num_classes; k++) 
	        matrix->data[i][k] += (ensemble.boosting_betas[j]*class_probs[k])/sum_betas;
        }
    }
    //print_pred_matrix("other", *matrix);
}

//Added by DACIESL June-11-08: Laplacean Estimates
//constructs a matrix of Laplacean probability estimates
/* WARNING: This does NOT give the correct answer */
void build_boost_probability_matrix_for_ivote(Vote_Cache cache, CV_Prob_Matrix *matrix) {
  int i, j;
    
    matrix->data = (float **)malloc(cache.num_test_examples * sizeof(float*));
    for (i = 0; i < cache.num_test_examples; i++)
        matrix->data[i] = (float *)malloc((cache.num_classes) * sizeof(float));
    matrix->num_examples = cache.num_test_examples;
    matrix->num_classes = cache.num_classes;
    
    for (i = 0; i < cache.num_test_examples; i++) {
        for (j = 0; j < cache.num_classes; j++)
	    matrix->data[i][j] = cache.class_weighted_votes_test[i][j]/(double)cache.num_classifiers;
    }
}

/* WARNING: This does NOT give the correct answer */

void build_boost_prediction_matrix_for_ivote(Vote_Cache cache, CV_Matrix *matrix) {
    int i, j;
    
    matrix->data = (union data_type_union **)malloc(cache.num_test_examples * sizeof(union data_type_union *));
    for (i = 0; i < cache.num_test_examples; i++)
        matrix->data[i] = (union data_type_union *)calloc(cache.num_classes, sizeof(union data_type_union));
    matrix->num_examples = cache.num_test_examples;
    matrix->num_classifiers = cache.num_classifiers;
    matrix->additional_cols = 0;
    matrix->num_classes = cache.num_classes;
    
    for (i = 0; i < cache.num_test_examples; i++) {
        for (j = 0; j < cache.num_classes; j++) {
            int this_class = cache.best_test_class[i];
            matrix->data[i][this_class].Real += 1.0; // NEED BOOSTING BETA VALUES HERE
        }
    }
}

void concat_ensembles(int num_ensembles, DT_Ensemble *in, DT_Ensemble *out) {
    int i, j, k, m;
    
    out->Trees = (DT_Node **)malloc(sizeof(DT_Node *));
    out->Books = (Tree_Bookkeeping*)malloc(sizeof(Tree_Bookkeeping));

    // Copy metadata but ignore Books and weights (for now)
    out->num_trees = 0;
    out->num_classes = in[0].num_classes;
    out->num_attributes = in[0].num_attributes;
    free(out->attribute_types);
    out->attribute_types = (Attribute_Type *)malloc(out->num_attributes * sizeof(Attribute_Type));
    free(out->Missing);
    out->Missing = (union data_point_union *)malloc(out->num_attributes * sizeof(union data_point_union));
    for (i = 0; i < out->num_attributes; i++) {
        out->attribute_types[i] = in[0].attribute_types[i];
/* For now, assign missing values for the concatted ensemble as those for the first ensemble.
 * This will not be accurate for all ensembles but not sure what else to do
 */
        out->Missing[i] = in[0].Missing[i];
    }
    
    // Add trees
    for (i = 0; i < num_ensembles; i++) {
        out->Trees = (DT_Node **)realloc(out->Trees, (out->num_trees + in[i].num_trees) * sizeof(DT_Node *));
        out->Books = (Tree_Bookkeeping*)realloc(out->Books,(out->num_trees + in[i].num_trees) * sizeof(Tree_Bookkeeping));
        for (j = 0; j < in[i].num_trees; j++) {
            int n = out->num_trees + j;
            out->Trees[n] = (DT_Node *)malloc(in[i].Books[j].next_unused_node * sizeof(DT_Node));
            out->Books[n].next_unused_node = out->Books[n].num_malloced_nodes = in[i].Books[j].next_unused_node;
            for (k = 0; k < in[i].Books[j].next_unused_node; k++) {
                out->Trees[n][k].branch_type = in[i].Trees[j][k].branch_type;
                out->Trees[n][k].attribute = in[i].Trees[j][k].attribute;
                out->Trees[n][k].num_branches = in[i].Trees[j][k].num_branches;
                out->Trees[n][k].num_errors = in[i].Trees[j][k].num_errors;
                out->Trees[n][k].branch_threshold = in[i].Trees[j][k].branch_threshold;
                out->Trees[n][k].attribute_type = in[i].Trees[j][k].attribute_type;
                if (out->Trees[n][k].branch_type == LEAF) {
                    int jj=in[i].Trees[j][k].Node_Value.class_label;
                    // Add a sanity check on the concat
                    if (jj >= out->num_classes || jj < 0) {
                        printf("Concat Error: Class label %d found (only %d allowed)\n",jj,out->num_classes);exit(1);
                    }
                    out->Trees[n][k].Node_Value.class_label = in[i].Trees[j][k].Node_Value.class_label;
                } else if (out->Trees[n][k].branch_type == BRANCH) {
                    out->Trees[n][k].Node_Value.branch = (int *)malloc(out->Trees[n][k].num_branches * sizeof(int));
                    for (m = 0; m < out->Trees[n][k].num_branches; m++)
                        out->Trees[n][k].Node_Value.branch[m] = in[i].Trees[j][k].Node_Value.branch[m];
                }
            }
        }
        out->num_trees += in[i].num_trees;
    }
}

float compute_voted_accuracy(CV_Matrix matrix, int ***confusion_matrix, Args_Opts args) {
    int i;
    int best_class;
    int correct = 0;
    
    (*confusion_matrix) = (int **)malloc(matrix.num_classes * sizeof(int *));
    for (i = 0; i < matrix.num_classes; i++)
        (*confusion_matrix)[i] = (int *)calloc(matrix.num_classes, sizeof(int));
    for (i = 0; i < matrix.num_examples; i++) {
        //printf("EXAMPLE:%d TRUTH:%d VOTES:", i, matrix.data[i][0].Integer);
      best_class = find_best_class_from_matrix(i, matrix, args, i==0?-1:i, 0);
        //printf("WINNER:%d\n", best_class);
        if (matrix.data[i][0].Integer == best_class) {
            //printf("Sample %d is correct\n", i);
            correct++;
        } else {
            //printf("Sample %d is incorrect\n", i);
        }
        // Update confusion matrix
        (*confusion_matrix)[best_class][matrix.data[i][0].Integer]++;
    }
    return (float)correct/(float)matrix.num_examples;
}

/*
   Computes voted accuracy from the Boost_Matrix structure.
   This is the weighted voted accuracy
   
   TO DO:
   1) Do I need to worry about ties? I would think that they would be very rare
 */
float compute_boosting_accuracy(CV_Matrix matrix, int ***confusion_matrix) {
    int i, j;
    int best_class;
    float max_value;
    int correct = 0;
    
    (*confusion_matrix) = (int **)malloc(matrix.num_classes * sizeof(int *));
    for (i = 0; i < matrix.num_classes; i++)
        (*confusion_matrix)[i] = (int *)calloc(matrix.num_classes, sizeof(int));
    
    for (i = 0; i < matrix.num_examples; i++) {
        //printf("EXAMPLE:%d TRUTH:%d VOTES:", i, matrix.data[i][0]);
        best_class = 0;
        max_value = matrix.data[i][0].Real;
        for (j = 1; j < matrix.num_classes; j++) {
            if (matrix.data[i][j].Real > max_value) {
                best_class = j;
                max_value = matrix.data[i][j].Real;
            }
        }
        //printf("WINNER:%d\n", best_class);
        if (matrix.classes[i] == best_class)
            correct++;
        // Update confusion matrix
        (*confusion_matrix)[best_class][matrix.classes[i]]++;
    }
    
    return (float)correct/(float)matrix.num_examples;
}

void count_class_votes_from_matrix(int example_num, CV_Matrix matrix, int **votes) {
    int i;
    (*votes) = (int *)calloc(matrix.num_classes, sizeof(int));
    for (i = matrix.additional_cols; i < matrix.num_classifiers + matrix.additional_cols; i++)
        (*votes)[matrix.data[example_num][i].Integer]++;
}

int find_best_class_from_matrix(int example_num, CV_Matrix matrix, Args_Opts args, int seed_flag, int clean) {
    int i;
    int num_ties = 0;
    int tied_classes[matrix.num_classes];
    int best_class = -1;

    //static gsl_rng *R = NULL;
    //static long rng_seed;
    static struct ParkMiller* rng;
    static long rng_seed;

    if (clean == 1)
    {
      if (rng != NULL)
      {
        free(rng);
        rng = NULL;
      }
      return -1;
    }
    
    // seed > 0 means do nothing to the generator, just use it as is
    // seed = 0 means set seed to previous value to reproduce the stream
    // seed < 0 means use the next RN as the seed for the RNG
    /*
    if (R == NULL) {
        R = gsl_rng_alloc(gsl_rng_ranlxs2);
        rng_seed = args.random_seed;
        //printf("Initialize RNG with %ld\n", rng_seed);
        gsl_rng_set(R, rng_seed);
    }
    if (seed_flag == 0) {
        //printf("Seeding with old value of %ld\n", rng_seed);
        gsl_rng_set(R, rng_seed);
    } else if (seed_flag < 0) {
        rng_seed = gsl_rng_get(R);
        //printf("Seeding with new value of %ld\n", rng_seed);
        gsl_rng_set(R, rng_seed);
    } else {
        //printf("Using current seed\n");
        }*/
    if (rng == NULL)
    {
      rng = malloc(sizeof(struct ParkMiller));
      rng_seed = args.random_seed;
      av_pm_default_init(rng, rng_seed);
    }
    if (seed_flag == 0)
    {
      av_pm_default_init(rng, rng_seed);
    }
    else if (seed_flag < 0)
    {
      rng_seed = rng->state;
      av_pm_default_init(rng, rng_seed);
    }
    
    if (args.do_boosting == TRUE) {
        float highest_prob = matrix.data[example_num][0].Real;
        tied_classes[num_ties++] = 0;
        for (i = 1; i < matrix.num_classes; i++) {
            if (av_eqf(matrix.data[example_num][i].Real, highest_prob)) {
                tied_classes[num_ties++] = i;
            } else if (matrix.data[example_num][i].Real > highest_prob) {
                highest_prob = matrix.data[example_num][i].Real;
                num_ties = 0;
                tied_classes[num_ties++] = i;
            }
        }
    } else {
        int most_votes = -1;
        int *count_classifications;
        count_classifications = (int *)calloc(matrix.num_classes, sizeof(int));
        for (i = matrix.additional_cols; i < matrix.num_classifiers + matrix.additional_cols; i++)
            count_classifications[matrix.data[example_num][i].Integer]++;
        for (i = 0; i < matrix.num_classes; i++) {
            //printf("%d ", count_classifications[i]);
            if (count_classifications[i] > most_votes) {
                // This class is better than all the rest
                most_votes = count_classifications[i];
                num_ties = 0;
                tied_classes[num_ties++] = i;
                //best_class = i;
            } else if (count_classifications[i] == most_votes) {
                // This class is as good as the current best
                tied_classes[num_ties++] = i;
            }
        }
        free(count_classifications);
    }
        
    //if (num_ties > 1)
    //    printf("Must break a tie...\n");
    if (num_ties == 1 || args.break_ties_randomly == FALSE) {
        // Have one clear winner
        best_class = tied_classes[0];
    } else {
        // Randomly choose from among the best classes --
        // either actually randomly or from array holding previous random choice
        //printf("Example %d has a tie among", example_num);
        //for (i = 0; i < num_ties; i++)
        //    printf(" %d", tied_classes[i]);
        //printf("\n");
        //best_class = tied_classes[gsl_rng_uniform_int(R, num_ties)];
        best_class = tied_classes[av_pm_uniform_int(rng, num_ties)];
    }
    
    if (best_class < 0) {
        fprintf(stderr, "Error: Did not find a best class for example %d. Returning class 0\n", example_num);
        return 0;
    }
    //printf("WINNER %d:%d\n", i, best_class);
    return best_class;
}

float compute_average_accuracy(CV_Matrix matrix) {
    int i, j;
    int correct = 0;
    
    for (i = 0; i < matrix.num_examples; i++)
        for (j = matrix.additional_cols; j < matrix.num_classifiers + matrix.additional_cols; j++)
            if (matrix.data[i][j].Integer == matrix.data[i][0].Integer)
                correct++;
    return (float)correct/(float)(matrix.num_examples * matrix.num_classifiers);
}

/*
 * Unused at this time
 *
void _generate_confusion_matrix(CV_Matrix matrix, int ***confusion_matrix, Args_Opts args) {
    int i, j;
    int most_votes;
    int num_ties;
    int tied_classes[matrix.num_classes];
    int best_class = -1;
    int *count_classifications;
    
    (*confusion_matrix) = (int **)malloc(matrix.num_classes * sizeof(int *));
    for (i = 0; i < matrix.num_classes; i++)
        (*confusion_matrix)[i] = (int *)calloc(matrix.num_classes, sizeof(int));
    
    count_classifications = (int *)malloc(matrix.num_classes * sizeof(int));
    for (i = 0; i < matrix.num_examples; i++) {
        for (j = 0; j < matrix.num_classes; j++)
            count_classifications[j] = 0;
        for (j = matrix.additional_cols; j < matrix.num_classifiers + matrix.additional_cols; j++)
            count_classifications[matrix.data[i][j]]++;
        most_votes = -1;
        num_ties = 0;
        for (j = 0; j < matrix.num_classes; j++) {
            if (count_classifications[j] > most_votes) {
                // This class is better than all the rest
                most_votes = count_classifications[j];
                num_ties = 0;
                tied_classes[num_ties++] = j;
            } else if (count_classifications[j] == most_votes) {
                // This class is as good as the current best
                tied_classes[num_ties++] = j;
            }
        }
        if (num_ties == 1 || args.break_ties_randomly == FALSE) {
            // Have one clear winner
            best_class = tied_classes[0];
        } else {
            // Randomly choose from among the best classes
            best_class = tied_classes[lrand48() % num_ties];
        }
        if (best_class < 0) {
            fprintf(stderr, "Something went horribly awry. No valid best class was found\n");
            exit(-1);
        }
        (*confusion_matrix)[best_class][matrix.data[i][0]]++;
    }
    free(count_classifications);
}
 *
 */

/*
   Prints the confusion matrix stored in confusion_matrix and frees confusion_matrix
 */

void print_confusion_matrix(int num_classes, int **confusion_matrix, char **class_names) {
    int i, j;
    int *max_width;
    char *format;
    
    max_width = (int *)malloc(num_classes * sizeof(int));
    for (i = 0; i < num_classes; i++)
        max_width[i] = strlen(class_names[i]);
    
    for (i = 0; i < num_classes; i++) {
        for (j = 0; j < num_classes; j++) {
            int digits = num_digits(confusion_matrix[i][j]);
            max_width[i] = digits > max_width[i] ? digits : max_width[i];
        }
    }
    
    int total_width = 0;
    for (i = 0; i < num_classes; i++)
        total_width += max_width[i];
    total_width += num_classes - 1;
    format = (char *)malloc(total_width * sizeof(char));
    sprintf(format, "%%%ds\n", total_width/2 + 2);
    printf(format, "TRUTH");
    
    for (i = 0; i < num_classes; i++) {
        sprintf(format, "%%%ds ", max_width[i]);
        printf(format, class_names[i]);
    }
    printf("\n");
    
    for (i = 0; i < num_classes; i++) {
        for (j = 0; j < max_width[i]; j++)
            printf("-");
        printf(" ");
    }
    printf("\n");
    
    for (i = 0; i < num_classes; i++) {
        for (j = 0; j < num_classes; j++) {
            sprintf(format, "%%%dd ", max_width[j]);
            printf(format, confusion_matrix[i][j]);
        }
        if (i == num_classes/2)
            printf("  %s     PREDICTIONS\n", class_names[i]);
        else
            printf("  %s\n", class_names[i]);
    }
    
    for (i = 0; i < num_classes; i++)
        free(confusion_matrix[i]);
    free(confusion_matrix);
    free(max_width);
    free(format);
}

void print_performance_metrics(CV_Subset data, CV_Prob_Matrix matrix, char ** class_names) {
    int i;
    int *max_width;
    char *format;
    
    max_width = (int *)malloc(data.meta.num_classes * sizeof(int));
    for (i = 0; i < data.meta.num_classes; i++)
        max_width[i] = strlen(class_names[i]);
    
    for (i = 0; i < data.meta.num_classes; i++) {
      max_width[i] = 9 > max_width[i] ? 9 : max_width[i];
    }
    int total_width = 0;
    for (i = 0; i < data.meta.num_classes; i++)
        total_width += max_width[i];
    total_width += data.meta.num_classes - 1;
    format = (char *)malloc(total_width * sizeof(char));


    printf("Performance Metrics:\n");
    //call loss functions here
    double * performance_metric;
    performance_metric = (double *)calloc(data.meta.num_classes + 2, sizeof(double));
    printf("%-13s"," ");

    for (i = 0; i < data.meta.num_classes; i++) {
        sprintf(format, "%%%ds ", max_width[i]);
        printf(format, class_names[i]);
    }
    sprintf(format, "%%%ds ", 9);
    printf(format, "Class Avg");
    sprintf(format, "%%%ds ", 7);
    printf(format, "Overall");
    printf("\n");

    calculate_accuracy(data, matrix, performance_metric);
    printf("%-13s", "WVAcc:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    sprintf(format, "%%%d.4f\n", 7);
    printf(format, performance_metric[data.meta.num_classes]);

    calculate_precision(data, matrix, performance_metric);
    printf("%-13s", "Precision:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    printf("-------\n");

    calculate_recall(data, matrix, performance_metric);
    printf("%-13s", "Recall:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    printf("-------\n");

    calculate_fmeasure(data, matrix, performance_metric);
    printf("%-13s", "F-measure:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    printf("-------\n");

    calculate_brier_score(data, matrix, performance_metric);
    printf("%-13s", "Brier Score:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    sprintf(format, "%%%d.4f\n", 7);
    printf(format, performance_metric[data.meta.num_classes]);

    calculate_nce(data, matrix, performance_metric);
    printf("%-13s", "NCE:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    sprintf(format, "%%%d.4f\n", 7);
    printf(format, performance_metric[data.meta.num_classes]);

    calculate_auroc(data, matrix, performance_metric);
    printf("%-13s", "AUROC:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    sprintf(format, "%%%d.4f\n", 7);
    printf(format, performance_metric[data.meta.num_classes]);

    calculate_calibration(data, matrix, performance_metric);
    printf("%-13s", "Calibration:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    printf("-------\n");

    calculate_refinement(data, matrix, performance_metric);
    printf("%-13s", "Refinement:");
    for (i = 0; i < data.meta.num_classes; i++) {
	 sprintf(format, "%%%d.4f ", max_width[i]);
	 printf(format, performance_metric[i]);
    }
    sprintf(format, "%%%d.4f ", 9);
    printf(format, performance_metric[data.meta.num_classes + 1]);
    printf("-------\n");

    free(performance_metric);
    free(max_width);
    free(format);
}

char* set_to_NULL() {
    return NULL;
}

//Added by DACIESL June-17-08: Performance
//All of these added from my own code and adapted into the avatar framework
//They are printed out at each fold under the VERBOSE output_accuracies mode

void calculate_accuracy(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double max;
    double *c;
    int i,j;
    int temp;
    int ties=0;

    c = (double *)calloc(data.meta.num_classes,sizeof(double));
    for (i =0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }
    for (i = 0; i < data.meta.num_classes ; i++) {
        c[i]=0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        max = matrix.data[j][0];
        temp=0;
        for (i = 1; i < data.meta.num_classes; i++) {
            if (matrix.data[j][i] > max) {
                max  = matrix.data[j][i];
                temp = i;
            }
        }
        for (i = 0; i < data.meta.num_classes ; i++) {
	    if (matrix.data[j][i] == max && i !=temp) 	    
                ties++;
	}
        if (temp == data.examples[j].containing_class_num) {
            L[data.examples[j].containing_class_num] += 1.0;
            L[data.meta.num_classes]                 += 1.0;
        }
        c[data.examples[j].containing_class_num] += 1.0;
  }
  L[data.meta.num_classes] = L[data.meta.num_classes]/(double)data.meta.num_examples;
  for (i = 0; i < data.meta.num_classes; i++) {
      if (c[i] != 0) 
          L[i]=L[i]/c[i];
      else 
          L[i]=0.0;
  }
  for (i = 0; i < data.meta.num_classes; i++) {
      L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
  }
  printf("#Ties: %d\n", ties);
  free(c);
}

void calculate_precision(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double max;
    double * c;
    int i,j;
    int temp;

    c = calloc(data.meta.num_classes, sizeof(double));
    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        max=0.0;
        temp=0;
        for (i = 0; i < data.meta.num_classes; i++) {
            if(matrix.data[j][i] > max) {
                max=matrix.data[j][i];
                temp=i;
            }
        }
        if(temp == data.examples[j].containing_class_num) {
            L[data.examples[j].containing_class_num] += 1.0;
        }
        c[temp] += 1.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        if (L[i]!=0)
            L[i]=L[i]/c[i];
        else
            L[i]=0.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(c);
}

void calculate_recall(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double max;
    double * c;
    int i,j;
    int temp;

    c = (double *)calloc(data.meta.num_classes, sizeof(double));
    for (i = 0 ; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        max=0.0;
        temp=0;
        for (i = 0; i < data.meta.num_classes; i++) {
            if(matrix.data[j][i] > max) {
                max = matrix.data[j][i];
                temp=i;
            }
        }
        if (temp == data.examples[j].containing_class_num) {
            L[data.examples[j].containing_class_num] += 1.0;
        }
        c[data.examples[j].containing_class_num] += 1.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        if(c[i] != 0)
            L[i] = L[i]/c[i];
        else 
            L[i]=0.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(c);
}

void calculate_fmeasure(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double max;
    double * c, * recall, * precision;
    int i,j;
    int temp;

    c = (double *)calloc(data.meta.num_classes, sizeof(double));
    recall = (double *)calloc(data.meta.num_classes, sizeof(double));
    precision = (double *)calloc(data.meta.num_classes, sizeof(double));
    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }

    for (j = 0; j < data.meta.num_examples; j++) {
        max=0.0;
        temp=0;
        for ( i = 0; i < data.meta.num_classes; i++) {
            if(matrix.data[j][i] > max) {
                max = matrix.data[j][i];
                temp=i;
            }
        }
        if ( temp == data.examples[j].containing_class_num) {
            recall[data.examples[j].containing_class_num] += 1.0;
        }
        c[data.examples[j].containing_class_num] += 1.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        if (c[i] != 0) 
            recall[i] = recall[i]/c[i];
        else 
            recall[i]=0.0;
    }

    for(i = 0; i < data.meta.num_classes; i++) {
        c[i] = 0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        max = 0.0;
        temp = 0;
        for(i = 0; i < data.meta.num_classes; i++) {
            if (matrix.data[j][i] > max) {
                max = matrix.data[j][i];
                temp=i;
            }
        }
        if (temp == data.examples[j].containing_class_num) {
            precision[data.examples[j].containing_class_num] += 1.0;
        }
        c[temp] += 1.0;
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        if(c[i] != 0) 
            precision[i]=precision[i]/c[i];
        else 
            precision[i]=0.0;
        if(recall[i]==0.0 || precision[i]==0.0)
            L[i] = 0.0;
        else
            L[i] = (2.0 * recall[i] * precision[i])/(recall[i] + precision[i]);
    }
    for (i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(c);
    free(recall);
    free(precision);
}

void calculate_brier_score(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double *c;
    int i,j;

    c = (double *)calloc(data.meta.num_classes, sizeof(double));
    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        for (i = 0; i < data.meta.num_classes; i++) {
            if(i == data.examples[j].containing_class_num) {
                L[data.examples[j].containing_class_num] += (1.0 - matrix.data[j][i]) * (1.0 - matrix.data[j][i]);
                L[data.meta.num_classes]                 += (1.0 - matrix.data[j][i]) * (1.0 - matrix.data[j][i]);
            } else {
                L[data.examples[j].containing_class_num] += matrix.data[j][i] * matrix.data[j][i];
                L[data.meta.num_classes]                 += matrix.data[j][i] * matrix.data[j][i];
            }
        }
        c[data.examples[j].containing_class_num] += 1.0;
    }
    L[data.meta.num_classes] = L[data.meta.num_classes]/(double)data.meta.num_examples;
    for(i = 0; i < data.meta.num_classes; i++) {
        if(c[i] != 0)
            L[i]=L[i]/c[i];
        else
            L[i]=0.0;
    }
    for(i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(c);
}

void calculate_nce(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double * c;
    int i,j;
    double val;

    c = (double *)calloc(data.meta.num_classes, sizeof(double));
    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        if (matrix.data[j][data.examples[j].containing_class_num] < 1e-6)
            val = 1e-6;
        else
	    val = matrix.data[j][data.examples[j].containing_class_num];
        L[data.examples[j].containing_class_num] -= dlog_2(val);
        L[data.meta.num_classes]                 -= dlog_2(val);
        c[data.examples[j].containing_class_num] += 1.0;
    }
    L[data.meta.num_classes] = L[data.meta.num_classes]/(double)data.meta.num_examples;
    for (i = 0; i < data.meta.num_classes; i++) {
        if(c[i] != 0)
            L[i] = L[i]/c[i];
        else
            L[i]=0.0;
    }
    for(i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(c);
}

void calculate_auroc(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    int i,j,k,ii,jj;
    int size0,size1;
    double * prop0, * prop1;
    double counter,a,b;

    //calculate 1 class vs. all others
    prop0 = (double *)calloc(data.meta.num_examples, sizeof(double));
    prop1 = (double *)calloc(data.meta.num_examples, sizeof(double));

    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }

    for (i = 0; i < data.meta.num_classes; i++) {
        size0 = 1;
        size1 = 1;
        counter = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            prop0[j] = 0.0;
            prop1[j] = 0.0;
        }

        for (j = 0; j < data.meta.num_examples; j++) {
            if (data.examples[j].containing_class_num == i)
                prop1[size1++] = matrix.data[j][i]; 
            else
	        prop0[size0++] = matrix.data[j][i]; 
        }

        for (ii = 1; ii <= size1; ii++)
            for (jj = 1; jj <= size0; jj++) {
                if (prop1[ii] > prop0[jj])
                    counter++; 
                if (prop1[ii] == prop0[jj])
                    counter=counter+0.5;
            }
        L[i] = counter/(size0*size1);
    }
    for(i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }

    //calculate each pairwise
    //A(i,j) = [A(i|j) + A(j|i)]/2  

    for (i = 1 ; i < data.meta.num_classes; i++) {
        for (k = 0; k < i; k++) {
            //(A(i|j)
            size0 = 1;
            size1 = 1;
            counter = 0;
            for(j = 0; j < data.meta.num_examples; j++) {
                prop0[j] = 0.0;
                prop1[j] = 0.0;
            }

            for (j = 0; j < data.meta.num_examples; j++) {
                if (data.examples[j].containing_class_num == i)
                    prop1[size1++] = matrix.data[j][i]; 
                if (data.examples[j].containing_class_num == k)
	            prop0[size0++] = matrix.data[j][i]; 
            }

            for (ii = 1; ii <= size1; ii++)
                for (jj = 1;jj <= size0; jj++) {
                    if (prop1[ii] > prop0[jj])
                        counter++; 
                    if (prop1[ii] == prop0[jj])
                        counter = counter+0.5;
                }

            a = counter/(size0*size1);

            size0 = 1;
            size1 = 1;
            counter = 0;
            for (j = 0; j < data.meta.num_examples; j++) {
                prop0[j] = 0.0;
                prop1[j] = 0.0;
            }

            for(j = 0; j < data.meta.num_examples; j++) {
                if (data.examples[j].containing_class_num == k)
		    prop1[size1++] = matrix.data[j][k]; 
                if (data.examples[j].containing_class_num == i)
	            prop0[size0++] = matrix.data[j][k]; 
            }

            for (ii = 1; ii <= size1; ii++)
                for (jj = 1; jj <= size0; jj++) {
                    if (prop1[ii] > prop0[jj])
                        counter++; 
                    if (prop1[ii] == prop0[jj])
                       counter=counter+0.5;
                }

            b = counter/(size0*size1);

            L[data.meta.num_classes] += (a+b)/2.0;
        }
    }
    L[data.meta.num_classes] = L[data.meta.num_classes] * 2.0 / (data.meta.num_classes * (data.meta.num_classes - 1.0));

    free(prop0);
    free(prop1);
}

void calculate_calibration(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double val=0.0,val1=0.0;
    int i,j,k,max;
    int total=0;

    struct sortstore * store;
    store = malloc(sizeof(struct sortstore)*data.meta.num_examples);

    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    }

    //do for each class first
    for (i = 0; i < data.meta.num_classes; i++) {
        for(j = 0; j < data.meta.num_examples; j++) {
            store[j].value = matrix.data[j][i];
            store[j].class = data.examples[j].containing_class_num;
        }

        //step 1, sort x
        qsort (store, data.meta.num_examples, sizeof(struct sortstore), compare_ref_cal);

        //step 2, do binning: find absolute value between rate of predicted vs. occurence
        if ( data.meta.num_examples < 100)
            max=50;
        else
            max=100;

        for (j = 0; j < data.meta.num_examples - max; j++) {
            val = 0.0;
            val1 = 0.0;
            total++;
            for ( k = 0; k < max; k++) {
	        val1 += store[j+k].value;
                if (store[j+k].class == i)
  	            val += 1.0;
            }
           val1 = 1.0 * val1 / max;
           val  = 1.0 * val / max;
           L[i] += fabs(val1 - val);
        }
        //step 3, average over all bins
        L[i] = L[i] / total;
    }

    for(i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(store);
}

void calculate_refinement(CV_Subset data, CV_Prob_Matrix matrix, double * L) {
    double val=0.0;
    int i,j,k,max;
    int total=0;

    struct sortstore * store;
    store = malloc(sizeof(struct sortstore)*data.meta.num_examples);

    for (i = 0; i < data.meta.num_classes + 2; i++) {
        L[i] = 0.0;
    } 

    for (i = 0; i < data.meta.num_classes; i++) {
        for(j = 0; j < data.meta.num_examples; j++) {
            store[j].value = matrix.data[j][i];
            store[j].class = data.examples[j].containing_class_num;
        }

        //step 1, sort x
        qsort (store, data.meta.num_examples, sizeof(struct sortstore), compare_ref_cal);

        //step 2, do binning: find absolute value between rate of predicted vs. occurence
        if(data.meta.num_examples < 100) 
            max = 50;
        else 
            max = 100;
        for (j = 0; j < data.meta.num_examples - max; j++) {
            val = 0.0;
            total++;
            for (k = 0; k < max; k++) 
                if (store[j+k].class == i) 
                    val += 1.0;
            val = 1.0 * val / max;
            L[i] += fabs(val * (1.0 - val));
        }
        //step 3, average over all bins
        L[i] =L[i] / total;
    }

    for(i = 0; i < data.meta.num_classes; i++) {
        L[data.meta.num_classes+1] += L[i]/(double)data.meta.num_classes;
    }
    free(store);
}

int compare_ref_cal (const void * a, const void * b) {
  if(  ((struct sortstore*) a)->value-((struct sortstore*) b)->value > 0) {
    return 1;
  }
  else if(  ((struct sortstore*) a)->value-((struct sortstore*) b)->value < 0) {
    return -1;
  }
  else {
    return 0;
  }
}
