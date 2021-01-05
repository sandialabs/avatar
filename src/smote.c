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
#include <math.h>
#include "crossval.h"
#include "knn.h"
#include "smote.h"
#include "rw_data.h"
#include "util.h"
#include "array.h"
#include "av_rng.h"

void smote(CV_Subset *data, CV_Subset *knn_src, AV_SortedBlobArray *blob, Args_Opts args) {
    int i, j, k, m;
    int num_examples;
    int *class_deficits;
    int total_deficit;
    Nearest_Neighbors **kNN;
    int running_count;
    float *new_c_vals; // Keep a running list of continuous values to add
    int *new_d_vals;   // Keep a running list of discrete values to add
    
    //printf("There are %d examples in the dataset\n", data->meta.num_examples);
    // Find the nearest neighbors
    int original_num_examples = knn_src->meta.num_examples;
    compute_knn(*knn_src, args.smote_knn, args.smote_Ln, args.smote_type, &kNN);
    //printf("\n");
    /* for (i = 0; i < data->meta.num_examples; i++) { */
    /*     printf("%2d: ", i); */
    /*     for (j = 0; j < args.smote_knn; j++) { */
    /*         if (kNN[i][j].neighbor < 0) */
    /*             printf("%2d/%7.4f,", kNN[i][j].neighbor, -1.0); */
    /*         else */
    /*             printf("%2d/%7.4f,", kNN[i][j].neighbor, kNN[i][j].distance); */
    /*     } */
    /*     printf("\b \n"); */
    /* } */
    
    compute_deficits(data->meta, args, &num_examples, &class_deficits);
    total_deficit = 0;
    int num_errs = 0;
    // Check that we are not reducing the population of any class and sum up total number of added examples
    for (i = 0; i < data->meta.num_classes; i++) {
        if (class_deficits[i] < 0) {
            fprintf(stderr, "ERROR: Class %d ('%s') will end up with fewer examples. I will not continue.\n",
                             i, data->meta.class_names[i]);
            fprintf(stderr, "  Details: Proportions would require %d but there are %d in the original dataset.\n",
                               data->meta.num_examples_per_class[i] + class_deficits[i], data->meta.num_examples_per_class[i]);
            num_errs++;
        }
        total_deficit += class_deficits[i];
    }
    if (num_errs > 0)
        exit(-1);
    data->examples = (CV_Example *)realloc(data->examples, num_examples * data->meta.num_attributes * sizeof(CV_Example));
    new_c_vals = (float *)malloc(total_deficit * data->meta.num_attributes * sizeof(float));
    new_d_vals = (int *)malloc(total_deficit * data->meta.num_attributes * sizeof(int));
    
    // Regenerate the tree for handling the attributes so that the new samples can be added
    init_att_handling(data->meta);
    // Since data->examples has been moved due to the realloc, the existing blob points nowhere so regenerate
    av_freeSortedBlobArray(blob);
    av_exitIfError(av_initSortedBlobArray(blob));
    for (i = 0; i < data->meta.num_examples; i++) {
        for (j = 0; j < data->meta.num_attributes; j++) {
            if (data->meta.attribute_types[j] == DISCRETE) {
                char *val;
                val = av_strdup(data->meta.discrete_attribute_map[j][data->examples[i].distinct_attribute_values[j]]);
                process_attribute_char_value(val, j, data->meta);
                free(val);
            } else if (data->meta.attribute_types[j] == CONTINUOUS) {
                process_attribute_float_value(data->float_data[j][data->examples[i].distinct_attribute_values[j]], j);
            }
        }
        av_addBlobToSortedBlobArray(blob, &data->examples[i], cv_example_compare_by_seq_id);
    }
    
    // If we have less than 2 examples in a class that gets SMOTEd, skip it
    for (i = 0; i < data->meta.num_classes; i++) {
        if (knn_src->meta.num_examples_per_class[i] < 2 && class_deficits[i] > 0) {
            class_deficits[i] = 0;
            fprintf(stderr, "\nWill not SMOTE class '%s'; it does not have enough examples\n",
                             knn_src->meta.class_names[i]);
        }
    }
    
    // SMOTE gets its own seed
    struct ParkMiller* rng = malloc(sizeof(struct ParkMiller));
    av_pm_default_init(rng, args.random_seed);

    /* gsl_rng *R; */
    /* R = gsl_rng_alloc(gsl_rng_ranlxs2); */
    /* gsl_rng_set(R, args.random_seed); */
    running_count = 0;
    int print_new_points = 0; // Use this to generate data for regression test
    // Add new, SMOTEd values to the tree
    for (i = 0; i < data->meta.num_classes; i++) {
        //printf("Class has %d examples needs %d more\n", data->meta.num_examples_per_class[i], class_deficits[i]);
        for (j = 0; j < class_deficits[i]; j++) {
            // Pick a random example from this class
            //int example_to_pick = gsl_rng_uniform_int(R, knn_src->meta.num_examples_per_class[i]);
            int example_to_pick = av_pm_uniform_int(rng, knn_src->meta.num_examples_per_class[i]);
            int class_count = 0;
            int example = -1;
            while (class_count < example_to_pick+1 && example < knn_src->meta.num_examples) {
                example++;
                if (knn_src->examples[example].containing_class_num == i)
                    class_count++;
            }
            // Pick a random nearest neighbor
            int neighbor = -1;
            while (neighbor < 0)
                //neighbor = kNN[example][gsl_rng_uniform_int(R, args.smote_knn)].neighbor;
                neighbor = kNN[example][av_pm_uniform_int(rng, args.smote_knn)].neighbor;
/*
Don't do this now. The YEAST dataset HAS valid duplicates
            // Make sure the neighbor and the original example are not the same point
            // This is possible for SMOTEBoost where both are picked from the boosted set
            int num_dup_atts = 0;
            for (k = 0; k < data->meta.num_attributes; k++)
                if (knn_src->examples[example].distinct_attribute_values[k] == knn_src->examples[neighbor].distinct_attribute_values[k])
                    num_dup_atts++;
            while (num_dup_atts == data->meta.num_attributes) {
                fprintf(stderr, "\nPicked a duplicate sample; repicking ...\n");
                neighbor = -1;
                while (neighbor < 0)
                    neighbor = kNN[example][gsl_rng_uniform_int(R, args.smote_knn)].neighbor;
                num_dup_atts = 0;
                for (k = 0; k < data->meta.num_attributes; k++)
                    if (knn_src->examples[example].distinct_attribute_values[k] == knn_src->examples[neighbor].distinct_attribute_values[k])
                        num_dup_atts++;
            }
 */
            // Pick a random fractional distance between the two
            //double fraction = gsl_rng_uniform(R);
            double fraction = av_pm_iterate(rng);
            if (print_new_points) {
                printf("ADDING new point %f between points %d (%s) and %d (%s) (", fraction,
                        example+1, knn_src->meta.class_names[knn_src->examples[example].containing_class_num],
                        neighbor+1, knn_src->meta.class_names[knn_src->examples[neighbor].containing_class_num]);
                for (k = 0; k < args.smote_knn; k++)
                    printf("%d,", kNN[example][k].neighbor + 1);
                printf("\b)\nADDING:");
            }
            
            // Compute attribute values for new point
            for (k = 0; k < data->meta.num_attributes; k++) {
                if (data->meta.attribute_types[k] == CONTINUOUS) {
                    float e = knn_src->float_data[k][knn_src->examples[example].distinct_attribute_values[k]];
                    float n = knn_src->float_data[k][knn_src->examples[neighbor].distinct_attribute_values[k]];
                    new_c_vals[running_count] = ((n - e) * fraction) + e;
                    if (print_new_points)
                        printf("%.16f,", new_c_vals[running_count]);
                    process_attribute_float_value(new_c_vals[running_count], k);
                    running_count++;
                } else if (data->meta.attribute_types[k] == DISCRETE) {
                    int *discrete;
                    discrete = (int *)calloc(data->meta.num_attributes, sizeof(int));
                    for (m = 0; m < args.smote_knn; m++)
                        if (kNN[example][m].neighbor >= 0)
                            discrete[knn_src->examples[kNN[example][m].neighbor].distinct_attribute_values[k]]++;
                    int max_val = 0;
                    int max_att = -1;
                    for (m = 0; m < data->meta.num_discrete_values[k]; m++) {
                        if (discrete[m] > max_val) {
                            max_val = discrete[m];
                            max_att = m;
                        }
                    }
                    if (max_att < 0)
                        fprintf(stderr, "Something is amiss in smote.c (max_att < 0)\n");
                    new_d_vals[running_count] = max_att;
                    if (print_new_points)
                        printf("%s,", knn_src->meta.discrete_attribute_map[k][new_d_vals[running_count]]);
                    running_count++;
                    free(discrete);
                }
            }
            if (print_new_points)
                printf("%s\n", knn_src->meta.class_names[data->examples[example].containing_class_num]);
        }
    }
    //gsl_rng_free(R);
    free(rng);
    
    // Need a copy of the original float_data in order to translate the original attribute values
    float **original_fd;
    original_fd = (float **)malloc(data->meta.num_attributes * sizeof(float *));
    for (i = 0; i < data->meta.num_attributes; i++) {
        if (data->meta.attribute_types[i] == CONTINUOUS) {
            original_fd[i] = (float *)malloc((data->high[i]+1) * sizeof(float));
            for (j = 0; j < data->high[i]+1; j ++)
                original_fd[i][j] = data->float_data[i][j];
        }
    }
    
    // DON'T FREE data->float_data
    // This was originally pointing to the same location as Full_Trainset.float_data
    // so freeing it would erase Full_Trainset.float_data.
    // Instead, in create_float_data() we malloc data->float_data to put it in a new memory location.
    
    // Now do the second step and actually add the data to the dataset
    create_float_data(data);
    
    // With new float_data, some of the original distinct_attribute_values will be wrong.
    // Recompute all of these before going on:
    for (i = 0; i < data->meta.num_examples; i++)
        for (k = 0; k < data->meta.num_attributes; k++)
            if (data->meta.attribute_types[k] == CONTINUOUS)
                add_attribute_float_value(original_fd[k][data->examples[i].distinct_attribute_values[k]], data, i, k);
    // Free original_fd
    for (i = 0; i < data->meta.num_attributes; i++)
        if (data->meta.attribute_types[i] == CONTINUOUS)
            free(original_fd[i]);
    free(original_fd);
    
    running_count = 0;
    AV_ReturnCode rc;
    // Add new, SMOTEd values to the tree
    for (i = 0; i < data->meta.num_classes; i++) {
        for (j = 0; j < class_deficits[i]; j++) {
            // Compute attribute values for new point
            for (k = 0; k < data->meta.num_attributes; k++) {
            // Initialize some values for this new example
                if (k == 0) {
                    data->examples[data->meta.num_examples].global_id_num = data->meta.num_examples;
                    data->examples[data->meta.num_examples].fclib_id_num = data->meta.num_examples;
                    data->examples[data->meta.num_examples].fclib_seq_num = 0;
                    data->examples[data->meta.num_examples].distinct_attribute_values =
                                                                (int *)malloc(data->meta.num_attributes * sizeof(int));
                    data->examples[data->meta.num_examples].containing_class_num = i;
                    
                    rc = av_addBlobToSortedBlobArray(blob, &data->examples[data->meta.num_examples],
                                                     cv_example_compare_by_seq_id);
                    if (rc < 0) {
                        av_exitIfErrorPrintf(rc, "Failed to add SMOTEd example %d to SBA\n", data->meta.num_examples);
                    } else if (rc == 0) {
                        fprintf(stderr, "Example %d already exists in SBA\n", data->meta.num_examples);
                    }
                }
                if (data->meta.attribute_types[k] == CONTINUOUS) {
                    //printf("%f ", new_c_vals[running_count]);
                    add_attribute_float_value(new_c_vals[running_count], data, data->meta.num_examples, k);
                    //printf("Done\n");
                    running_count++;
                } else if (data->meta.attribute_types[k] == DISCRETE) {
                    data->examples[data->meta.num_examples].distinct_attribute_values[k] = new_d_vals[running_count];
                    running_count++;
                }
            }
            //printf("-2\n");
            data->meta.num_examples++;
            if (data->meta.num_examples > num_examples) {
                fprintf(stderr, "Something is wrong. Have more than the expected number (%d) of examples.\n", num_examples);
                exit(-1);
            }
        }
        data->meta.num_examples_per_class[i] += class_deficits[i];
    }
    
    free(new_c_vals);
    free(new_d_vals);
    free(class_deficits);
    for (i = 0; i < original_num_examples; i++)
        free(kNN[i]);
    free(kNN);
}

void compute_deficits(CV_Metadata meta, Args_Opts args, int *new_N, int **deficits) {
    int i;
    
    int N = meta.num_examples;  // Total number of examples in original dataset
    int Nm = 0;     // Total number of examples in minority classes
    float Np = 0.0; // Total of minority class proportions
    for (i = 0; i < args.num_minority_classes; i++) {
        Nm += meta.num_examples_per_class[args.minority_classes[i]];
        Np += args.proportions[i];
    }
    
    float new_N_estimate = ((float)(N - Nm))/(1.0 - Np);
    
    (*deficits) = (int *)malloc(meta.num_classes * sizeof(int));
    *new_N = 0;
    int min_class_id = 0;
    for (i = 0; i < meta.num_classes; i++) {
      if (find_int(i, args.num_minority_classes, args.minority_classes)) {
            (*deficits)[i] = ceilf(args.proportions[min_class_id++] * new_N_estimate) - meta.num_examples_per_class[i];
            *new_N += meta.num_examples_per_class[i] + (*deficits)[i];
        } else {
            (*deficits)[i] = 0;
            *new_N += meta.num_examples_per_class[i];
        }
    }
    find_int_release();
}

void __print_smoted_data(int fold_num, CV_Subset data) {
    int i, k;
    for (i = 0; i < data.meta.num_examples; i++) {
        printf("FOLD %d:", fold_num+1);
        for (k = 0; k < data.meta.num_attributes; k++) {
            if (data.meta.attribute_types[k] == CONTINUOUS) {
                printf("%.16g,", data.float_data[k][data.examples[i].distinct_attribute_values[k]]);
            } else if (data.meta.attribute_types[k] == DISCRETE) {
                printf("%s,", data.meta.discrete_attribute_map[k][data.examples[i].distinct_attribute_values[k]]);
            }
        }
        printf("%s\n", data.meta.class_names[data.examples[i].containing_class_num]);
    }
}

