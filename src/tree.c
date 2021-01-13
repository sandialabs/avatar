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
#include "tree.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "gain.h"
#include "array.h"
#include "util.h"
#include "memory.h"
#include "safe_memory.h"
#include "subspaces.h"
#include "bagging.h"
#include "evaluate.h"
#include "rw_data.h"
#include "ivote.h"
#include "att_noising.h"
#include "balanced_learning.h"
#include "boost.h"
#include "skew.h"
#include "smote.h"
#include "options.h"
#include "heartbeat.h"
#include "reset.h"

/* Prototype declarations for internal module functions. */
void free_copied_CV_Subset(CV_Subset *sub);


void train(CV_Subset *data, DT_Ensemble *ensemble, int fold_num, Args_Opts args) {
    int i, num_trees;
    // For stopping algorithm
    int stop_building_at = 0;
    float best_oob_acc;
    Vote_Cache *cache;
    Vote_Cache **noisy_cache;
    // Just initialize to 1 here. We'll realloc later if we're using it
    cache = (Vote_Cache *)malloc(sizeof(Vote_Cache));
    noisy_cache = (Vote_Cache **)malloc(sizeof(Vote_Cache *));
    FILE *oob_file = NULL;
    
    // Compute OOB accuracy if using stopping algorithm or if bagging
    // Since stopping algorithm must be used with bagging or ivoting, then compute if bagging
    Boolean compute_oob_acc = args.do_bagging;
    
    ensemble->num_trees = 10;
    if (args.num_trees > 0) {
        ensemble->num_trees = args.num_trees;
    } else if (args.auto_stop == TRUE) {
        ensemble->num_trees = 100;      // Start with 100 for the stopping algorithm
        args.num_trees = -1;            // Don't need this
    }
    ensemble->num_classes = data->meta.num_classes;
    ensemble->num_attributes = data->meta.num_attributes;
    //Cosmin added if statement (09/01/2020)
    if (ensemble->attribute_types != NULL) {
        free(ensemble->attribute_types);
        ensemble->attribute_types = NULL;
    }
    ensemble->attribute_types = (Attribute_Type*) malloc(ensemble->num_attributes * sizeof(Attribute_Type));
    memcpy(ensemble->attribute_types, data->meta.attribute_types, ensemble->num_attributes * sizeof(Attribute_Type));
    //Cosmin added if statement (09/01/2020)
    if (ensemble->Trees != NULL) {
        free(ensemble->Trees);
        ensemble->Trees = NULL;
    }
    ensemble->Trees = (DT_Node **)malloc(ensemble->num_trees * sizeof(DT_Node *));
    ensemble->Books = (Tree_Bookkeeping *)malloc(ensemble->num_trees * sizeof(Tree_Bookkeeping));
    ensemble->weights = (float *)malloc(ensemble->num_trees * sizeof(float));
    int num_boosting_betas = ensemble->num_trees;
    ensemble->boosting_betas = (double *)malloc(num_boosting_betas * sizeof(double));
    
    CV_Subset *data_rs, *data_bag, *data_skw;
    data_skw = (CV_Subset *)calloc(1,sizeof(CV_Subset));
    data_bag = (CV_Subset *)calloc(1,sizeof(CV_Subset));
    data_rs  = (CV_Subset *)calloc(1,sizeof(CV_Subset));

    // data_skw = (CV_Subset *)malloc(sizeof(CV_Subset));
    // data_bag = (CV_Subset *)malloc(sizeof(CV_Subset));
    // data_rs  = (CV_Subset *)malloc(sizeof(CV_Subset));
    // //Cosmin added reset statements
    // reset_CV_Subset(data_rs);
    // reset_CV_Subset(data_bag);
    // reset_CV_Subset(data_skw);

    // Initialize the oob-data file for --verbose-oob
    if (args.output_verbose_oob)
        write_tree_file_header(args.num_trees, data->meta, fold_num, args.oob_file, args);

    // If we're bagging, initialize the Vote_Cache so we can report OOB accuracy
    if (compute_oob_acc == TRUE) {
        cache = (Vote_Cache *)realloc(cache, sizeof(Vote_Cache));
        cache->num_classes = data->meta.num_classes;
        cache->num_train_examples = data->meta.num_examples;
        cache->num_classifiers = 0;
        cache->oob_error = 0.0;
        cache->average_train_accuracy = 0.0;
        cache->best_train_class = (int *)malloc(cache->num_train_examples * sizeof(int));
        for (i = 0; i < cache->num_train_examples; i++)
            cache->best_train_class[i] = -1;
        cache->oob_class_votes = (int **)malloc(cache->num_train_examples * sizeof(int *));
        for (i = 0; i < cache->num_train_examples; i++)
            cache->oob_class_votes[i] = (int *)calloc(cache->num_classes, sizeof(int));
        cache->oob_class_weighted_votes = (float **)malloc(cache->num_train_examples * sizeof(float *));
        for (i = 0; i < cache->num_train_examples; i++)
            cache->oob_class_weighted_votes[i] = (float *)calloc(cache->num_classes, sizeof(float));
    }
    if (args.do_noising == TRUE) {
        noisy_cache = (Vote_Cache **)realloc(noisy_cache, data->meta.num_attributes * sizeof(Vote_Cache *));
        for (i = 0; i < data->meta.num_attributes; i++) {
            noisy_cache[i] = (Vote_Cache *)malloc(sizeof(Vote_Cache));
            initialize_cache(noisy_cache[i], data->meta.num_classes, data->meta.num_examples, 0, 0);
/*            noisy_cache[i]->num_classes = data->meta.num_classes;
            noisy_cache[i]->num_train_examples = data->meta.num_examples;
            noisy_cache[i]->num_classifiers = 0;
            noisy_cache[i]->oob_error = 0.0;
            noisy_cache[i]->average_train_accuracy = 0.0;
            noisy_cache[i]->best_train_class = (int *)malloc(noisy_cache[i]->num_train_examples * sizeof(int));
            int j;
            for (j = 0; j < noisy_cache[i]->num_train_examples; j++)
                noisy_cache[i]->best_train_class[j] = -1;
            noisy_cache[i]->oob_class_votes = (int **)malloc(noisy_cache[i]->num_train_examples * sizeof(int *));
            for (j = 0; j < noisy_cache[i]->num_train_examples; j++)
                noisy_cache[i]->oob_class_votes[j] = (int *)calloc(noisy_cache[i]->num_classes, sizeof(int));*/
        }
    }
    
    // If we're boosting, set weights to flat and initialize the RNG
    if (args.do_boosting == TRUE) {
        data->weights = (double *)malloc(data->meta.num_examples * sizeof(double));
        for (i = 0; i < data->meta.num_examples; i++)
            data->weights[i] = 1.0/data->meta.num_examples;
        init_weighted_rng(data->meta.num_examples, data->weights, args);
    }
    
    begin_progress_counters(1);
    num_trees = 0;
    while ((args.num_trees > 0 && num_trees < args.num_trees) || (args.auto_stop == TRUE && stop_building_at == 0)) {
        
        if (compute_oob_acc == TRUE)
            cache->current_classifier_count = num_trees + 1;
        if (args.do_noising == TRUE)
            for (i = 0; i < data->meta.num_attributes; i++)
                noisy_cache[i]->current_classifier_count = num_trees + 1;
        
        if (args.do_balanced_learning == TRUE) {
            get_next_balanced_set(num_trees, data, data_skw, args);
        } else if (args.do_boosting == TRUE) {
            get_boosted_set(data, data_skw);
            //printf("Boosted data for tree %d has %d samples\n", num_trees, data_skw->meta.num_examples);
            //char label[128];
            //sprintf(label, "B-TREE:%d", num_trees);
            //__print_datafile(*data_skw, label);
            if (args.do_smoteboost == TRUE) {
                update_actual_att_props(num_trees==0?1:0, data_skw->meta, &args);
                // Just need to initialize; it gets freed and regenerated in smote()
                AV_SortedBlobArray SmoteExamples;
                av_exitIfError(av_initSortedBlobArray(&SmoteExamples));
                if (args.smoteboost_type == ALL_MINORITY_CLASSES)
                    smote(data_skw, data, &SmoteExamples, args);
                else if (args.smoteboost_type == FROM_BOOSTED_SET_ONLY)
                    smote(data_skw, data_skw, &SmoteExamples, args);
                av_freeSortedBlobArray(&SmoteExamples);
            }
        } else {
            data_skw = data;
        }
        //printf("Smoted data for tree %d has %d samples\n", num_trees, data_skw->meta.num_examples);
        //char label[128];
        //sprintf(label, "SB-TREE:%d", num_trees);
        //__print_datafile(*data_skw, label);
        
        if (args.do_bagging == TRUE) {
          make_bag(data_skw, data_bag, args, 0);
            // Need to copy the in_bag array from data_skw back to data for later
            for (i = 0; i < data->meta.num_examples; i++)
                data->examples[i].in_bag = data_skw->examples[i].in_bag;
        } else {
            data_bag = data_skw;
        }
        
        if (args.random_subspaces > 0)
            apply_random_subspaces(*data_bag, data_rs, args);
        else {
            data_rs = data_bag;
        }
        
        // Check mallocs
        if (num_trees + 1 > ensemble->num_trees) {
            ensemble->num_trees *= 2;
            ensemble->Trees = (DT_Node **)realloc(ensemble->Trees, ensemble->num_trees * sizeof(DT_Node *));
            ensemble->Books = (Tree_Bookkeeping *)realloc(ensemble->Books, ensemble->num_trees * sizeof(Tree_Bookkeeping));
            ensemble->weights = (float *)realloc(ensemble->weights, ensemble->num_trees * sizeof(float));
        }

        // Initialize the array of DT_Nodes for this ensemble and some other values
        ensemble->Books[num_trees].num_malloced_nodes = 1;
        ensemble->Books[num_trees].next_unused_node = 1;
        ensemble->Books[num_trees].current_node = 0;
        ensemble->Trees[num_trees] =
                                (DT_Node *)calloc(ensemble->Books[num_trees].num_malloced_nodes, sizeof(DT_Node));
        build_tree(data_rs, &ensemble->Trees[num_trees], &ensemble->Books[num_trees], args);
        if (compute_oob_acc == TRUE) {
            cache->oob_error = compute_oob_error_rate(ensemble->Trees[num_trees], *data, cache, args);
            if (args.do_noising == TRUE)
                compute_noised_oob_error_rate(ensemble->Trees[num_trees], *data, noisy_cache, args);
            // Check stopping algoritm
            char *mod_oob_file = NULL;
            if (args.output_verbose_oob)
                mod_oob_file = build_output_filename(fold_num, args.oob_file, args);
            stop_building_at = check_stopping_algorithm(0, 0, 1.0 - cache->oob_error, num_trees + 1,
                                                        &best_oob_acc, mod_oob_file, args);
            //if (args.output_verbose_oob) {
            //    char *mod_oob_file = build_output_filename(fold_num, args.oob_file, args);
            //    if ((oob_file = fopen(mod_oob_file, "a")) == NULL) {
            //        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n", mod_oob_file);
            //        exit(8);
            //    }
            //    fprintf(oob_file, "%d %6g\n", num_trees+1, 1.0-cache->oob_error);
            //}
            //printf("%d trees: voted/avg = %6g/%6g\n", num_trees+1, 1-cache->oob_error, cache->average_train_accuracy);
            if (args.auto_stop == TRUE && stop_building_at > 0) {
                ensemble->num_trees = stop_building_at;
                //printf("Stopping with %d trees\n", ensemble->num_trees);
                if (args.output_verbose_oob) {
                    if ((oob_file = fopen(mod_oob_file, "a")) == NULL) {
                        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n", mod_oob_file);
                        exit(8);
                    }
                    fprintf(oob_file, "#Stopping with %d trees\n", ensemble->num_trees);
                    fclose(oob_file);
                }
            }
        }
        
        // For boosting, update the weights and reset the RNG for the new weights
        if (args.do_boosting == TRUE) {
            while (num_boosting_betas < num_trees) {
                num_boosting_betas *= 2;
                ensemble->boosting_betas = (double *)realloc(ensemble->boosting_betas, num_boosting_betas * sizeof(double));
            }
            
            ensemble->boosting_betas[num_trees] = update_weights(&data->weights, ensemble->Trees[num_trees], *data);
            reset_weights(data->meta.num_examples, data->weights);
        }
        
        // Increment number of trees in ensemble
        num_trees++;
        
        if (args.debug)
            printf("\n    ");
        update_progress_counters(1, &num_trees);
        
        if (args.do_bagging == TRUE) {
            for (i = 0; i < (int)(args.bag_size * (float)data_bag->meta.num_examples / 100.0); i++)
                free(data_bag->examples[i].distinct_attribute_values);
            free_CV_Subset_inter(data_bag, args, TRAIN_MODE);
        }
        if (args.random_subspaces > 0)
            free_CV_Subset_inter(data_rs, args, TRAIN_MODE);
        if (args.do_balanced_learning || args.do_boosting)
            free_CV_Subset_inter(data_skw, args, TRAIN_MODE);
        //printf("%d %d %d %d\n", args.num_trees, num_trees, args.auto_stop, stop_building_at);
    }
    end_progress_counters();
    if (args.auto_stop == TRUE) {
        printf("Stopping Algorithm Result: %d trees with an oob accuracy of %.4f%%\n",
               ensemble->num_trees, best_oob_acc * 100.0);
    } else if (compute_oob_acc == TRUE) {
        printf("oob accuracy = %.4f%%\n", (1.0 - cache->oob_error) * 100.0);
    }
    if (args.do_noising == TRUE) {
        Boolean strip_prefix = att_label_has_leading_number(data->meta.attribute_names, data->meta.num_attributes, args);
        // Compute the length of the longest attribute name
        unsigned int max_att_name_length = 0;
        int used_atts = -1;
        char format[2048];
        for (i = 0; i < data->meta.num_attributes + args.num_skipped_features; i++) {
            if (args.truth_column > 0 && i+1 >= args.truth_column &&
                find_int(i+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column < 0 &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            }
            used_atts++;
            if (strip_prefix == TRUE) {
                char *label;
                strtol(data->meta.attribute_names[used_atts], &label, 10);
                if (strlen(label) > max_att_name_length)
                    max_att_name_length = strlen(label);
            } else {
                if (strlen(data->meta.attribute_names[used_atts]) > max_att_name_length)
                    max_att_name_length = strlen(data->meta.attribute_names[used_atts]);
            }
        }
        // Write the format string using the just-computed length
        sprintf(format, "%%%% increase in error for att %%3d, %%%ds = %%12.4f%%%% [%%8.4f%%%% -> %%8.4f%%%%]\n", max_att_name_length);
        used_atts = -1;
        for (i = 0; i < data->meta.num_attributes + args.num_skipped_features; i++) {
            if (args.truth_column > 0 && i+1 >= args.truth_column &&
                find_int(i+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column < 0 &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            }
            used_atts++;
            if (strip_prefix == TRUE) {
                char *label;
                strtol(data->meta.attribute_names[used_atts], &label, 10);
                printf(format, (i+1>=args.truth_column?i+2:i+1), label,
                       100.0 * (noisy_cache[used_atts]->oob_error - cache->oob_error) / cache->oob_error,
                       cache->oob_error * 100.0, noisy_cache[used_atts]->oob_error * 100.0);
            } else {
                printf(format, (i+1>=args.truth_column?i+2:i+1), data->meta.attribute_names[used_atts],
                       100.0 * (noisy_cache[used_atts]->oob_error - cache->oob_error) / cache->oob_error,
                       cache->oob_error * 100.0, noisy_cache[used_atts]->oob_error * 100.0);
            }
        }
    }
    //Cosmin added free statements
    free(cache);
    free(noisy_cache);
    // free_CV_Subset(data_skw,args,TRAIN_MODE);
    // free_CV_Subset(data_bag,args,TRAIN_MODE);
    // free_CV_Subset(data_rs, args,TRAIN_MODE);

    find_int_release();
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_tree & save_ensembles calls
void train_ivote(CV_Subset train_data, CV_Subset test_data, int fold_num, Vote_Cache *cache, Args_Opts args) {
    int i, j, num_trees;
    // For stopping algorithm
    int stop_building_at = 0;
    float best_oob_acc;
    Vote_Cache **noisy_cache;
    // Just initialize to 1 here. We'll realloc later if we're using it
    noisy_cache = (Vote_Cache **)malloc(sizeof(Vote_Cache *));
    FILE *oob_file = NULL;
    int num_boosting_betas;
    double *boosting_betas;
    
    // Compute OOB accuracy if using stopping algorithm or if bagging
    // Since stopping algorithm must be used with bagging or ivoting, then compute if bagging
    Boolean compute_oob_acc = TRUE;
    float unweighted_oob_error = 0.0;
    
    CV_Subset *data_skw, data_bite, *data_rs;
    data_skw = (CV_Subset *)malloc(sizeof(CV_Subset));
    data_rs  = (CV_Subset *)malloc(sizeof(CV_Subset));
    reset_CV_Subset(&data_bite);

    Tree_Bookkeeping Books;
    DT_Node *Tree;
    static int count = 0;
    count++;
    
    initialize_cache(cache, train_data.meta.num_classes, train_data.meta.num_examples, test_data.meta.num_examples, 0);
    
    // Initialize the oob-data file for --verbose-oob
    if (args.output_verbose_oob) {
        write_tree_file_header(args.num_trees, train_data.meta, fold_num, args.oob_file, args);
    }
    num_boosting_betas = (args.auto_stop==TRUE ? 100 : args.num_trees);
    boosting_betas = (double *)malloc(num_boosting_betas * sizeof(double));
    
    if (args.do_noising == TRUE) {
        noisy_cache = (Vote_Cache **)realloc(noisy_cache, train_data.meta.num_attributes * sizeof(Vote_Cache *));
        for (i = 0; i < train_data.meta.num_attributes; i++) {
            noisy_cache[i] = (Vote_Cache *)malloc(sizeof(Vote_Cache));
            noisy_cache[i]->num_classes = train_data.meta.num_classes;
            noisy_cache[i]->num_train_examples = train_data.meta.num_examples;
            noisy_cache[i]->num_classifiers = 0;
            noisy_cache[i]->oob_error = 0.0;
            noisy_cache[i]->average_train_accuracy = 0.0;
            noisy_cache[i]->best_train_class = (int *)malloc(noisy_cache[i]->num_train_examples * sizeof(int));
            for (j = 0; j < noisy_cache[i]->num_train_examples; j++)
                noisy_cache[i]->best_train_class[j] = -1;
            noisy_cache[i]->oob_class_votes = (int **)malloc(noisy_cache[i]->num_train_examples * sizeof(int *));
            for (j = 0; j < noisy_cache[i]->num_train_examples; j++)
                noisy_cache[i]->oob_class_votes[j] = (int *)calloc(noisy_cache[i]->num_classes, sizeof(int));

            noisy_cache[i]->oob_class_votes = (int **)malloc(noisy_cache[i]->num_train_examples * sizeof(int *));
            for (j = 0; j < noisy_cache[i]->num_train_examples; j++)
	        noisy_cache[i]->oob_class_weighted_votes[j] = (float *)calloc(noisy_cache[i]->num_classes, sizeof(float));
        }
    }
    
    // If we're boosting, set weights to flat and initialize the RNG
    if (args.do_boosting == TRUE) {
        train_data.weights = (double *)malloc(train_data.meta.num_examples * sizeof(double));
        for (i = 0; i < train_data.meta.num_examples; i++)
            train_data.weights[i] = 1.0/train_data.meta.num_examples;
        init_weighted_rng(train_data.meta.num_examples, train_data.weights, args);
    }
    
    if (args.save_trees ||
        (args.auto_stop && (args.output_probabilities == TRUE || args.output_margins == TRUE)))
        write_tree_file_header(args.num_trees, train_data.meta, fold_num, args.trees_file, args);
    
    begin_progress_counters(1);
    num_trees = 0;
    while ( (args.num_trees > 0 && num_trees < args.num_trees) ||
            (args.auto_stop == TRUE && stop_building_at == 0) ) {
        cache->current_classifier_count = num_trees + 1;
        if (args.do_noising == TRUE)
            for (i = 0; i < train_data.meta.num_attributes; i++)
                noisy_cache[i]->current_classifier_count = num_trees + 1;
        
       if (args.do_balanced_learning == TRUE) {
            get_next_balanced_set(num_trees, &train_data, data_skw, args);
        } else if (args.do_boosting == TRUE) {
            get_boosted_set(&train_data, data_skw);
        } else {
            data_skw = &train_data;
        }
        
        make_bite(data_skw, &data_bite, cache, args);
        // Need to copy the in_bag array from data_skw back to train_data for later
        for (i = 0; i < train_data.meta.num_examples; i++)
            train_data.examples[i].in_bag = data_skw->examples[i].in_bag;
        
        //printf("data_bite.high(b) is at memory location 0x%lx\n", &data_bite.high);
        if (args.random_subspaces > 0) {
            apply_random_subspaces(data_bite, data_rs, args);
            //printf("data_bite.high(m) is at memory location 0x%lx\n", &data_bite.high);
            //printf("data_rs.high(m)   is at memory location 0x%lx\n", &(data_rs->high));
        } else {
            data_rs = &data_bite;
        }

        // Initialize the array of DT_Nodes for this ensemble and some other values
        if (num_trees > 0) {
            // After the first time through, need to free up the Tree and start all over
            free_DT_Node(Tree, Books.next_unused_node);
        }
        Books.num_malloced_nodes = 1;
        Books.next_unused_node = 1;
        Books.current_node = 0;
        Tree = (DT_Node *)malloc(Books.num_malloced_nodes * sizeof(DT_Node));
        
        build_tree(&data_bite, &Tree, &Books, args);
        if (args.save_trees ||
           (args.auto_stop && (args.output_probabilities == TRUE || args.output_margins == TRUE))) {
            save_tree(Tree, fold_num, num_trees+1, args, train_data.meta.num_classes);
        }
        
        unweighted_oob_error = compute_oob_error_rate(Tree, train_data, cache, args);
        if (args.do_noising == TRUE)
            compute_noised_oob_error_rate(Tree, train_data, noisy_cache, args);
        if (compute_oob_acc == TRUE) {
            // Check stopping algoritm
            char *mod_oob_file = NULL;
            if (args.output_verbose_oob)
                mod_oob_file = build_output_filename(fold_num, args.oob_file, args);
            stop_building_at = check_stopping_algorithm(0, 0, 1.0 - unweighted_oob_error, num_trees+1,
                                                        &best_oob_acc, mod_oob_file, args);
            //if (args.output_verbose_oob) {
            //    char *mod_oob_file = build_output_filename(fold_num, args.oob_file, args);
            //    if ((oob_file = fopen(mod_oob_file, "a")) == NULL) {
            //        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n", mod_oob_file);
            //        exit(8);
            //    }
            //    fprintf(oob_file, "%d %6g\n", num_trees+1, 1-unweighted_oob_error);
            //}
            if (args.auto_stop == TRUE && stop_building_at > 0) {
                cache->num_classifiers = stop_building_at;
                //printf("Stopping with %d trees\n", ensemble->num_trees);
                if (args.output_verbose_oob) {
                    if ((oob_file = fopen(mod_oob_file, "a")) == NULL) {
                        fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n", mod_oob_file);
                        exit(8);
                    }
                    fprintf(oob_file, "#Stopping with %d trees\n", cache->num_classifiers);
                    fclose(oob_file);
                }
            }
        }
        cache->oob_error = args.ivote_p_factor * cache->oob_error + (1.0 - args.ivote_p_factor) * unweighted_oob_error;
        cache->test_error = compute_test_error_rate(Tree, test_data, cache, args);
        
        // For boosting, update the weights and reset the RNG for the new weights
        if (args.do_boosting == TRUE) {
            while (num_boosting_betas < num_trees) {
                num_boosting_betas *= 2;
                boosting_betas = (double *)realloc(boosting_betas, num_boosting_betas * sizeof(double));
            }
            
            boosting_betas[num_trees] = update_weights(&train_data.weights, Tree, train_data);
            reset_weights(train_data.meta.num_examples, train_data.weights);
        }
        
        // Increment number of trees in ensemble
        num_trees++;
        
        if (args.debug)
            printf("\n    ");
        update_progress_counters(1, &num_trees);
        
        if (args.random_subspaces > 0)
            free_CV_Subset_inter(data_rs, args, TRAIN_MODE);

        if (args.do_balanced_learning == TRUE || args.do_boosting == TRUE)
            free_CV_Subset_inter(data_skw, args, TRAIN_MODE);
        
        for (i = 0; i < args.bite_size; i++)
            free(data_bite.examples[i].distinct_attribute_values);
        free_CV_Subset_inter(&data_bite, args, TRAIN_MODE);
        //free(data_bite.high);
        
    }
    free_DT_Node(Tree, Books.next_unused_node);

    end_progress_counters();

    if (compute_oob_acc == TRUE) {
        if (args.auto_stop == TRUE) {
            printf("Stopping Algorithm Result: %d trees with an oob accuracy of %.4f%%\n",
                   stop_building_at, best_oob_acc * 100.0);
            // Read the ensemble if we're saving trees or printing probabilities after using the stopping algorithm
            if (args.save_trees ||
                (args.auto_stop && (args.output_probabilities == TRUE || args.output_margins == TRUE))) {
                DT_Ensemble ensemble;
                reset_DT_Ensemble(&ensemble);

                read_ensemble(&ensemble, fold_num, stop_building_at, &args);
                // Rewrite ensemble file to get NumTrees right and eliminate extra trees
                if (args.save_trees) {
  		    save_ensemble(ensemble, train_data.meta, fold_num, args, test_data.meta.num_classes);
                }
                // Recompute Vote_Cache so it reflects the optimal number of trees
                if (args.auto_stop) {
                    // Re-initialize
                    initialize_cache(cache, test_data.meta.num_classes, train_data.meta.num_examples, test_data.meta.num_examples, 1);
                    for (i = 0; i < ensemble.num_trees; i++) {
                        cache->current_classifier_count = i + 1;
                        compute_test_error_rate(ensemble.Trees[i], test_data, cache, args);
                    }
                    cache->num_classifiers = ensemble.num_trees;
                }
            }
        } else {
            printf("oob accuracy = %.4f%%\n", (1.0 - unweighted_oob_error) * 100.0);
        }
    }
    if (args.do_noising == TRUE) {
        Boolean strip_prefix = att_label_has_leading_number(train_data.meta.attribute_names,
                                                            train_data.meta.num_attributes, args);
        // Compute the length of the longest attribute name
        unsigned int max_att_name_length = 0;
        int used_atts = -1;
        char format[2048];
        for (i = 0; i < train_data.meta.num_attributes + args.num_skipped_features; i++) {
            if (args.truth_column > 0 && i+1 >= args.truth_column &&
                find_int(i+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column < 0 &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            }
            used_atts++;
            if (strip_prefix == TRUE) {
                char *label;
                strtol(train_data.meta.attribute_names[used_atts], &label, 10);
                if (strlen(label) > max_att_name_length)
                    max_att_name_length = strlen(label);
            } else {
                if (strlen(train_data.meta.attribute_names[used_atts]) > max_att_name_length)
                    max_att_name_length = strlen(train_data.meta.attribute_names[used_atts]);
            }
        }
        // Write the format string using the just-computed length
        sprintf(format, "%%%% increase in error for att %%3d, %%%ds = %%12.4f%%%% [%%8.4f%%%% -> %%8.4f%%%%]\n", max_att_name_length);
        
        used_atts = -1;
        for (i = 0; i < train_data.meta.num_attributes + args.num_skipped_features; i++) {
            if (args.truth_column > 0 && i+1 >= args.truth_column &&
                find_int(i+2, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is in a column past the truth column so look for the
                // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
                // of features to skip since the list is based on column number and not attribute number
                //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The current attribute is before the truth column so attribute number corresponds
                // to column number.
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            } else if (args.truth_column < 0 &&
                       find_int(i+1, args.num_skipped_features, args.skipped_features)) {
                // The truth column is last so all attributes are before the truth column
                //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
                continue;
            }
            used_atts++;
            if (strip_prefix == TRUE) {
                char *label;
                strtol(train_data.meta.attribute_names[used_atts], &label, 10);
                printf(format, (i+1>=args.truth_column?i+2:i+1), label,
                       100.0 * (noisy_cache[used_atts]->oob_error - cache->oob_error) / cache->oob_error,
                       cache->oob_error * 100.0, noisy_cache[used_atts]->oob_error * 100.0);
            } else {
                printf(format, (i+1>=args.truth_column?i+2:i+1), train_data.meta.attribute_names[used_atts],
                       100.0 * (noisy_cache[used_atts]->oob_error - cache->oob_error) / cache->oob_error,
                       cache->oob_error * 100.0, noisy_cache[used_atts]->oob_error * 100.0);
            }
        }
    }
    find_int_release();
}

//Modified by DACIESL June-05-08: Laplacean Estimates
//added consideration for cases of Laplacean Estimate output
void test(CV_Subset test_data, int num_ensembles, DT_Ensemble *ensemble, FC_Dataset dataset, int fold_num, Args_Opts args) {
    int i, j, k;
    if (args.output_accuracies == ON || args.output_accuracies == VERBOSE ||args.output_predictions || args.output_laplacean || args.output_confusion_matrix) 
    {
        if (num_ensembles == 1 || args.do_mass_majority_vote) {
            // Plain ol' voting on a single ensemble
            // or
            // Combine all trees into one ensemble and then do plain ol' voting on a single ensemble
            
            CV_Matrix Matrix;
            CV_Matrix Boost_Matrix;
            CV_Prob_Matrix Prob_Matrix;
            int **Confusion;
            static int **Overall_Confusion;
            static int **Overall_Confusion_5x2;
            static int diagonal_sum = 0;
            static int total_sum = 0;
            static int overall_diagonal_sum = 0;
            static int overall_total_sum = 0;

            // First time through with crossvalfc, initialize Overall_Confusion
            if (args.caller == CROSSVALFC_CALLER && ( (args.do_nfold_cv == TRUE && fold_num == 0) ||
                                                      (args.do_5x2_cv == TRUE && fold_num % 2 == 0) )) {
                // If this is 5x2 CV, and fold_num > 0, we need to free the old one before allocating
                //if (args.do_5x2_cv == TRUE && fold_num > 0) {
                //    for (i = 0; i < test_data.meta.num_classes; i++)
                //        free(Overall_Confusion[i]);
                //    free(Overall_Confusion);
                //}
                
                Overall_Confusion = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
                for (i = 0; i < test_data.meta.num_classes; i++)
                    Overall_Confusion[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
                    
                if (fold_num == 0 && args.do_5x2_cv == TRUE) {
                    Overall_Confusion_5x2 = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
                    for (i = 0; i < test_data.meta.num_classes; i++)
                        Overall_Confusion_5x2[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
                }
                
                diagonal_sum = 0;
                total_sum = 0;
            }
            
            if (num_ensembles == 1) {
                test_data.meta.Missing = ensemble[0].Missing;
                if (args.do_boosting == TRUE) {
                    build_boost_prediction_matrix(test_data, ensemble[0], &Boost_Matrix);
                    if (args.output_laplacean)
                        build_boost_probability_matrix(test_data, ensemble[0], &Prob_Matrix);
                    else
  		        Prob_Matrix.num_classes = 0;
	        } else {
		    build_prediction_matrix(test_data, ensemble[0], &Matrix);
                    if (args.output_laplacean)
                        build_probability_matrix(test_data, ensemble[0], &Prob_Matrix);
                    else
  		        Prob_Matrix.num_classes = 0;
	        }
            } else if (num_ensembles > 1) {
                DT_Ensemble big_ensemble;
                concat_ensembles(num_ensembles, ensemble, &big_ensemble);
                //save_ensemble(big_ensemble, test_data.meta, -1, args);
                test_data.meta.Missing = big_ensemble.Missing;
                if (args.do_boosting == TRUE) {
                    build_boost_prediction_matrix(test_data, big_ensemble, &Boost_Matrix);
                    if (args.output_laplacean)
                        build_boost_probability_matrix(test_data, big_ensemble, &Prob_Matrix);
                    else
  		        Prob_Matrix.num_classes = 0;
                } else {
                    build_prediction_matrix(test_data, big_ensemble, &Matrix);
                    if (args.output_laplacean)
                        build_probability_matrix(test_data, big_ensemble, &Prob_Matrix);
                    else
  		        Prob_Matrix.num_classes = 0;
		}
            }
            
            if (args.output_accuracies == ON || args.output_accuracies == VERBOSE) {
                if (args.do_boosting == TRUE) {
                    printf("Boosting Accuracy = %.4f%%\n", compute_boosting_accuracy(Boost_Matrix, &Confusion) * 100.0);
                    //printf("Average Accuracy = %.4f%%\n", compute_average_accuracy(Matrix) * 100.0);
                } else {
                    printf("Voted Accuracy   = %.4f%%\n", compute_voted_accuracy(Matrix, &Confusion, args) * 100.0);
                    printf("Average Accuracy = %.4f%%\n", compute_average_accuracy(Matrix) * 100.0);
                }
                
                // Compute Overall_Confusion and Overall_Confusion_5x2 matrixes
                if (args.caller == CROSSVALFC_CALLER) {
                    for (i = 0; i < test_data.meta.num_classes; i++) {
                        for (j = 0; j < test_data.meta.num_classes; j++) {
                            Overall_Confusion[i][j] += Confusion[i][j];
                            total_sum += Confusion[i][j];
                            if (i == j)
                                diagonal_sum += Confusion[i][j];
                            if (args.do_5x2_cv == TRUE) {
                                Overall_Confusion_5x2[i][j] += Confusion[i][j];
                                overall_total_sum += Confusion[i][j];
                                if (i == j)
                                    overall_diagonal_sum += Confusion[i][j];
                            }
                        }
                    }
                }
            }
            if (args.output_accuracies == VERBOSE) {

	    //DACIESL: NOTE TO SELF, PUT IN SUPPORT HERE FOR ``VERBOSE PERFORMANCE''
	        if (! args.output_laplacean) {
                    if (num_ensembles == 1) {
                        if (args.do_boosting == TRUE)
                            build_boost_probability_matrix(test_data, ensemble[0], &Prob_Matrix);
                        else 
                            build_probability_matrix(test_data, ensemble[0], &Prob_Matrix);
                    } else if (num_ensembles > 1) {
                        DT_Ensemble big_ensemble;
                        concat_ensembles(num_ensembles, ensemble, &big_ensemble);
                        if (args.do_boosting == TRUE)
                            build_boost_probability_matrix(test_data, big_ensemble, &Prob_Matrix);
	  	        else
                            build_probability_matrix(test_data, big_ensemble, &Prob_Matrix);
		    }
	        }

		print_performance_metrics(test_data, Prob_Matrix, test_data.meta.class_names);

		if (args.output_predictions && !args.output_laplacean) {
  	            Prob_Matrix.num_classes = 0;
		}
            }
            if (args.output_predictions || args.output_laplacean) {
                CV_Voting votes;
                votes.num_classes = 0;
                Vote_Cache cache;
                cache.num_classes = 0;
                if (args.format == EXODUS_FORMAT &&
                    ((args.do_boosting == TRUE && ! store_predictions(test_data, cache, Boost_Matrix, votes, Prob_Matrix, dataset, fold_num, args)) ||
                    (args.do_boosting == FALSE && ! store_predictions(test_data, cache, Matrix, votes, Prob_Matrix, dataset, fold_num, args))))
                    fprintf(stderr, "Store Predictions failed\n");
                
                if (args.format == AVATAR_FORMAT &&
                    ((args.do_boosting == TRUE && ! store_predictions_text(test_data, cache, Boost_Matrix, votes, Prob_Matrix, fold_num, args)) ||
                    (args.do_boosting == FALSE && ! store_predictions_text(test_data, cache, Matrix, votes, Prob_Matrix, fold_num, args))))
                    fprintf(stderr, "Store Predictions failed\n");
                
            }
            if (args.output_confusion_matrix) {
                if (args.do_boosting == TRUE)
                    print_confusion_matrix(Boost_Matrix.num_classes, Confusion, test_data.meta.class_names);
                else
                    print_confusion_matrix(Matrix.num_classes, Confusion, test_data.meta.class_names);
            }
            if (args.output_accuracies == ON)
                if ( (args.do_nfold_cv == TRUE && fold_num == args.num_folds-1) ||
                     (args.do_5x2_cv   == TRUE && fold_num % 2 == 1) )
                         printf("\nOverall Voted Accuracy = %.4f%%\n", (float)diagonal_sum * 100.0 / (float)total_sum);
            if (args.output_confusion_matrix) {
                if ( (args.do_nfold_cv == TRUE && fold_num == args.num_folds-1) ||
                     (args.do_5x2_cv   == TRUE && fold_num % 2 == 1) ) {
                    printf("Overall Confusion Matrix:\n\n");
                    if (args.do_boosting == TRUE)
                        print_confusion_matrix(Boost_Matrix.num_classes, Overall_Confusion, test_data.meta.class_names);
                    else
                        print_confusion_matrix(Matrix.num_classes, Overall_Confusion, test_data.meta.class_names);
                }
            }
            printf("\n");
            if (args.do_5x2_cv == TRUE && fold_num == 9) {
                if (args.output_accuracies == ON) {
                    printf("\n5x2 Overall Voted Accuracy = %.4f%%\n", (float)overall_diagonal_sum * 100.0 / (float)overall_total_sum);
                }
                if (args.output_confusion_matrix == TRUE) {
                    printf("5x2 Overall Confusion Matrix:\n\n");
                    if (args.do_boosting == TRUE)
                        print_confusion_matrix(Boost_Matrix.num_classes, Overall_Confusion_5x2, test_data.meta.class_names);
                    else
                        print_confusion_matrix(Matrix.num_classes, Overall_Confusion_5x2, test_data.meta.class_names);
                }
                printf("\n");
            }
            
            if (args.do_boosting == TRUE) {
                for (i = 0; i < test_data.meta.num_examples; i++)
                    free(Boost_Matrix.data[i]);
                free(Boost_Matrix.data);
            } else {
                for (i = 0; i < test_data.meta.num_examples; i++)
                    free(Matrix.data[i]);
                free(Matrix.data);
            }
	    if (args.output_laplacean == TRUE) {
 	        for (i = 0; i < test_data.meta.num_examples; i++)
		    free(Prob_Matrix.data[i]);
		free(Prob_Matrix.data);
	    }
        } else if (args.do_ensemble_majority_vote) {
            // Build matrix for each ensemble
            CV_Matrix *Matrix;
            CV_Prob_Matrix *Prob_Matrix = NULL;
            int **Confusion;
            
            Matrix = (CV_Matrix *)malloc(num_ensembles * sizeof(CV_Matrix));
            for (i = 0; i < num_ensembles; i++) {
                test_data.meta.Missing = ensemble[i].Missing;
                build_prediction_matrix(test_data, ensemble[i], &Matrix[i]);
                //char title[100];
                //sprintf(title, "Matrix%1d", i+1);
                //print_pred_matrix(title, Matrix[i]);
            }

            if (args.output_laplacean) {
                Prob_Matrix = (CV_Prob_Matrix *)malloc(num_ensembles * sizeof(CV_Prob_Matrix));
                    for (i = 0; i < num_ensembles; i++) {
                    test_data.meta.Missing = ensemble[i].Missing;
                    build_probability_matrix(test_data, ensemble[i], &Prob_Matrix[i]);
                }
            }
            
            // Build new matrix which treats each ensemble as a classifier
            CV_Matrix Ensemble_Matrix;
            Ensemble_Matrix.data = (union data_type_union **)malloc(test_data.meta.num_examples * sizeof(union data_type_union *));
            for (i = 0; i < test_data.meta.num_examples; i++)
                Ensemble_Matrix.data[i] = (union data_type_union *)malloc((num_ensembles + 1) * sizeof(union data_type_union));
            Ensemble_Matrix.num_examples = test_data.meta.num_examples;
            Ensemble_Matrix.num_classifiers = num_ensembles;
            Ensemble_Matrix.additional_cols = 1;
            Ensemble_Matrix.num_classes = test_data.meta.num_classes;

            for (i = 0; i < test_data.meta.num_examples; i++) {
                Ensemble_Matrix.data[i][0].Integer = test_data.examples[i].containing_class_num;
                for (j = 0; j < num_ensembles; j++)
                  Ensemble_Matrix.data[i][j+1].Integer = find_best_class_from_matrix(i, Matrix[j], args, i, 0);
            }
            //print_pred_matrix("Ensemble-based", Ensemble_Matrix);

            //DACIESL: Add in for Laplacean support
            CV_Prob_Matrix Ensemble_Prob_Matrix;
            if (args.output_laplacean) {
	        Ensemble_Prob_Matrix.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
		for(i = 0; i < test_data.meta.num_examples; i++)
		    Ensemble_Prob_Matrix.data[i] = (float *)calloc(test_data.meta.num_classes, sizeof(float));
		Ensemble_Prob_Matrix.num_examples = test_data.meta.num_examples;
		Ensemble_Prob_Matrix.num_classes = test_data.meta.num_classes;
                for (i = 0; i < test_data.meta.num_examples; i++) 
                    for (j = 0; j < test_data.meta.num_classes; j++)
		        for (k = 0; k < num_ensembles; k++)
		            Ensemble_Prob_Matrix.data[i][j] += Prob_Matrix[k].data[i][j]/num_ensembles;
	    }
            else
	        Ensemble_Prob_Matrix.num_classes=0;
            if (args.output_accuracies == ON) {
                printf("Voted Accuracy   = %.4f%%\n", compute_voted_accuracy(Ensemble_Matrix, &Confusion, args) * 100.0);
                printf("Average Accuracy = %.4f%%\n", compute_average_accuracy(Ensemble_Matrix) * 100.0);
            }
            if (args.output_predictions || args.output_laplacean) {
                CV_Voting votes;
                votes.num_classes = 0;
                Vote_Cache cache;
                cache.num_classes = 0;
                if (args.format == EXODUS_FORMAT && ! store_predictions(test_data, cache, Ensemble_Matrix, votes, Ensemble_Prob_Matrix, dataset, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
                if (args.format == AVATAR_FORMAT && ! store_predictions_text(test_data, cache, Ensemble_Matrix, votes, Ensemble_Prob_Matrix, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
            }
            if (args.output_confusion_matrix)
                print_confusion_matrix(Ensemble_Matrix.num_classes, Confusion, test_data.meta.class_names);
            
            for (i = 0; i < test_data.meta.num_examples; i++) {
                for (j = 0; j < num_ensembles; j++)
                    free(Matrix[j].data[i]);
                free(Ensemble_Matrix.data[i]);
            }
            for (j = 0; j < num_ensembles; j++)
                free(Matrix[j].data);
            free(Ensemble_Matrix.data);
            free(Matrix);

            //DACIESL: Add in for Laplacean support
            if (args.output_laplacean || args.output_accuracies == VERBOSE) {
                for (i = 0; i < test_data.meta.num_examples; i++) {
                    for (j = 0; j < num_ensembles; j++)
                        free(Prob_Matrix[j].data[i]);
                    free(Ensemble_Prob_Matrix.data[i]);
                }
                for (j = 0; j < num_ensembles; j++)
                    free(Prob_Matrix[j].data);
                free(Ensemble_Prob_Matrix.data);
                free(Prob_Matrix);
	    }           

        } else if (args.do_margin_ensemble_majority_vote) {

            CV_Matrix *Matrix;
            CV_Prob_Matrix *Prob_Matrix;
            int *temp_votes;
            int *temp_classes;
            float *margin_sum;
            int num_correct = 0;
            
            // Initialize the confusion matrix
            int **Confusion;
            Confusion = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
            for (i = 0; i < test_data.meta.num_classes; i++)
                Confusion[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
            
            // Array used for sorting by some value and dragging the class number around, too
            temp_classes = (int *)malloc(test_data.meta.num_classes * sizeof(int));
            margin_sum = (float *)malloc(test_data.meta.num_classes * sizeof(float));
            
            // Build matrix for each ensemble
            Matrix = (CV_Matrix *)malloc(num_ensembles * sizeof(CV_Matrix));
            for (i = 0; i < num_ensembles; i++) {
                test_data.meta.Missing = ensemble[i].Missing;
                build_prediction_matrix(test_data, ensemble[i], &Matrix[i]);
            }

            //DACIESL: Add in for Laplacean support
	    Prob_Matrix = (CV_Prob_Matrix *)malloc(num_ensembles * sizeof(CV_Prob_Matrix));
	    for(i = 0; i < num_ensembles; i++) {
                test_data.meta.Missing = ensemble[i].Missing;
                build_probability_matrix(test_data, ensemble[i], &Prob_Matrix[i]);
	    }

            //DACIESL: Add in for Laplacean support
            CV_Prob_Matrix Ensemble_Prob_Matrix;
            if (args.output_laplacean) {
	        Ensemble_Prob_Matrix.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
		for (i = 0; i < test_data.meta.num_examples; i++)
		    Ensemble_Prob_Matrix.data[i] = (float *)calloc(test_data.meta.num_classes, sizeof(float));
		Ensemble_Prob_Matrix.num_examples = test_data.meta.num_examples;
		Ensemble_Prob_Matrix.num_classes = test_data.meta.num_classes;
                for (i = 0; i < test_data.meta.num_examples; i++) 
                    for (j = 0; j < test_data.meta.num_classes; j++)
		        for (k = 0; k < num_ensembles; k++)
		            Ensemble_Prob_Matrix.data[i][j] += Prob_Matrix[k].data[i][j]/num_ensembles;
	    }
            else
	        Ensemble_Prob_Matrix.num_classes=0;
            
            // Build new matrix which treats each ensemble as a classifier
            CV_Voting Class_Voting;
            Class_Voting.class = (int *)malloc(test_data.meta.num_examples * sizeof(int));
            Class_Voting.best_class = (int *)malloc(test_data.meta.num_examples * sizeof(int));
            Class_Voting.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
            for (i = 0; i < test_data.meta.num_examples; i++)
                Class_Voting.data[i] = (float *)malloc(test_data.meta.num_classes * sizeof(int));
            Class_Voting.num_examples = test_data.meta.num_examples;
            Class_Voting.num_classes = test_data.meta.num_classes;

            for (i = 0; i < test_data.meta.num_examples; i++) {
                for (j = 0; j < test_data.meta.num_classes; j++)
                    margin_sum[j] = 0.0;
                Class_Voting.class[i] = test_data.examples[i].containing_class_num;
                for (j = 0; j < num_ensembles; j++) {
                    //printf("For ensemble %d\n", j);
                    // Count votes for each class for this ensemble
                    count_class_votes_from_matrix(i, Matrix[j], &temp_votes);
                    for (k = 0; k < test_data.meta.num_classes; k++) {
                        //printf("Ensemble %d had %d votes for class %d\n", j, temp_votes[k], k);
                        temp_classes[k] = k;
                    }
                    // Shuffle then sort (which emulates breaking ties randomly) on votes
                    //   or just sort
                    // Keep a running total of the margin for each winning class
                    if (args.break_ties_randomly) {
                        shuffle_sort_int_int(test_data.meta.num_classes, temp_votes, temp_classes, DESCENDING);
                        margin_sum[temp_classes[0]] += (float)(temp_votes[0] - temp_votes[1])/(float)ensemble[j].num_trees;
                        Class_Voting.data[i][temp_classes[0]] = margin_sum[temp_classes[0]];
                        //printf("Winning class %d with margin %d-%d\n", temp_classes[0], temp_votes[0], temp_votes[1]);
                    } else {
                        int_two_array_sort(test_data.meta.num_classes, temp_votes-1, temp_classes-1);
                        int a = test_data.meta.num_classes-1;
                        margin_sum[temp_classes[a]] += (float)(temp_votes[a] - temp_votes[a-1])/(float)ensemble[j].num_trees;
                        Class_Voting.data[i][temp_classes[a]] = margin_sum[temp_classes[a]];
                        //printf("Winning class (%d)%d with margin %d-%d\n", a, temp_classes[a], temp_votes[a], temp_votes[a-1]);
                    }
                }
                //printf("Sample %d has margin sums of", i);
                //for (j = 0; j < test_data.meta.num_classes; j++)
                //    printf(" %g", margin_sum[j]);
                //printf("\n");
                
                // Now shuffle (or not) and sort the margin sums to get the winner.
                // Increment num_correct and add to confusion matrix
                for (j = 0; j < test_data.meta.num_classes; j++)
                    temp_classes[j] = j;
                if (args.break_ties_randomly) {
                    shuffle_sort_float_int(test_data.meta.num_classes, margin_sum, temp_classes, DESCENDING);
                    if (temp_classes[0] == test_data.examples[i].containing_class_num)
                        num_correct++;
                    Class_Voting.best_class[i] = temp_classes[0];
                    Confusion[temp_classes[0]][test_data.examples[i].containing_class_num]++;
                } else {
                    float_int_array_sort(test_data.meta.num_classes, margin_sum-1, temp_classes-1);
                    int a = test_data.meta.num_classes-1;
                    if (temp_classes[a] == test_data.examples[i].containing_class_num)
                        num_correct++;
                    Class_Voting.best_class[i] = temp_classes[a];
                    Confusion[temp_classes[a]][test_data.examples[i].containing_class_num]++;
                }
            }
            
            if (args.output_accuracies == ON || args.output_accuracies == VERBOSE) {
                printf("Voted Accuracy   = %.4f%%\n", (float)num_correct * 100.0 / (float)test_data.meta.num_examples);
            }
            if (args.output_accuracies == VERBOSE) {
                if (!args.output_laplacean) {
	            Ensemble_Prob_Matrix.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
		    for(i = 0; i < test_data.meta.num_examples; i++)
		        Ensemble_Prob_Matrix.data[i] = (float *)calloc(test_data.meta.num_classes, sizeof(float));
		    Ensemble_Prob_Matrix.num_examples = test_data.meta.num_examples;
		    Ensemble_Prob_Matrix.num_classes = test_data.meta.num_classes;
                    for (i = 0; i < test_data.meta.num_examples; i++) 
                        for (j = 0; j < test_data.meta.num_classes; j++)
		            for (k = 0; k < num_ensembles; k++)
		                Ensemble_Prob_Matrix.data[i][j] += Prob_Matrix[k].data[i][j]/num_ensembles;
	        }

		print_performance_metrics(test_data, Ensemble_Prob_Matrix, test_data.meta.class_names);

		if (args.output_predictions && !args.output_laplacean) {
  	            Ensemble_Prob_Matrix.num_classes = 0;
		}
            }
            if (args.output_predictions) {
                CV_Matrix matrix;
                matrix.num_classes = 0;
                Vote_Cache cache;
                cache.num_classes = 0;
                if (args.format == EXODUS_FORMAT && ! store_predictions(test_data, cache, matrix, Class_Voting, Ensemble_Prob_Matrix, dataset, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
                if (args.format == AVATAR_FORMAT && ! store_predictions_text(test_data, cache, matrix, Class_Voting, Ensemble_Prob_Matrix, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
            }
            if (args.output_confusion_matrix)
                print_confusion_matrix(test_data.meta.num_classes, Confusion, test_data.meta.class_names);
            
            for (i = 0; i < test_data.meta.num_examples; i++) {
                for (j = 0; j < num_ensembles; j++)
                    free(Matrix[j].data[i]);
                free(Class_Voting.data[i]);
            }
            for (j = 0; j < num_ensembles; j++)
                free(Matrix[j].data);
            free(Class_Voting.data);
            free(Class_Voting.class);
            free(Class_Voting.best_class);
            free(Matrix);

            //DACIESL: Add in for Laplacean support
            if (args.output_laplacean || args.output_accuracies == VERBOSE) {
                for (i = 0; i < test_data.meta.num_examples; i++) {
                    for (j = 0; j < num_ensembles; j++)
                        free(Prob_Matrix[j].data[i]);
                    free(Ensemble_Prob_Matrix.data[i]);
                }
                for (j = 0; j < num_ensembles; j++)
                    free(Prob_Matrix[j].data);
                free(Ensemble_Prob_Matrix.data);
                free(Prob_Matrix);
	    }   
            
        } else if (args.do_probabilistic_majority_vote || args.do_scaled_probabilistic_majority_vote) {
            
            CV_Matrix *Matrix;
            CV_Prob_Matrix *Prob_Matrix;
            int *temp_classes;
            int *winners_across_ensembles;
            float *weighted_winners;
            int *seen_across_ensembles;
            float *percent_across_ensembles;
            int num_correct = 0;
            
            temp_classes = (int *)malloc(test_data.meta.num_classes * sizeof(int));
            winners_across_ensembles = (int *)malloc(test_data.meta.num_classes * sizeof(int));
            weighted_winners = (float *)malloc(test_data.meta.num_classes * sizeof(float));
            seen_across_ensembles = (int *)calloc(test_data.meta.num_classes, sizeof(int));
            percent_across_ensembles = (float *)calloc(test_data.meta.num_classes, sizeof(float));
            
            // Initialize the confusion matrix
            int **Confusion;
            Confusion = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
            for (i = 0; i < test_data.meta.num_classes; i++)
                Confusion[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
            
            // Build matrix for each ensemble
            Matrix = (CV_Matrix *)malloc(num_ensembles * sizeof(CV_Matrix));
            for (i = 0; i < num_ensembles; i++) {
                test_data.meta.Missing = ensemble[i].Missing;
                build_prediction_matrix(test_data, ensemble[i], &Matrix[i]);
            }

            //DACIESL: Add in for Laplacean support
	    Prob_Matrix = (CV_Prob_Matrix *)malloc(num_ensembles * sizeof(CV_Prob_Matrix));
	    for(i = 0; i < num_ensembles; i++) {
                test_data.meta.Missing = ensemble[i].Missing;
                build_probability_matrix(test_data, ensemble[i], &Prob_Matrix[i]);
	    }

            //DACIESL: Add in for Laplacean support
            CV_Prob_Matrix Ensemble_Prob_Matrix;
            if (args.output_laplacean) {
	        Ensemble_Prob_Matrix.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
		for (i = 0; i < test_data.meta.num_examples; i++)
		    Ensemble_Prob_Matrix.data[i] = (float *)calloc(test_data.meta.num_classes, sizeof(float));
		Ensemble_Prob_Matrix.num_examples = test_data.meta.num_examples;
		Ensemble_Prob_Matrix.num_classes = test_data.meta.num_classes;
                for (i = 0; i < test_data.meta.num_examples; i++) 
                    for (j = 0; j < test_data.meta.num_classes; j++)
		        for (k = 0; k < num_ensembles; k++)
		            Ensemble_Prob_Matrix.data[i][j] += Prob_Matrix[k].data[i][j]/num_ensembles;
	    }
            else
	        Ensemble_Prob_Matrix.num_classes=0;
            
            // Build new matrix which treats each ensemble as a classifier
            CV_Voting Class_Voting;
            Class_Voting.class = (int *)malloc(test_data.meta.num_examples * sizeof(int));
            Class_Voting.best_class = (int *)malloc(test_data.meta.num_examples * sizeof(int));
            Class_Voting.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
            for (i = 0; i < test_data.meta.num_examples; i++)
                Class_Voting.data[i] = (float *)malloc(test_data.meta.num_classes * sizeof(int));
            Class_Voting.num_examples = test_data.meta.num_examples;
            Class_Voting.num_classes = test_data.meta.num_classes;
            
            // Get number of ensembles who have seen each class
            for (i = 0; i < num_ensembles; i++) {
                for (j = 0; j < test_data.meta.num_classes; j++) {
                    if (ensemble[i].num_training_examples_per_class[j] > 0) {
                        if (args.do_probabilistic_majority_vote)
                            seen_across_ensembles[j]++;
                        else if (args.do_scaled_probabilistic_majority_vote)
                            percent_across_ensembles[j] += (float)ensemble[i].num_training_examples_per_class[j] /
                                                           (float)ensemble[i].num_training_examples;
                    }
                }
            }
            
            for (i = 0; i < test_data.meta.num_examples; i++) {
                for (j = 0; j < test_data.meta.num_classes; j++) {
                    winners_across_ensembles[j] = 0;
                }
                for (j = 0; j < num_ensembles; j++) {
                    //printf("For ensemble %d\n", j);
                  winners_across_ensembles[find_best_class_from_matrix(i, Matrix[j], args, i, 0)]++;
                }
                Class_Voting.class[i] = test_data.examples[i].containing_class_num;
                for (j = 0; j < test_data.meta.num_classes; j++) {
                    temp_classes[j] = j;
                    if (args.do_probabilistic_majority_vote) {
                        if (seen_across_ensembles[j] > 0) {
                            weighted_winners[j] = (float)winners_across_ensembles[j] / (float)seen_across_ensembles[j];
                        } else {
                            // If this class has not been seen by any ensemble, it will never be a winner so 0/0=0
                            weighted_winners[j] = 0.0;
                        }
                    } else if (args.do_scaled_probabilistic_majority_vote) {
                        if (percent_across_ensembles[j] > 0) {
                            weighted_winners[j] = (float)winners_across_ensembles[j] / percent_across_ensembles[j];
                        } else {
                            weighted_winners[j] = 0.0;
                        }
                    }
                    Class_Voting.data[i][temp_classes[j]] = weighted_winners[j];
                }
                
                // Now shuffle (or not) and sort the weighted winners to get the winner.
                // Increment num_correct and add to confusion matrix
                if (args.break_ties_randomly) {
                    shuffle_sort_float_int(test_data.meta.num_classes, weighted_winners, temp_classes, DESCENDING);
                    if (temp_classes[0] == test_data.examples[i].containing_class_num)
                        num_correct++;
                    Class_Voting.best_class[i] = temp_classes[0];
                    Confusion[temp_classes[0]][test_data.examples[i].containing_class_num]++;
                } else {
                    float_int_array_sort(test_data.meta.num_classes, weighted_winners-1, temp_classes-1);
                    int a = test_data.meta.num_classes-1;
                    if (temp_classes[a] == test_data.examples[i].containing_class_num)
                        num_correct++;
                    Class_Voting.best_class[i] = temp_classes[a];
                    Confusion[temp_classes[a]][test_data.examples[i].containing_class_num]++;
                }
            }
            
            if (args.output_accuracies == ON || args.output_accuracies == VERBOSE) {
                printf("Voted Accuracy   = %.4f%%\n", (float)num_correct * 100.0 / (float)test_data.meta.num_examples);
            }
            if (args.output_accuracies == VERBOSE) {
                if (!args.output_laplacean) {
	            Ensemble_Prob_Matrix.data = (float **)malloc(test_data.meta.num_examples * sizeof(float *));
		    for (i = 0; i < test_data.meta.num_examples; i++)
		        Ensemble_Prob_Matrix.data[i] = (float *)calloc(test_data.meta.num_classes, sizeof(float));
		    Ensemble_Prob_Matrix.num_examples = test_data.meta.num_examples;
		    Ensemble_Prob_Matrix.num_classes = test_data.meta.num_classes;
                    for (i = 0; i < test_data.meta.num_examples; i++) 
                        for (j = 0; j < test_data.meta.num_classes; j++)
		            for (k = 0; k < num_ensembles; k++)
		                Ensemble_Prob_Matrix.data[i][j] += Prob_Matrix[k].data[i][j]/num_ensembles;
	        }

		print_performance_metrics(test_data, Ensemble_Prob_Matrix, test_data.meta.class_names);

		if (args.output_predictions && !args.output_laplacean) {
  	            Ensemble_Prob_Matrix.num_classes = 0;
		}
            }
            if (args.output_predictions) {
                CV_Matrix matrix;
                matrix.num_classes = 0;
                Vote_Cache cache;
                cache.num_classes = 0;
                if (args.format == EXODUS_FORMAT && ! store_predictions(test_data, cache, matrix, Class_Voting, Ensemble_Prob_Matrix, dataset, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
                if (args.format == AVATAR_FORMAT && ! store_predictions_text(test_data, cache, matrix, Class_Voting, Ensemble_Prob_Matrix, fold_num, args)) {
                   fprintf(stderr, "Store Predictions failed\n");
                }
            }
            if (args.output_confusion_matrix)
                print_confusion_matrix(test_data.meta.num_classes, Confusion, test_data.meta.class_names);
            
            for (i = 0; i < test_data.meta.num_examples; i++) {
                for (j = 0; j < num_ensembles; j++)
                    free(Matrix[j].data[i]);
                free(Class_Voting.data[i]);
            }
            for (j = 0; j < num_ensembles; j++)
                free(Matrix[j].data);
            free(Class_Voting.data);
            free(Class_Voting.class);
            free(Class_Voting.best_class);
            free(Matrix);

            //DACIESL: Add in for Laplacean support
            if (args.output_laplacean || args.output_accuracies == VERBOSE) {
                for (i = 0; i < test_data.meta.num_examples; i++) {
                    for (j = 0; j < num_ensembles; j++)
                        free(Prob_Matrix[j].data[i]);
                    free(Ensemble_Prob_Matrix.data[i]);
                }
                for (j = 0; j < num_ensembles; j++)
                    free(Prob_Matrix[j].data);
                free(Ensemble_Prob_Matrix.data);
                free(Prob_Matrix);
	    } 
            
        } else {
            fprintf(stderr, "Don't know what you want me to do with these %d ensembles\n", num_ensembles);
        }
    }
    
}

void test_ivote(CV_Subset test_data, Vote_Cache cache, FC_Dataset dataset, int fold_num, Args_Opts args) {
    int i, j;
    
    CV_Matrix Matrix;
    CV_Matrix Boost_Matrix;
    CV_Prob_Matrix Prob_Matrix;
    Prob_Matrix.num_classes=0;
    int **Confusion;
    static int **Overall_Confusion;
    static int **Overall_Confusion_5x2;
    static int diagonal_sum = 0;
    static int total_sum = 0;
    static int overall_diagonal_sum = 0;
    static int overall_total_sum = 0;
    
    if (args.output_accuracies == ON || args.output_accuracies == VERBOSE || args.output_predictions || args.output_confusion_matrix || args.output_laplacean) {
        
        // First time through with crossvalfc, initialize Overall_Confusion
        if (args.caller == CROSSVALFC_CALLER && ( (args.do_nfold_cv == TRUE && fold_num == 0) ||
                                                  (args.do_5x2_cv == TRUE && fold_num % 2 == 0) )) {
            Overall_Confusion = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
            for (i = 0; i < test_data.meta.num_classes; i++)
                Overall_Confusion[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
            
            if (fold_num == 0 && args.do_5x2_cv == TRUE) {
                Overall_Confusion_5x2 = (int **)malloc(test_data.meta.num_classes * sizeof(int *));
                for (i = 0; i < test_data.meta.num_classes; i++)
                    Overall_Confusion_5x2[i] = (int *)calloc(test_data.meta.num_classes, sizeof(int));
            }
            
            diagonal_sum = 0;
            total_sum = 0;
        }
        
        if (args.do_boosting == TRUE) {
            build_boost_prediction_matrix_for_ivote(cache, &Boost_Matrix);
	    if(args.output_laplacean == TRUE)
                build_boost_probability_matrix_for_ivote(cache, &Prob_Matrix);
        } else {
            build_prediction_matrix_for_ivote(test_data, cache, &Matrix);
	    if(args.output_laplacean == TRUE)
                build_probability_matrix_for_ivote(cache, &Prob_Matrix);
        }
        if (args.output_accuracies == ON || args.output_accuracies == VERBOSE) {
            if (args.do_boosting == TRUE) {
                printf("Boosting Accuracy = %.4f%%\n", compute_boosting_accuracy(Boost_Matrix, &Confusion) * 100.0);
            } else {
                printf("Voted Accuracy   = %.4f%%\n", compute_voted_accuracy(Matrix, &Confusion, args) * 100.0);
                printf("Average Accuracy = %.4f%%\n", cache.average_test_accuracy * 100.0);
                //printf("Average Accuracy = %.4f%%\n", compute_average_accuracy(Matrix) * 100.0);
            }
            
            // Compute Overall_Confusion matrix
            if (args.caller == CROSSVALFC_CALLER) {
                for (i = 0; i < test_data.meta.num_classes; i++) {
                    for (j = 0; j < test_data.meta.num_classes; j++) {
                        Overall_Confusion[i][j] += Confusion[i][j];
                        total_sum += Confusion[i][j];
                        if (i == j)
                            diagonal_sum += Confusion[i][j];
                        if (args.do_5x2_cv == TRUE) {
                            Overall_Confusion_5x2[i][j] += Confusion[i][j];
                            overall_total_sum += Confusion[i][j];
                            if (i == j)
                                overall_diagonal_sum += Confusion[i][j];
                        }
                    }
                }
            }
        }
	if (args.output_accuracies == VERBOSE) {
	    if (!args.output_laplacean) {
                if (args.do_boosting == TRUE) 
                    build_boost_probability_matrix_for_ivote(cache, &Prob_Matrix);
                else 
                    build_probability_matrix_for_ivote(cache, &Prob_Matrix);
	    }
	    print_performance_metrics(test_data, Prob_Matrix, test_data.meta.class_names);
	    if (args.output_predictions && !args.output_laplacean) {
  	        Prob_Matrix.num_classes = 0;
	    }
	}
        if (args.output_predictions || args.output_laplacean) {
            CV_Voting votes;
            votes.num_classes = 0;
            //printf("Going in with %d classes and %d classifiers\n", Matrix.num_classes, Matrix.num_classifiers);
            if (args.format == EXODUS_FORMAT && ! store_predictions(test_data, cache, Matrix, votes, Prob_Matrix, dataset, fold_num, args)) {
               fprintf(stderr, "Store Predictions failed\n");
            }
            if (args.format == AVATAR_FORMAT && ! store_predictions_text(test_data, cache, Matrix, votes, Prob_Matrix, fold_num, args)) {
               fprintf(stderr, "Store Predictions failed\n");
            }
        }
        if (args.output_confusion_matrix)
            print_confusion_matrix(Matrix.num_classes, Confusion, test_data.meta.class_names);
        
        if (args.output_accuracies == ON)
            if ( (args.do_nfold_cv == TRUE && fold_num == args.num_folds-1) ||
                 (args.do_5x2_cv   == TRUE && fold_num % 2 == 1) )
                     printf("\nOverall Voted Accuracy = %.4f%%\n", (float)diagonal_sum * 100.0 / (float)total_sum);
        if (args.output_confusion_matrix) {
            if ( (args.do_nfold_cv == TRUE && fold_num == args.num_folds-1) ||
                 (args.do_5x2_cv   == TRUE && fold_num % 2 == 1) ) {
                     printf("Overall Confusion Matrix:\n\n");
                     print_confusion_matrix(Matrix.num_classes, Overall_Confusion, test_data.meta.class_names);
            }
        }
        printf("\n");
        if (args.do_5x2_cv == TRUE && fold_num == 9) {
            if (args.output_accuracies == ON) {
                printf("\n5x2 Overall Voted Accuracy = %.4f%%\n", (float)overall_diagonal_sum * 100.0 / (float)overall_total_sum);
            }
            if (args.output_confusion_matrix == TRUE) {
                printf("5x2 Overall Confusion Matrix:\n\n");
                if (args.do_boosting == TRUE)
                    print_confusion_matrix(Boost_Matrix.num_classes, Overall_Confusion_5x2, test_data.meta.class_names);
                else
                    print_confusion_matrix(Matrix.num_classes, Overall_Confusion_5x2, test_data.meta.class_names);
            }
            printf("\n");
        }
    }
    
/* No clue why this is causing memory errors
    for (i = 0; i < cache.num_test_examples; i++)
        free(Matrix.data[i]);
    free(Matrix.data);
*/
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//added consideration for saving class counts at leafs
void _save_node(DT_Node *tree, int node, FILE *fh, int num_classes) {
    int i;
    if (tree[node].branch_type == LEAF) {
        if (tree[node].class_count[0]==-1) {
            fprintf(fh, "LEAF %d\n", tree[node].Node_Value.class_label);
        } else {
	  fprintf(fh, "LEAF Class %d Proportions ", tree[node].Node_Value.class_label);
            for (i = 0; i < num_classes; i++) {
	        fprintf(fh, "%d%s",tree[node].class_count[i],i==(num_classes-1)?"\n":" ");
            }
        }
    } else {
        if (tree[node].attribute_type == CONTINUOUS) {
            fprintf(fh, "SPLIT CONTINUOUS ATT# %d < %#6g\n", tree[node].attribute, tree[node].branch_threshold);
            _save_node(tree, tree[node].Node_Value.branch[0], fh, num_classes);
            fprintf(fh, "SPLIT CONTINUOUS ATT# %d >= %#6g\n", tree[node].attribute, tree[node].branch_threshold);
            _save_node(tree, tree[node].Node_Value.branch[1], fh, num_classes);
        } else if (tree[node].attribute_type == DISCRETE) {
            for (i = 0; i < tree[node].num_branches; i++) {
                fprintf(fh, "SPLIT DISCRETE ATT# %d VAL# %d / %d\n", tree[node].attribute, i+1, tree[node].num_branches);
                _save_node(tree, tree[node].Node_Value.branch[i], fh, num_classes);
            }
        }
    }
}

char* build_output_filename(int fold_num, char *filename, Args_Opts args) {
    char *output_filename;
    if (fold_num < 0) {
        output_filename = av_strdup(filename);
    } else {
        File_Bits bits = explode_filename(filename);
        int size = strlen(bits.dirname) + strlen(bits.basename) + 2*num_digits(fold_num+1) + strlen(bits.extension) + 10;
        output_filename = (char *)malloc(size * sizeof(char));
        if (args.do_5x2_cv)
            sprintf(output_filename, "%s/%s-%d-%d.%s", bits.dirname, bits.basename, fold_num/args.num_folds + 1,
                                                     fold_num%args.num_folds + 1, bits.extension);
        else
            sprintf(output_filename, "%s/%s-%d.%s", bits.dirname, bits.basename, fold_num+1, bits.extension);
    }
    return(output_filename);
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//added consideration for saving class counts at leafs
void save_tree(DT_Node *tree, int fold_num, int tree_num, Args_Opts args, int num_classes) {
    FILE *tree_file;
    char *tree_filename = build_output_filename(fold_num, args.trees_file, args);
    if ((tree_file = fopen(tree_filename, "a")) == NULL) {
        fprintf(stderr, "Failed to open file for saving trees: '%s'\nExiting ...\n", tree_filename);
        exit(8);
    }
    fprintf(tree_file, "\nTree %d\n", tree_num);
    _save_node(tree, 0, tree_file, num_classes);
    fclose(tree_file);
    free(tree_filename);
}

void read_ensemble_metadata(FILE *fh, DT_Ensemble *ensemble, int force_num_trees, Args_Opts *args) {
    int i;
    int skip_offset;
    char strbuf[262144];
    
    // Read metadata
    
    // Skip comments at top of file
    fscanf(fh, "%s", strbuf);
    while (! strncmp(strbuf, "#", 1)) {
        // Finish reading the comment line
        if (fscanf(fh, "%[^\n]", strbuf) <= 0)
            fscanf(fh, "%[\n]", strbuf);
        // Read start of next line
        fscanf(fh, "%s", strbuf);
    }
    if (! strcmp(strbuf, "NumTrainingExamples") && fscanf(fh, "%s", strbuf) > 0) {
        ensemble->num_training_examples = atoi(strbuf);
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "NumClasses") && fscanf(fh, "%s", strbuf) > 0) {
        ensemble->num_classes = atoi(strbuf);
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "NumExamplesPerClass:") && ensemble->num_classes > 0) {
        free(ensemble->num_training_examples_per_class);
        ensemble->num_training_examples_per_class = (int *)malloc(ensemble->num_classes * sizeof(int));
        for (i = 0; i < ensemble->num_classes; i++) {
            fscanf(fh, "%s", strbuf);
            ensemble->num_training_examples_per_class[i] = atoi(strbuf);
        }
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "NumAttributes") && fscanf(fh, "%s", strbuf) > 0) {
        ensemble->num_attributes = atoi(strbuf);
        free(ensemble->attribute_types);
        ensemble->attribute_types = (Attribute_Type *)malloc(ensemble->num_attributes * sizeof(Attribute_Type));
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "NumSkippedAttributes") && fscanf(fh, "%s", strbuf) > 0) {
        args->num_skipped_features = atoi(strbuf);
        free(args->skipped_features);
        args->skipped_features = (int *)malloc(args->num_skipped_features * sizeof(int));
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "NumTrees") && fscanf(fh, "%s", strbuf) > 0) {
        ensemble->num_trees = atoi(strbuf);
        // If force_num_trees > 0, then don't trust the number in the file
        if (force_num_trees > 0)
            ensemble->num_trees = force_num_trees;
        args->num_trees = ensemble->num_trees;
        free(ensemble->Trees);
        ensemble->Trees = (DT_Node **)calloc(ensemble->num_trees, sizeof(DT_Node *));
        free(ensemble->Books);
        ensemble->Books = (Tree_Bookkeeping *)calloc(ensemble->num_trees, sizeof(Tree_Bookkeeping));
    }
    fscanf(fh, "%s", strbuf);
    skip_offset = 0;
    if (! strcmp(strbuf, "AttributeTypes:") && ensemble->num_attributes > 0) {
        for (i = 0; i < ensemble->num_attributes + args->num_skipped_features; i++) {
            fscanf(fh, "%s", strbuf);
            if (! strcmp(strbuf, "CONTINUOUS")) {
                ensemble->attribute_types[i-skip_offset] = CONTINUOUS;
            } else if (! strcmp(strbuf, "DISCRETE")) {
                ensemble->attribute_types[i-skip_offset] = DISCRETE;
            } else if (! strcmp(strbuf, "UNKNOWN")) {
                skip_offset++;
            } else {
                fprintf(stderr, "Got unexpected attribute type: '%s'\n", strbuf);
            }
        }
    }
    fscanf(fh, "%s", strbuf);
    skip_offset = 0;
    if (! strcmp(strbuf, "SkipAttributes:") && ensemble->num_attributes > 0) {
        for (i = 0; i < ensemble->num_attributes + args->num_skipped_features; i++) {
            fscanf(fh, "%s", strbuf);
            if (! strcmp(strbuf, "SKIP")) {
                args->skipped_features[skip_offset++] = i+1;
                if (! args->do_training) {
                    //printf("Adding feature %d to skipped list\n", i);
                }
            } else if (! strcmp(strbuf, "NOSKIP")) {
                // Do nothing for now. We're just skipping over these
            } else {
                fprintf(stderr, "Expected 'SKIP' or 'NOSKIP' but got '%s'\n", strbuf);
            }
        }
    }
    fscanf(fh, "%s", strbuf);
    if (! strcmp(strbuf, "MissingAttributeValues:") && ensemble->num_attributes > 0) {
        free(ensemble->Missing);
        ensemble->Missing = (union data_point_union *)malloc(ensemble->num_attributes * sizeof(union data_point_union));
        fscanf(fh, "%[^\n]", strbuf);
        int num_values;
        char **values = NULL;
        parse_delimited_string(',', strbuf, &num_values, &values);
        skip_offset = 0;
        for (i = 0; i < num_values; i++) {
            if (strcmp(values[i], "??")) {
                if (ensemble->attribute_types[i-skip_offset] == CONTINUOUS)
                    ensemble->Missing[i-skip_offset].Continuous = atof(values[i]);
                else if (ensemble->attribute_types[i-skip_offset] == DISCRETE)
                    ensemble->Missing[i-skip_offset].Discrete = atoi(values[i]);
            } else {
                skip_offset++;
            }
        }
        if(values)
        {
          for(i = 0; i < num_values; i++)
            free(values[i]);
          free(values);
        }
    }
}

//Added by DACIESL June-09-08: Laplacean Estimates
//added to check if new or old trees are being used
void check_tree_version(int fold_num, Args_Opts *args) {
    char strbuf[262144];
    char *tree_filename = build_output_filename(fold_num, args->trees_file, *args);
    FILE *tree_file;


    const int tree_file_is_a_string = 0;
    // Open tree file
    if (tree_file_is_a_string){
        tree_file = fmemopen(args->trees_file, strlen(args->trees_file), "r");
    }
    else {
	if ((tree_file = fopen(tree_filename, "r")) == NULL) {
            fprintf(stderr, "Failed to open file for reading trees: '%s'\nExiting ...\n", tree_filename);
            exit(8);
        }
    }

    
    fscanf(tree_file, "%s", strbuf);
    while (! strcmp(strbuf, "LEAF")) {
        strbuf[0]='\0';
        fscanf(tree_file, "%s", strbuf);
    }
    
    fscanf(tree_file, "%s", strbuf);
    strbuf[0]='\0';
    fscanf(tree_file, "%s", strbuf);
    
    if (strcmp(strbuf, "Class"))
        args->output_probabilities_warning=TRUE;
    fclose(tree_file);
    free(tree_filename);
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//added new _read_tree call
void read_ensemble(DT_Ensemble *ensemble, int fold_num, int force_num_trees, Args_Opts *args) {
    int i;
    char strbuf[262144];
    char *tree_filename = build_output_filename(fold_num, args->trees_file, *args);
    FILE *tree_file;

    // Initialize
    ensemble->num_classes = 0;
    ensemble->num_attributes = 0;
    ensemble->num_trees = 0;
    
    // Open tree file
    if (args->trees_file_is_a_string == TRUE){
	tree_file = fmemopen(args->trees_string, strlen(args->trees_string), "r");
    } else {
	if ((tree_file = fopen(tree_filename, "r")) == NULL) {
        	fprintf(stderr, "Failed to open file for reading trees: '%s'\nExiting ...\n", tree_filename);
        	exit(8);
    	}
    }

    
    // Read metadata
    read_ensemble_metadata(tree_file, ensemble, force_num_trees, args);

    // Read trees
    int this_tree_num = -1;
    for (i = 0; i < ensemble->num_trees; i++) {
        fscanf(tree_file, "%s", strbuf);
        while (! strncmp(strbuf, "#", 1)) {
            // Finish reading the comment line
            //printf("Comment line: '%s", strbuf);
            if (fscanf(tree_file, "%[^\n]", strbuf) <= 0)
                fscanf(tree_file, "%[\n]", strbuf);
            //printf("%s'\n", strbuf);
            // Read start of next line
            fscanf(tree_file, "%s", strbuf);
        }
        if (! strcmp(strbuf, "Tree") && fscanf(tree_file, "%s", strbuf) > 0) {
            this_tree_num = atoi(strbuf);
            if (this_tree_num != i+1)
                fprintf(stderr, "Found tree #%d but expected tree #%d\n", this_tree_num, i+1);
        }
        fscanf(tree_file, "%s", strbuf);
        if (! strcmp(strbuf, "Beta") && fscanf(tree_file, "%s", strbuf) > 0) {
            if (this_tree_num == 1) {
                // Initialize boosting_betas array and set do_boosting to TRUE
                ensemble->boosting_betas = (double *)malloc(sizeof(double));
                args->do_boosting = TRUE;
            } else {
                // Reallocate for new number of trees
                ensemble->boosting_betas = (double *)realloc(ensemble->boosting_betas, this_tree_num * sizeof(double));
            }
            ensemble->boosting_betas[this_tree_num-1] = atof(strbuf);
            strbuf[0] = '\0'; // Zero out strbuf to signify that _read_tree needs to read first
        } else {
            // Didn't find a 'Beta' line so _read_tree should NOT read as its first action
            // If we're boosting, this is a problem
            if (args->do_boosting == TRUE) {
                fprintf(stderr, "ERROR: Ensemble file format error - no beta for tree %d\n", this_tree_num);
                exit(-1);
            }
        }
        //printf("Reading Tree %d (i=%d)\n", this_tree_num, i);
        ensemble->Books[i].num_malloced_nodes = 1;
        ensemble->Books[i].next_unused_node = 1;
        ensemble->Books[i].current_node = 0;
        ensemble->Trees[i] = (DT_Node *)malloc(ensemble->Books[i].num_malloced_nodes * sizeof(DT_Node));
        _read_tree(strbuf, &ensemble->Trees[i], &ensemble->Books[i], tree_file, ensemble->num_classes);
    }
    
    fclose(tree_file);
    free(tree_filename);
    
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//added consideration for reading class counts from file
void _read_tree(char *initial_str, DT_Node **tree, Tree_Bookkeeping *book, FILE *fh, int num_classes) {
    int this_node = book->current_node;
    char strbuf[16384];
    int i;
    int rc = 1; // Initialize in case we don't do an initial read
    //printf("Reading trees...\n");
    
    // If we're passed an empty string, then read from fh. Otherwise, use what we were given
    if (! strcmp(initial_str, ""))
        rc = fscanf(fh, "%s", strbuf);
    else
        strcpy(strbuf, initial_str);
    
    //printf("First read: '%s' (%d)\n", strbuf, rc);
    if (strbuf == NULL) {
        fprintf(stderr, "Didn't read anything. Returning\n");
        return;
    }
    if (rc == EOF) {
        fprintf(stderr, "Got EOF. Returning\n");
        return;
    }
    
    // We have a leaf
    if (! strcmp(strbuf, "LEAF")) {
        while (this_node >= book->num_malloced_nodes) {
            book->num_malloced_nodes *= 2;
            *tree = (DT_Node *)realloc(*tree, book->num_malloced_nodes * sizeof(DT_Node));
        }
        //printf("Node %d is a leaf\n", this_node);
        (*tree)[this_node].branch_type = LEAF;
        (*tree)[this_node].num_branches = 0;
        int j;
        float total = 0.0;
        fscanf(fh, "%s", strbuf);
	if (! strcmp(strbuf, "Class")) {
            fscanf(fh, "%d", &j);
            (*tree)[this_node].Node_Value.class_label = j;
            (*tree)[this_node].class_count = (int *)calloc(num_classes, sizeof(int));
            //printf("Allocate class_probs in _read_tree() for Class for node %d\n", this_node);
            (*tree)[this_node].class_probs = (float *)calloc(num_classes, sizeof(float));
            fscanf(fh, "%s", strbuf);
            for (i = 0; i < num_classes; i++) {
                fscanf(fh, "%d", &j);
                (*tree)[this_node].class_count[i] = j;
                total += j;
            }
            for (i = 0; i < num_classes; i++) {
  	        (*tree)[this_node].class_probs[i] = ((*tree)[this_node].class_count[i] + 1.0)/(num_classes + total);
            }
	} else {
            (*tree)[this_node].Node_Value.class_label = atoi(strbuf);
            (*tree)[this_node].class_count = (int *)calloc(num_classes, sizeof(int));
            //printf("Allocate class_probs in _read_tree() for !Class for node %d\n", this_node);
            (*tree)[this_node].class_probs = (float *)calloc(num_classes, sizeof(float));
            for (i = 0; i < num_classes; i++) {
                (*tree)[this_node].class_count[i] = -1;
            }
            for (i = 0; i < num_classes; i++) {
	        (*tree)[this_node].class_probs[i] = 0.0;
            }
	    (*tree)[this_node].class_probs[(*tree)[this_node].Node_Value.class_label] = 1.0;
	}
        return;
    }
    
    // We have a split
    else if (! strcmp(strbuf, "SPLIT")) {
        fscanf(fh, "%s", strbuf);
        if (! strcmp(strbuf, "DISCRETE")) {
            while (this_node >= book->num_malloced_nodes) {
                book->num_malloced_nodes *= 2;
                *tree = (DT_Node *)realloc(*tree, book->num_malloced_nodes * sizeof(DT_Node));
            }
            (*tree)[this_node].branch_type  = BRANCH;
            (*tree)[this_node].attribute_type = DISCRETE;
            fscanf(fh, "%s", strbuf); // should be "ATT#"
            fscanf(fh, "%s", strbuf);
            (*tree)[this_node].attribute = atoi(strbuf);
            fscanf(fh, "%s", strbuf); // should be "VAL#"
            fscanf(fh, "%s", strbuf); // should be the branch number
            int this_branch_number = atoi(strbuf);
            fscanf(fh, "%s", strbuf); // should be "/"
            fscanf(fh, "%s", strbuf); // should be total number of splits
            (*tree)[this_node].num_branches = atoi(strbuf);
            if (this_branch_number == 1)
                (*tree)[this_node].Node_Value.branch = (int *)malloc((*tree)[this_node].num_branches * sizeof(int));
            //printf("Node %d has child %d\n", this_node, book->next_unused_node);
            (*tree)[this_node].Node_Value.branch[this_branch_number-1] = book->next_unused_node;
            book->current_node = book->next_unused_node;
            book->next_unused_node++;
            _read_tree("", tree, book, fh, num_classes);
            while (this_branch_number < (*tree)[this_node].num_branches) {
                // Handle comments
                fscanf(fh, "%s", strbuf);
                while (! strncmp(strbuf, "#", 1)) {
                    // Finish reading the comment line
                    //printf("Comment line: '%s", strbuf);
                    if (fscanf(fh, "%[^\n]", strbuf) <= 0)
                        fscanf(fh, "%[\n]", strbuf);
                    //printf("%s'\n", strbuf);
                    // Read start of next line
                    fscanf(fh, "%s", strbuf);
                }
                for (i = 1; i < 5; i++)
                    fscanf(fh, "%s", strbuf);
                fscanf(fh, "%s", strbuf); // should be the branch number
                this_branch_number = atoi(strbuf);
                fscanf(fh, "%s", strbuf); // should be "/"
                fscanf(fh, "%s", strbuf); // should be total number of splits
                if (atoi(strbuf) != (*tree)[this_node].num_branches) {
                    fprintf(stderr, "ERROR: Error reading tree file: Got %d branches but expected %d\n",
                                    atoi(strbuf), (*tree)[this_node].num_branches);
                    exit(-8);
                }
                //printf("Node %d has child %d\n", this_node, book->next_unused_node);
                (*tree)[this_node].Node_Value.branch[this_branch_number-1] = book->next_unused_node;
                book->current_node = book->next_unused_node;
                book->next_unused_node++;
                _read_tree("", tree, book, fh, num_classes);
            }
        } else if (! strcmp(strbuf, "CONTINUOUS")) {
            while (this_node >= book->num_malloced_nodes) {
                book->num_malloced_nodes *= 2;
                *tree = (DT_Node *)realloc(*tree, book->num_malloced_nodes * sizeof(DT_Node));
            }
            (*tree)[this_node].branch_type  = BRANCH;
            (*tree)[this_node].attribute_type = CONTINUOUS;
            fscanf(fh, "%s", strbuf); // should be "ATT#"
            fscanf(fh, "%s", strbuf);
            (*tree)[this_node].attribute = atoi(strbuf);
            fscanf(fh, "%s", strbuf); // should be "<" or ">="
            fscanf(fh, "%s", strbuf);
            (*tree)[this_node].branch_threshold = atof(strbuf);
            (*tree)[this_node].num_branches = 2;
            (*tree)[this_node].Node_Value.branch = (int *)malloc(2 * sizeof(int));
            //printf("Node %d has child %d\n", this_node, book->next_unused_node);
            (*tree)[this_node].Node_Value.branch[0] = book->next_unused_node;
            book->current_node = book->next_unused_node;
            book->next_unused_node++;
            _read_tree("", tree, book, fh, num_classes);
            //printf("Node %d has child %d\n", this_node, book->next_unused_node);
            (*tree)[this_node].Node_Value.branch[1] = book->next_unused_node;
            book->current_node = book->next_unused_node;
            book->next_unused_node++;
            // Skip over next line which is the redefinition of the previous SPLIT
            // First check for comments
            fscanf(fh, "%s", strbuf);
            while (! strncmp(strbuf, "#", 1)) {
                // Finish reading the comment line
                //printf("Comment line: '%s", strbuf);
                if (fscanf(fh, "%[^\n]", strbuf) <= 0)
                    fscanf(fh, "%[\n]", strbuf);
                //printf("%s'\n", strbuf);
                // Read start of next line
                fscanf(fh, "%s", strbuf);
            }
            for (i = 1; i < 6; i++)
                fscanf(fh, "%s", strbuf);
            _read_tree("", tree, book, fh, num_classes);
        } else {
            fprintf(stderr, "Got unknown node type: '%s'\n", strbuf);
        }
    }
    
    else if (! strncmp(strbuf, "#", 1)) {
        // Finish reading the comment line
        //printf("Comment line: '%s", strbuf);
        if (fscanf(fh, "%[^\n]", strbuf) <= 0)
            fscanf(fh, "%[\n]", strbuf);
        //printf("%s'\n", strbuf);
        // I am ignoring comments so call myself again to get the next line and keep going
        _read_tree("", tree, book, fh, num_classes);
    }
    
    // Unknown line
    else {
        fprintf(stderr, "Got unexpected token inside a tree: '%s'\n", strbuf);
    }
    
    return;
}

char* write_tree_file_header(int num_trees, CV_Metadata meta, int fold_num, char *tree_filename, Args_Opts args) {
    int i;
    FILE *tree_file;
    // Print hash marks before metadata if we're writing the oob-data file
    char *comment;
    if (! strcmp(tree_filename, args.oob_file))
        comment = av_strdup("#");
    else
        comment = av_strdup("");
    char *mod_tree_filename = build_output_filename(fold_num, tree_filename, args);
    if ((tree_file = fopen(mod_tree_filename, "w")) == NULL) {
        fprintf(stderr, "Failed to open file for writing metadata: '%s'\nExiting ...\n", mod_tree_filename);
        exit(8);
    }
    
    // Print run summary info to ensemble file
    if (! strcmp(tree_filename, args.trees_file)) {
        if (args.caller == CROSSVALFC_CALLER)
            display_cv_opts(tree_file, "#", args);
        else if (args.caller == AVATARDT_CALLER)
            display_dt_opts(tree_file, "#", args, meta);
        else if (args.caller == AVATARMPI_CALLER)
            display_dtmpi_opts(tree_file, "#", args);
        else if (args.caller == RFFEATUREVALUE_CALLER)
            display_fv_opts(tree_file, "#", args);
    }

    fprintf(tree_file, "%sNumTrainingExamples  %d\n", comment, meta.num_examples);
    fprintf(tree_file, "%sNumClasses           %d\n", comment, meta.num_classes);
    fprintf(tree_file, "%sNumExamplesPerClass: ", comment);
    for (i = 0; i < meta.num_classes; i++)
        fprintf(tree_file, "%d ", meta.num_examples_per_class[i]);
    fprintf(tree_file, "\n");
    fprintf(tree_file, "%sNumAttributes        %d\n", comment, meta.num_attributes);
    fprintf(tree_file, "%sNumSkippedAttributes %d\n", comment, args.num_skipped_features);
    fprintf(tree_file, "%sNumTrees             %d\n", comment, num_trees);
    fprintf(tree_file, "%sAttributeTypes: ", comment);
    int skip_offset = 0;
    for (i = 0; i < meta.num_attributes + args.num_skipped_features; i++) {
        if (args.truth_column > 0 && i+1 >= args.truth_column &&
            find_int(i+2, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is in a column past the truth column so look for the
            // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
            // of features to skip since the list is based on column number and not attribute number
            //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "UNKNOWN ");
        } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is before the truth column so attribute number corresponds
            // to column number.
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "UNKNOWN ");
        } else if (args.truth_column < 0 &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The truth column is last so all attributes are before the truth column
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "UNKNOWN ");
        } else if (meta.attribute_types[i-skip_offset] == DISCRETE) {
            fprintf(tree_file, "DISCRETE ");
        } else if (meta.attribute_types[i-skip_offset] == CONTINUOUS) {
            fprintf(tree_file, "CONTINUOUS ");
        } else {
            fprintf(tree_file, "?? ");
        }
    }
    fprintf(tree_file, "\n");
    fprintf(tree_file, "%sSkipAttributes: ", comment);
    for (i = 0; i < meta.num_attributes + args.num_skipped_features; i++) {
        if (args.truth_column > 0 && i+1 >= args.truth_column &&
            find_int(i+2, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is in a column past the truth column so look for the
            // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
            // of features to skip since the list is based on column number and not attribute number
            //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "SKIP ");
        } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is before the truth column so attribute number corresponds
            // to column number.
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "SKIP ");
        } else if (args.truth_column < 0 &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The truth column is last so all attributes are before the truth column
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "SKIP ");
        } else {
            fprintf(tree_file, "NOSKIP ");
        }
    }
    fprintf(tree_file, "\n");
    fprintf(tree_file, "%sMissingAttributeValues: ", comment);
    skip_offset = 0;
    for (i = 0; i < meta.num_attributes + args.num_skipped_features; i++) {
        //printf("Looking at 0-based att %d\n", i);
        if (args.truth_column > 0 && i+1 >= args.truth_column &&
            find_int(i+2, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is in a column past the truth column so look for the
            // attribute number + 2 (extra +1 is because all_atts is 0-based) in the list
            // of features to skip since the list is based on column number and not attribute number
            //printf("Skipping this att because %d+2 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "??");
        } else if (args.truth_column > 0 && i+1 < args.truth_column &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The current attribute is before the truth column so attribute number corresponds
            // to column number.
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "??");
        } else if (args.truth_column < 0 &&
                   find_int(i+1, args.num_skipped_features, args.skipped_features)) {
            // The truth column is last so all attributes are before the truth column
            //printf("Skipping this att because %d+1 is in the list of features to skip\n", i);
            skip_offset++;
            fprintf(tree_file, "??");
        } else if (args.format == AVATAR_FORMAT && meta.attribute_types[i-skip_offset] == DISCRETE) {
            //printf("Trying to print mapping for discrete %d\n", i-skip_offset);
            fprintf(tree_file, "%s", meta.discrete_attribute_map[i-skip_offset][meta.Missing[i-skip_offset].Discrete]);
        } else if (args.format == AVATAR_FORMAT && meta.attribute_types[i-skip_offset] == CONTINUOUS) {
            //printf("Trying to print mapping for continuous %d\n", i-skip_offset);
            fprintf(tree_file, "%g", meta.Missing[i-skip_offset].Continuous);
        } else {
            //printf("Just printing ??\n");
            fprintf(tree_file, "??");
        }
        if (i < meta.num_attributes + args.num_skipped_features - 1)
            fprintf(tree_file, ",");
    }
    fprintf(tree_file, "\n");
    
    fclose(tree_file);
    
    find_int_release();
    return(mod_tree_filename);
}

/*
 * Not used at the current time
 *
void write_tree_file_header_E(DT_Ensemble ensemble, int fold_num, Args_Opts args) {
    int i;
    FILE *tree_file;
    char *tree_filename;
    
    int size;
    if (fold_num < 0) {
        size = strlen(args.data_path) + strlen(args.base_filestem) + 9;
        tree_filename = (char *)malloc(size * sizeof(char));
        sprintf(tree_filename, "%s/%s.trees", args.data_path, args.base_filestem);
    } else {
        size = strlen(args.data_path) + strlen(args.base_filestem) + 2*num_digits(fold_num+1) + 10;
        tree_filename = (char *)malloc(size * sizeof(char));
        if (args.do_5x2_cv)
            sprintf(tree_filename, "%s/%s-%d-%d.trees", args.data_path, args.base_filestem,
                                                        fold_num/args.num_folds + 1, fold_num%args.num_folds +1);
        else
            sprintf(tree_filename, "%s/%s-%d.trees", args.data_path, args.base_filestem, fold_num+1);
    }
    if ((tree_file = fopen(tree_filename, "w")) == NULL) {
        fprintf(stderr, "Failed to open file for saving trees: '%s'\nExiting ...\n", tree_filename);
        exit(8);
    }
    fprintf(tree_file, "NumClasses      %d\n", ensemble.num_classes);
    fprintf(tree_file, "NumAttributes   %d\n", ensemble.num_attributes);
    fprintf(tree_file, "NumTrees        %d\n", ensemble.num_trees);
    fprintf(tree_file, "AttributeTypes: ");
    for (i = 0; i < ensemble.num_attributes; i++)
        fprintf(tree_file, "%s ", ensemble.attribute_types[i]==CONTINUOUS?"CONTINUOUS":"DISCRETE");
        fclose(tree_file);
    free(tree_filename);
}
 *
 */

//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new _save_node call
void save_ensemble(DT_Ensemble ensemble, CV_Metadata data, int fold_num, Args_Opts args, int num_classes) {
    int i;
    FILE *tree_file;
    char *tree_filename = write_tree_file_header(ensemble.num_trees, data, fold_num, args.trees_file, args);
    if ((tree_file = fopen(tree_filename, "a")) == NULL) {
        fprintf(stderr, "Failed to open file for saving trees: '%s'\nExiting ...\n", tree_filename);
        exit(8);
    }
    for (i = 0; i < ensemble.num_trees; i++) {
        fprintf(tree_file, "Tree %d\n", i+1);
        if (args.do_boosting == TRUE)
            fprintf(tree_file, "Beta %g\n", ensemble.boosting_betas[i]);
        _save_node(ensemble.Trees[i], 0, tree_file, num_classes);
    }
    fclose(tree_file);
    free(tree_filename);
}

//Added by MEGOLDS August, 2012: subsampling
//Modified by MEGOLDS September, 2012
// Sample source without replacement to produce subsample of given size
static CV_Subset *sample_without_replacement(CV_Subset *src, int size) {

    // allocate subsample (caller must free)
    CV_Subset *dest = (CV_Subset *)e_calloc(1, sizeof(CV_Subset));

    // copy metadata and attribute info to subsample
    // (caller must free memory allocated by copy_subset_meta)
    copy_subset_meta(*src, dest, size);
    copy_subset_data(*src, dest);

    // set size of subsample
    dest->meta.num_examples = size;

    // allocate index of data point IDs
    int num_examples = src->meta.num_examples;
    int *indices = (int *)e_calloc(num_examples, sizeof(int));
    int i;
    for (i = 0; i < num_examples; i++) {
        indices[i] = i;
    }

    // select examples without replacement
    int k = 0;
    int pop_size = num_examples;
    while (k < size) {
        i = (int)(lrand48() % pop_size);
        int dataPointID = indices[i];

        // Copy chosen example into our sample.
        // Ideally, we could just use a pointer alias, but that won't
        // work b/c CV_Subset stores an array of examples, not an
        // array of pointers to examples.  Next best option is a
        // shallow copy.
        dest->examples[k] = src->examples[dataPointID];
        k++;

        // remove dataPointID from population index so can't select it again
        indices[i] = indices[pop_size - 1];
        pop_size--;
        //assert(pop_size > 0 || k == size);
    }

    // free data point indices
    free(indices);

    // return the subsample
    return dest;
}

// Added by MEGOLDS, September 2012: subsampling
void free_copied_CV_Subset(CV_Subset *sub) {
    // free items allocated by sample_without_replacement
    free(sub->meta.num_examples_per_class);
    // NB: because sample_without_replacement() uses shallow copy for
    // data examples, do not free the attribute values.  That would cause a
    // double free.
    //int i;
    //for (i = 0; i < sub->meta.num_examples; i++) {
    //    free(sub->examples[i].distinct_attribute_values);
    //}
    free(sub->examples);
    free(sub->high);
    free(sub->low);
    free(sub->discrete_used);
}

//Modified by DACIESL June-03-08: Laplacean Estimates
//Modified stop(data, args) == true case to fill new class_count and class_prob variables at each node
//Modified to handle collapse cases as well
//Modified by MEGOLDS August, 2012: subsampling
//Allows subsampling before call to find_best_split
void build_tree(CV_Subset *data, DT_Node **tree, Tree_Bookkeeping *Books, Args_Opts args) {
    int i, j;
    int returned_high, returned_low;
    int this_node = Books->current_node;
    float total;
    
    // Check if we're supposed to stop
    if (stop(data, args)) {
        (*tree)[this_node].branch_type = LEAF;
        (*tree)[this_node].num_branches = 0;
        (*tree)[this_node].Node_Value.class_label = find_best_class(data);
        (*tree)[this_node].num_errors = errors_guessing_best_class(data);
        ((*tree)[this_node]).class_count = find_class_count(data);
	((*tree)[this_node]).class_probs = find_class_probs(data, ((*tree)[this_node]).class_count);

        //printf(":::Stopping with leaf of class %d\n", (*tree)[this_node].Node_Value.class_label);
    } else {
        // MEGOLDS: Add in support for subsampling
        CV_Subset *data_sample = data;
        Boolean use_subsampling = 0 < args.subsample 
            && args.subsample < data->meta.num_examples;
        if (use_subsampling) {
            data_sample = sample_without_replacement(data, args.subsample);
        }

        // Find the best split ...
        if (args.random_forests) {
            find_random_forest_split(data_sample, &(*tree)[this_node], &returned_high, &returned_low, args);
        }
        //else if (args.random_attributes)
        //    find_random_attribute_split(data_sample, &(*tree)[this_node], &returned_high, &returned_low, args);
        else if (args.totl_random_trees)
            find_trt_split(data_sample, &(*tree)[this_node], &returned_high, &returned_low, args);
        else if (args.extr_random_trees)
            find_ert_split(data_sample, &(*tree)[this_node], &returned_high, &returned_low, args);
        else {
            find_best_split(data_sample, &(*tree)[this_node], &returned_high, &returned_low, args);
        }

        // MEGOLDS: Add in support for subsampling
        if (use_subsampling) {
            free_copied_CV_Subset(data_sample);
        }
        data_sample = NULL;

        //printf("SPLIT node=%d\n", this_node);
        //printf("      branch_type=%s\n", (*tree)[this_node].branch_type==BRANCH?"BRANCH":"LEAF");
        //printf("      attribute=%d\n", (*tree)[this_node].attribute);
        //printf("      num_branches=%d\n", (*tree)[this_node].num_branches);

        if ((*tree)[this_node].branch_type == BRANCH) {
            if (0 && args.debug)
                printf(":::Split on att %d at %.6g (%d/%d)\n", (*tree)[this_node].attribute,
                       (*tree)[this_node].branch_threshold, returned_low, returned_high);
            // Create a CV_Subset for each branch
            CV_Subset *branch_data;
            //printf("malloc branch_data to %d\n", (*tree)[this_node].num_branches);
            branch_data = (CV_Subset *)calloc((*tree)[this_node].num_branches, sizeof(CV_Subset));
            
// NOTE: Can I do these one at a time so I don't have to have them all in memory at the same time?
//       Can I more accurately size them for the number of examples in the branch rather than allocating num_examples?

            // Initialize the branch datasets by malloc'ing and pointing to values that won't be modified
            for (i = 0; i < (*tree)[this_node].num_branches; i++)
                copy_subset_meta(*data, &branch_data[i], data->meta.num_examples);
            
            // Assign each example to the appropriate branch
            if ((*tree)[this_node].attribute_type == CONTINUOUS) {
                for (i = 0; i < data->meta.num_examples; i++) {
                    // < threshold goes left; >= threshold goes right
                    if (data->examples[i].distinct_attribute_values[(*tree)[this_node].attribute] <
                                                                        (float)(returned_low + returned_high)/2.0) {
                        branch_data[0].examples[branch_data[0].meta.num_examples] = data->examples[i];
                        branch_data[0].meta.num_examples++;
                    } else {
                        branch_data[1].examples[branch_data[1].meta.num_examples] = data->examples[i];
                        branch_data[1].meta.num_examples++;
                    }
                }
            } else if ((*tree)[this_node].attribute_type == DISCRETE) {
                for (i = 0; i < data->meta.num_examples; i++) {
                    int b = data->examples[i].distinct_attribute_values[(*tree)[this_node].attribute];
                    //printf("Example %d Attribute %d DistinctValue %d\n", i, (*tree)[this_node].attribute, b);
                    branch_data[b].examples[branch_data[b].meta.num_examples] = data->examples[i];
                    branch_data[b].meta.num_examples++;
                }
                //printf("Split up %d examples into: %d\n", data->meta.num_examples, branch_data[0].num_examples);
                //for (i = 1; i < (*tree)[this_node].num_branches; i++)
                //    printf("                           %d\n", branch_data[i].num_examples);
            }

            // Copy values that will be modified
            for (i = 0; i < (*tree)[this_node].num_branches; i++) {
                for (j = 0; j < data->meta.num_attributes; j++) {
                    if (data->meta.attribute_types[j] == DISCRETE) {
                        branch_data[i].discrete_used[j] = data->discrete_used[j];
                    } else if (data->meta.attribute_types[j] == CONTINUOUS) {
                        branch_data[i].low[j] = data->low[j];
                        branch_data[i].high[j] = data->high[j];
                    }
                }
            }
            
            // Update attribute's high/low or discrete_used
            if ((*tree)[this_node].attribute_type == CONTINUOUS) {
                branch_data[1].low[(*tree)[this_node].attribute] = returned_low;
                branch_data[0].high[(*tree)[this_node].attribute] = returned_high;
            } else if ((*tree)[this_node].attribute_type == DISCRETE) {
                for (i = 0; i < (*tree)[this_node].num_branches; i++)
                    branch_data[i].discrete_used[(*tree)[this_node].attribute] = TRUE;
            }
            
            // Allocate child branches ...
            
            // First make sure we have available array elements
            while (Books->next_unused_node + (*tree)[this_node].num_branches > Books->num_malloced_nodes) {
                Books->num_malloced_nodes *= 2;
                (*tree) = (DT_Node *)realloc(*tree, Books->num_malloced_nodes * sizeof(DT_Node));
                //printf("TREE now has %d nodes\n", Books->num_malloced_nodes);
            }
            (*tree)[this_node].Node_Value.branch = (int *)malloc((*tree)[this_node].num_branches * sizeof(int));
            for (i = 0; i < (*tree)[this_node].num_branches; i++) {
                //printf("Setting branch %d from node %d to node %d\n", i, this_node, Books->next_unused_node + i);
                (*tree)[this_node].Node_Value.branch[i] = Books->next_unused_node + i;
            }
            //printf("branch[1] from node %d = %d\n", this_node, (*tree)[this_node].Node_Value.branch[1]);
            // ... and recursively call build_tree on each branch
            Books->next_unused_node += (*tree)[this_node].num_branches;
            //printf("branch[1] from node %d == %d\n", this_node, (*tree)[this_node].Node_Value.branch[1]);
            //printf("next_unused_nodes is now %d\n", Books->next_unused_node);
            
            (*tree)[this_node].num_errors = 0;
            for (i = 0; i < (*tree)[this_node].num_branches; i++) {
                //printf("Handling branch %d from node %d\n", i, this_node);
                //printf("branch[%d] from node %d ===> %d\n", i, this_node, (*tree)[this_node].Node_Value.branch[i]);
                if (branch_data[i].meta.num_examples != 0) {
                    Books->current_node = (*tree)[this_node].Node_Value.branch[i];
                    build_tree(&branch_data[i], tree, Books, args);
                    (*tree)[this_node].num_errors += (*tree)[(*tree)[this_node].Node_Value.branch[i]].num_errors;
                } else {
                    (*tree)[(*tree)[this_node].Node_Value.branch[i]].branch_type = LEAF;
                    (*tree)[(*tree)[this_node].Node_Value.branch[i]].Node_Value.class_label = find_best_class(data);
                    (*tree)[(*tree)[this_node].Node_Value.branch[i]].num_errors = 0;
                    (*tree)[(*tree)[this_node].Node_Value.branch[i]].class_count = find_class_count(data);
                    //(*tree)[(*tree)[this_node].Node_Value.branch[i]].class_probs = find_class_probs(data, (*tree)[this_node].class_count);
                    (*tree)[(*tree)[this_node].Node_Value.branch[i]].class_probs = find_class_probs(data, (*tree)[(*tree)[this_node].Node_Value.branch[i]].class_count);
                    //printf("Node %d is a leaf of class %d\n", (*tree)[this_node].Node_Value.branch[i],
                    //                        (*tree)[(*tree)[this_node].Node_Value.branch[i]].Node_Value.class_label);
                }
            }
            
            // Store the number of branches in case we collapse. Used for clean up
            int num_branches_to_clean = (*tree)[this_node].num_branches;
            
            // Determine if splits lead to identical leaves and replace with leaf
            if ((*tree)[this_node].branch_type == BRANCH &&
                (*tree)[(*tree)[this_node].Node_Value.branch[0]].branch_type == LEAF) {
                int label = (*tree)[(*tree)[this_node].Node_Value.branch[0]].Node_Value.class_label;
                i = 1;
                short collapse = TRUE;
                while (i < (*tree)[this_node].num_branches && collapse) {
                    if ((*tree)[(*tree)[this_node].Node_Value.branch[i]].branch_type == LEAF) {
                        if ((*tree)[(*tree)[this_node].Node_Value.branch[i]].Node_Value.class_label == label)
                            i++;
                        else
                            collapse = FALSE;
                    } else {
                        collapse = FALSE;
                    }
                }
                if (collapse) {
                    (*tree)[this_node].branch_type = LEAF;
                    (*tree)[this_node].num_errors = 0;
                    (*tree)[this_node].class_count = (int *)calloc(data->meta.num_classes, sizeof(int));
                    //printf("Allocate class_probs in build_tree() for node %d\n", this_node);
                    (*tree)[this_node].class_probs = (float *)calloc(data->meta.num_classes, sizeof(float));
                    total = 0.0;
                    for (i = 0; i < (*tree)[this_node].num_branches; i++) {
                        (*tree)[this_node].num_errors += (*tree)[(*tree)[this_node].Node_Value.branch[i]].num_errors;
                        for (j = 0; j < data->meta.num_classes; j++) {
			    (*tree)[this_node].class_count[j] += (*tree)[(*tree)[this_node].Node_Value.branch[i]].class_count[j];
                            total += (*tree)[(*tree)[this_node].Node_Value.branch[i]].class_count[j];
                        }
                        //free((*tree)[(*tree)[this_node].Node_Value.branch[i]].class_count);
                        //free((*tree)[(*tree)[this_node].Node_Value.branch[i]].class_probs);
                    }
                    for (j = 0; j < data->meta.num_classes; j++) {
		        (*tree)[this_node].class_probs[j] = ((*tree)[this_node].class_count[j] + 1.0)/(data->meta.num_classes + total);
                    }
                    free((*tree)[this_node].Node_Value.branch);
		    //DACIESL: moved  (*tree)[this_node].num_branches = 0; here so the above forloop is reached
                    (*tree)[this_node].num_branches = 0;
                    (*tree)[this_node].Node_Value.class_label = label;
                }
            }
            
            // Determine if the same number of errors generated by guessing the
            // the best class is >= the number of errors from branching and then
            // making the errors.  This is not a pruning step, it is a legitimate
            // concern when there is a minimum # of examples required for a branch
            
            if (args.collapse_subtree && (*tree)[this_node].branch_type == BRANCH) {
                int num_errors = errors_guessing_best_class(data);
                if ((*tree)[this_node].num_errors >= num_errors) {
                    free((*tree)[this_node].Node_Value.branch);
                    (*tree)[this_node].branch_type = LEAF;
                    (*tree)[this_node].num_branches = 0;
                    (*tree)[this_node].Node_Value.class_label = find_best_class(data);
                    (*tree)[this_node].num_errors = num_errors;
                    ((*tree)[this_node]).class_count = find_class_count(data);
                    ((*tree)[this_node]).class_probs = find_class_probs(data, ((*tree)[this_node]).class_count);
                }
            }
            
            // Clean up
            for (i = 0; i < num_branches_to_clean; i++) {
                //free_CV_Subset_inter(branch_data[i], args, TRAIN_MODE);
                free(branch_data[i].examples);
                free(branch_data[i].high);
                free(branch_data[i].low);
                free(branch_data[i].discrete_used);
                //if (branch_data[i].exo_data.num_seq_meshes > 0)
                //    free(branch_data[i].exo_data.seq_meshes);
            }
            free(branch_data);
            //free((*tree)[this_node].Node_Value.branch);
        } else {
            (*tree)[this_node].Node_Value.class_label = find_best_class(data);
            (*tree)[this_node].num_errors = errors_guessing_best_class(data);
            (*tree)[this_node].num_branches = 0;
            ((*tree)[this_node]).class_count = find_class_count(data);
            ((*tree)[this_node]).class_probs = find_class_probs(data, ((*tree)[this_node]).class_count);
            //printf(":::Leaf of class %d\n", (*tree)->Node_Value.class_label);
        }
        
    }
}

int is_pure(CV_Subset *data) {
    int i = 1;
    int pure = 1;
    while (pure == 1 && i < data->meta.num_examples)
        if (data->examples[i++].containing_class_num != data->examples[0].containing_class_num)
            pure = 0;
    return pure;
}

int stop(CV_Subset *data, Args_Opts args) {
    if (is_pure(data))
        return 1;
    if (data->meta.num_examples < args.minimum_examples * 2)
        return 1;
    
    return 0;
}

//Added by DACIESL June-03-08: Laplacean Estimates
//returns array of ints that contains count of each class at a leaf
int *find_class_count(CV_Subset *data) {
    int i;
    int *class_histo;
        
    class_histo = (int *)calloc(data->meta.num_classes, sizeof(int));
    for (i = 0; i < data->meta.num_examples; i++)
      class_histo[data->examples[i].containing_class_num]++;
    //printf("Class Counts: %d %d\n", class_histo[0], class_histo[1]);
    return class_histo;
}

//Added by DACIESL June-03-08: Laplacean Estimates
//returns array of floats that contains Laplacean Smoothed probability of each class at a leaf
float *find_class_probs(CV_Subset *data, int *class_count) {
    int i;
    float total=0.0;
    float *class_probs;
    
    //printf("Allocate class_probs in find_class_probs()\n");
    class_probs=(float *)calloc(data->meta.num_classes, sizeof(float));
    for (i = 0; i < data->meta.num_classes; i++)
        total += class_count[i];
    for (i = 0; i < data->meta.num_classes; i++)
        class_probs[i] += (class_count[i] + 1.0)/(total + data->meta.num_classes);
    //printf("Class Probs: %f %f\n", class_probs[0], class_probs[1]);
    return class_probs;
}

int find_best_class(CV_Subset *data) {
    int i;
    int best_class;
    int best_class_population;
    int *class_histo;
    
    class_histo = (int *)calloc(data->meta.num_classes, sizeof(int));
    for (i = 0; i < data->meta.num_examples; i++)
        class_histo[data->examples[i].containing_class_num]++;
    best_class = int_find_max(class_histo, data->meta.num_classes, &best_class_population);
    free(class_histo);
    return best_class;
}

int errors_guessing_best_class(CV_Subset *data) {
    int i;
    int best_class_population;
    int *class_histo;
    
    class_histo = (int *)calloc(data->meta.num_classes, sizeof(int));
    for (i = 0; i < data->meta.num_examples; i++)
        class_histo[data->examples[i].containing_class_num]++;
    int_find_max(class_histo, data->meta.num_classes, &best_class_population);
    free(class_histo);
    return data->meta.num_examples - best_class_population;
}

//Added by Cosmin June-13-20: Totally Random Split
void find_trt_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args) {
    float best_split_info;
    float split_info = NO_SPLIT;
    int best_split_high, best_split_low;
    int att_num;
    
    best_split_info = args.split_on_zero_gain ? INTMIN : 0.0;
    
    tree->branch_type = LEAF;
    tree->num_branches = 0;
    
    float cut_threshold = -1.0e100;
    for (att_num = 0; att_num < data->meta.num_attributes; att_num++) {
        split_info = best_trt_split(data, att_num, &best_split_high, &best_split_low, &cut_threshold, args);
        //printf(":::split info for att %d = %.7g\n", att_num, split_info);
        
        if (! isnan(split_info)) {
            if (split_info > best_split_info) {
                best_split_info = split_info;
                tree->branch_type = BRANCH;
                if (data->meta.attribute_types[att_num] == CONTINUOUS) {
                    // printf(":::best_split for att %d = %.7g between %f/%f\n", att_num, split_info,
                    //                                         data->float_data[att_num][best_split_low],
                    //                                         data->float_data[att_num][best_split_high]);
                    // printf(":::::::%d/%d\n", best_split_low,best_split_high);
                    tree->attribute_type   = CONTINUOUS;
                    tree->attribute        = att_num;
                    tree->branch_threshold = cut_threshold;
                    tree->num_branches     = 2;
                    *returned_low  = best_split_low;
                    *returned_high = best_split_high;
                } else if (data->meta.attribute_types[att_num] == DISCRETE) {
                    //printf(":::best_split for att %d\n", att_num);
                    tree->attribute_type = DISCRETE;
                    tree->attribute      = att_num;
                    tree->num_branches   = data->meta.num_discrete_values[tree->attribute];
                }
            }
        }
    }
    return;
}

//Added by Cosmin June-27-20: Extremely Random Split
void find_ert_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args) {
    int i, att_num;
    float best_split_info;
    float split_info = NO_SPLIT;
    int best_split_high, best_split_low;
    int num_attempted = 0;
    
    int *array;
    array = (int *)malloc(data->meta.num_attributes * sizeof(int));
    for (i = 0; i < data->meta.num_attributes; i++)
        array[i] = i;
    if (args.use_opendt_shuffle)
        _opendt_shuffle(data->meta.num_attributes, array, args.data_path);
    else
        _knuth_shuffle(data->meta.num_attributes, array);
    
    best_split_info = args.split_on_zero_gain ? INTMIN : 0.0;
    
    //*tree = (DT_Node *)malloc(sizeof(DT_Node));
    tree->branch_type  = LEAF;
    tree->num_branches = 0;
    
    att_num = 0;
    float cut_threshold = -1.0e100;
    while ( att_num < data->meta.num_attributes && num_attempted < args.extr_random_trees ) {

        split_info = best_ert_split(data, array[att_num], &best_split_high, &best_split_low, &cut_threshold, args);
        
        if (! isnan(split_info)) {
            if (split_info > best_split_info) {
                best_split_info = split_info;
                tree->branch_type = BRANCH;
                if (data->meta.attribute_types[array[att_num]] == CONTINUOUS) {
                    tree->attribute_type   = CONTINUOUS;
                    tree->attribute        = array[att_num];
                    tree->branch_threshold = cut_threshold;
                    tree->num_branches     = 2;
                    *returned_low  = best_split_low;
                    *returned_high = best_split_high;
                } else if (data->meta.attribute_types[array[att_num]] == DISCRETE) {
                    tree->attribute_type = DISCRETE;
                    tree->attribute      = array[att_num];
                    tree->num_branches   = data->meta.num_discrete_values[tree->attribute];
                }
            }
            num_attempted++;
        }
        att_num++;
    }
    
    //printf("%d,%d,%d,%d\n",att_num,data->meta.num_attributes,num_attempted,args.extr_random_trees);
    free(array);
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//added best_helinger_split option to split_method check
void find_best_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args) {
    float best_split_info;
    float split_info = NO_SPLIT;
    int best_split_high, best_split_low;
    int att_num;
    
    best_split_info = args.split_on_zero_gain ? INTMIN : 0.0;
    
    //*tree = (DT_Node *)malloc(sizeof(DT_Node));
    tree->branch_type = LEAF;
    tree->num_branches = 0;
    
    for (att_num = 0; att_num < data->meta.num_attributes; att_num++) {
        if (args.split_method == INFOGAIN) {
            split_info = best_gain_split(data, att_num, &best_split_high, &best_split_low, args);
        } else if (args.split_method == GAINRATIO) {
            split_info = best_gain_ratio_split(data, att_num, &best_split_high, &best_split_low, args);
        } else if (args.split_method == C45STYLE) {
            split_info = best_c45_split(data, att_num, &best_split_high, &best_split_low, args);
        } else {
            split_info = best_hellinger_split(data, att_num, &best_split_high, &best_split_low, args);
        }
        //printf(":::split info for att %d = %.7g\n", att_num, split_info);
        
        if (! isnan(split_info)) {
            if (split_info > best_split_info) {
                best_split_info = split_info;
                tree->branch_type = BRANCH;
                if (data->meta.attribute_types[att_num] == CONTINUOUS) {
                    //printf(":::best_split for att %d = %.7g between %f/%f\n", att_num, split_info,
                    //                                                      data->float_data[att_num][best_split_low],
                    //                                                      data->float_data[att_num][best_split_high]);
                    tree->attribute_type = CONTINUOUS;
                    tree->attribute = att_num;
                    tree->branch_threshold = (data->float_data[att_num][best_split_low] +
                                                 data->float_data[att_num][best_split_high]) / 2.0;
                    tree->num_branches = 2;
                    *returned_low = best_split_low;
                    *returned_high = best_split_high;
                } else if (data->meta.attribute_types[att_num] == DISCRETE) {
                    //printf(":::best_split for att %d\n", att_num);
                    tree->attribute_type = DISCRETE;
                    tree->attribute = att_num;
                    tree->num_branches = data->meta.num_discrete_values[tree->attribute];
                }
            }
        }
    }
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//added best_helinger_split option to split_method check
void find_random_forest_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args) {
    int i, att_num;
    float best_split_info;
    float split_info = NO_SPLIT;
    int best_split_high, best_split_low;
    int num_attempted = 0;
    
    int *array;
    array = (int *)malloc(data->meta.num_attributes * sizeof(int));
    for (i = 0; i < data->meta.num_attributes; i++)
        array[i] = i;
    if (args.use_opendt_shuffle)
        _opendt_shuffle(data->meta.num_attributes, array, args.data_path);
    else
        _knuth_shuffle(data->meta.num_attributes, array);
    
    best_split_info = args.split_on_zero_gain ? INTMIN : 0.0;
    
    //*tree = (DT_Node *)malloc(sizeof(DT_Node));
    tree->branch_type = LEAF;
    
    att_num = 0;
    while ( att_num < data->meta.num_attributes && num_attempted < args.random_forests ) {
        if (args.split_method == INFOGAIN) {
            split_info = best_gain_split(data, array[att_num], &best_split_high, &best_split_low, args);
        } else if (args.split_method == GAINRATIO) {
            split_info = best_gain_ratio_split(data, array[att_num], &best_split_high, &best_split_low, args);
        } else if (args.split_method == C45STYLE) {
            split_info = best_c45_split(data, array[att_num], &best_split_high, &best_split_low, args);
        } else {
            split_info = best_hellinger_split(data, array[att_num], &best_split_high, &best_split_low, args);
        }
        
        if (! isnan(split_info)) {
            if (split_info > best_split_info) {
                best_split_info = split_info;
                tree->branch_type = BRANCH;
                if (data->meta.attribute_types[array[att_num]] == CONTINUOUS) {
                    tree->attribute_type = CONTINUOUS;
                    tree->attribute = array[att_num];
                    tree->branch_threshold = (data->float_data[array[att_num]][best_split_low] +
                                                 data->float_data[array[att_num]][best_split_high]) / 2.0;
                    tree->num_branches = 2;
                    *returned_low = best_split_low;
                    *returned_high = best_split_high;
                } else if (data->meta.attribute_types[array[att_num]] == DISCRETE) {
                    tree->attribute_type = DISCRETE;
                    tree->attribute = array[att_num];
                    tree->num_branches = data->meta.num_discrete_values[tree->attribute];
                }
            }
            num_attempted++;
        }
        att_num++;
    }
    
    free(array);
}

/*
 * Not currently used
 *
void find_random_attribute_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args) {
    int att_num;
    int *att_array, *low_array, *high_array;
    float *info_array;
    int *index_table;
    
    att_array = (int *)malloc(data->meta.num_attributes * sizeof(int));
    low_array = (int *)malloc(data->meta.num_attributes * sizeof(int));
    high_array = (int *)malloc(data->meta.num_attributes * sizeof(int));
    info_array = (float *)malloc(data->meta.num_attributes * sizeof(float));
    
    // *tree = (DT_Node *)malloc(sizeof(DT_Node));
    tree->branch_type = LEAF;
    
    for (att_num = 0; att_num < data->meta.num_attributes; att_num++) {
        att_array[att_num] = att_num;
        
        if (args.split_method == INFOGAIN) {
            info_array[att_num] = best_gain_split(data, att_num, &high_array[att_num], &low_array[att_num], args);
        } else if (args.split_method == GAINRATIO) {
            info_array[att_num] = best_gain_ratio_split(data, att_num, &high_array[att_num], &low_array[att_num], args);
        } else {
            info_array[att_num] = best_c45_split(data, att_num, &high_array[att_num], &low_array[att_num], args);
        }
        if (isnan(info_array[att_num]))
            info_array[att_num] = INTMIN;
    }
    
    // Instead of sorting all 4 *_array arrays, use an index table to get a lookup for the Nth sorted element
    float_index_table(data->meta.num_attributes, info_array, &index_table, DESCENDING);
    
    int current_position;
    if (args.random_subspaces > 0)
        current_position = _fminf(args.random_attributes, (int)(data->meta.num_attributes * args.random_subspaces / 100.0));
    else
        current_position = _fminf(args.random_attributes, data->meta.num_attributes);
    
    // The order is swapped from OpenDT because this will check for
    // current_position == 0 before trying to access index_table[-1]
    while ( current_position > 0 && info_array[index_table[current_position-1]] < 0.0 )
        current_position--;
    if (args.split_on_zero_gain)
        while (current_position > 0 && av_eqf(info_array[index_table[current_position-1]], 0.0) )
            current_position--;
    
    // If current_position == 0 we have a leaf but branch_type has been initialized to LEAF so nothing need be done
    
    if (current_position != 0) {
        tree->branch_type = BRANCH;
        int random_number = (int)(lrand48() % current_position);
        if (data->meta.attribute_types[att_array[index_table[random_number]]] == CONTINUOUS) {
            tree->attribute_type = CONTINUOUS;
            tree->attribute = att_array[index_table[random_number]];
            tree->branch_threshold =
                (data->float_data[att_array[index_table[random_number]]][low_array[index_table[random_number]]] +
                 data->float_data[att_array[index_table[random_number]]][high_array[index_table[random_number]]])/2.0;
            tree->num_branches = 2;
            *returned_low = low_array[index_table[random_number]];
            *returned_high = high_array[index_table[random_number]];
        } else if (data->meta.attribute_types[att_array[index_table[random_number]]] == DISCRETE) {
            tree->attribute_type = DISCRETE;
            tree->attribute = att_array[index_table[random_number]];
            tree->num_branches = data->meta.num_discrete_values[tree->attribute];
        }
    }

    free(att_array);
    free(low_array);
    free(high_array);
    free(info_array);
}
 *
 */

void copy_dataset_meta(CV_Dataset src, CV_Subset *dest, int population) {
    dest->meta.num_classes = src.meta.num_classes;
    dest->meta.num_attributes = src.meta.num_attributes;
    dest->meta.num_fclib_seq = src.meta.num_fclib_seq;
    dest->meta.class_names = src.meta.class_names;
    dest->meta.attribute_names = src.meta.attribute_names;
    dest->meta.attribute_types = src.meta.attribute_types;
    dest->meta.global_offset = src.meta.global_offset;
    if (population == -1)
        dest->examples = (CV_Example *)malloc(src.meta.num_examples * sizeof(CV_Example));
    else
        dest->examples = (CV_Example *)malloc(population * sizeof(CV_Example));
    dest->high = (int *)malloc(src.meta.num_attributes * sizeof(int));
    dest->low = (int *)malloc(src.meta.num_attributes * sizeof(int));
    dest->meta.num_examples = 0;
    dest->meta.exo_data.num_seq_meshes = src.meta.exo_data.num_seq_meshes;
    dest->meta.exo_data.seq_meshes = src.meta.exo_data.seq_meshes;
    dest->meta.exo_data.assoc_type = src.meta.exo_data.assoc_type;
//    if (src.meta.exo_data.num_seq_meshes > 0) {
//        dest->meta.exo_data.seq_meshes = (FC_Mesh *)malloc(dest->meta.exo_data.num_seq_meshes * sizeof(FC_Mesh));
//        int i;
//        for (i = 0; i < dest->meta.exo_data.num_seq_meshes; i++)
//            dest->meta.exo_data.seq_meshes[i] = src.meta.exo_data.seq_meshes[i];
//        dest->meta.exo_data.assoc_type = src.meta.exo_data.assoc_type;
//    }
}

void copy_subset_meta(CV_Subset src, CV_Subset *dest, int population) {
    int i;
    
    //printf("copy sees %d examples in %d classes\n", src.meta.num_examples, src.meta.num_classes);
    dest->meta.num_classes = src.meta.num_classes;
    dest->meta.num_attributes = src.meta.num_attributes; 
    dest->meta.num_fclib_seq = src.meta.num_fclib_seq;
    dest->meta.class_names = src.meta.class_names;
    dest->meta.attribute_names = src.meta.attribute_names;
    dest->meta.attribute_types = src.meta.attribute_types;
    dest->meta.global_offset = src.meta.global_offset;
    dest->float_data = src.float_data;
    dest->meta.discrete_attribute_map = src.meta.discrete_attribute_map;
    dest->meta.num_discrete_values = src.meta.num_discrete_values;
    dest->meta.Missing = src.meta.Missing;
    //Cosmin added if statement (09/01/2020)
    if (dest->meta.num_examples_per_class != NULL)
        free(dest->meta.num_examples_per_class);
    dest->meta.num_examples_per_class = (int *)malloc(src.meta.num_classes * sizeof(int));
    if (population == -1) {
        dest->malloc_examples = src.meta.num_examples;
        dest->examples = (CV_Example *)malloc(src.meta.num_examples * sizeof(CV_Example));
        // If we will have the same number of examples, keep the per class population as is
        for (i = 0; i < src.meta.num_classes; i++)
            dest->meta.num_examples_per_class[i] = src.meta.num_examples_per_class[i];
    } else {
        dest->malloc_examples = population;
        dest->examples = (CV_Example *)malloc(population * sizeof(CV_Example));
        // If we will have a different number of examples, zero out the per class population
        // This will need to be updated elsewhere
        for (i = 0; i < src.meta.num_classes; i++)
            dest->meta.num_examples_per_class[i] = 0;
    }
    //printf("data_rs.high(e)   is at memory location 0x%lx\n", &(dest->high));
    dest->high = (int *)malloc(src.meta.num_attributes * sizeof(int));
    //printf("data_rs.high(o)   is at memory location 0x%lx\n", &(dest->high));
    dest->low = (int *)malloc(src.meta.num_attributes * sizeof(int));
    dest->discrete_used = (Boolean *)calloc(src.meta.num_attributes, sizeof(Boolean));
    dest->meta.num_examples = 0;
    dest->meta.exo_data.num_seq_meshes = src.meta.exo_data.num_seq_meshes;
    dest->meta.exo_data.seq_meshes = src.meta.exo_data.seq_meshes;
    dest->meta.exo_data.assoc_type = src.meta.exo_data.assoc_type;
//    if (src.meta.exo_data.num_seq_meshes > 0) {
//        dest->meta.exo_data.seq_meshes = (FC_Mesh *)malloc(dest->meta.exo_data.num_seq_meshes * sizeof(FC_Mesh));
//        int i;
//        for (i = 0; i < dest->meta.exo_data.num_seq_meshes; i++)
//            dest->meta.exo_data.seq_meshes[i] = src.meta.exo_data.seq_meshes[i];
//    }
    //printf("copy sees %d examples in %d classes\n", src.meta.num_examples, src.meta.num_classes);
}

// LOOK INTO THIS FUNCTION. discrete_used, low, AND high ARE NOT BEING POINTED BUT COPIED
// Raggie? They look like they are SUPPOSED to be copied to me
void point_subset_data(CV_Subset src, CV_Subset *dest) {
    int j;
    for (j = 0; j < src.meta.num_attributes; j++) {
        dest->discrete_used[j] = src.discrete_used[j];
        dest->low[j] = src.low[j];
        dest->high[j] = src.high[j];
    }
    dest->meta.num_examples = src.meta.num_examples;
    for (j = 0; j < src.meta.num_examples; j++) {
        dest->examples[j] = src.examples[j];
    }
}

void copy_subset_data(CV_Subset src, CV_Subset *dest) {
    int j;
    for (j = 0; j < src.meta.num_attributes; j++) {
        if (src.meta.attribute_types[j] == DISCRETE) {
            dest->discrete_used[j] = src.discrete_used[j];
        } else if (src.meta.attribute_types[j] == CONTINUOUS) {
            dest->low[j] = src.low[j];
            dest->high[j] = src.high[j];
        }
    }
}

void copy_example_metadata(CV_Example src, CV_Example *dest) {
    dest->global_id_num = src.global_id_num;
    dest->random_gid = src.random_gid;
    dest->fclib_seq_num = src.fclib_seq_num;
    dest->fclib_id_num = src.fclib_id_num;
}

void copy_example_data(int num_atts, CV_Example src, CV_Example *dest) {
    int i;
    dest->global_id_num = src.global_id_num;
    dest->random_gid = src.random_gid;
    dest->fclib_seq_num = src.fclib_seq_num;
    dest->fclib_id_num = src.fclib_id_num;
    dest->containing_class_num = src.containing_class_num;
    dest->containing_fold_num = src.containing_fold_num;
    dest->distinct_attribute_values = (int *)malloc(num_atts * sizeof(int));
    for (i = 0; i < num_atts; i++)
        dest->distinct_attribute_values[i] = src.distinct_attribute_values[i];
}

int check_stopping_algorithm(int init, int part_num, float raw_accuracy, int trees, float *max_raw, char *oob_filename, Args_Opts args) {
    static float **raw_accuracies;
    static float **avg_accuracies;
    static float **max_smoothed_acc;
    static float **running_avg_acc;
    static float **running_max_acc;
    static int *num_malloced;
    static float *cumulative_avg_acc;
    
    int i, j;
    
    // Init
    // In this case, part_num holds the number of partitions
    if (init == 1) {
        num_malloced = (int *)malloc(part_num * sizeof(int));
        raw_accuracies = (float **)malloc(part_num * sizeof(float *));
        avg_accuracies = (float **)malloc(part_num * sizeof(float *));
        max_smoothed_acc = (float **)malloc(part_num * sizeof(float *));
        running_avg_acc = (float **)malloc(part_num * sizeof(float *));
        running_max_acc = (float **)malloc(part_num * sizeof(float *));
        cumulative_avg_acc = (float *)calloc(part_num, sizeof(float));
        return 1;
    }
    // Unint
    if (init == -1) {
        for (i = 0; i < part_num; i++) {
            free(raw_accuracies[i]);
            free(avg_accuracies[i]);
            free(max_smoothed_acc[i]);
            free(running_avg_acc[i]);
            free(running_max_acc[i]);
        }
        free(num_malloced);
        free(raw_accuracies);
        free(avg_accuracies);
        free(max_smoothed_acc);
        free(running_avg_acc);
        free(running_max_acc);
        free(cumulative_avg_acc);
        return 1;
    }
    
    // Initialize some stuff the first time through
    if (trees == 1) {
        num_malloced[part_num] = 10 * args.build_size;
        raw_accuracies[part_num] = (float *)calloc(num_malloced[part_num], sizeof(float));
        avg_accuracies[part_num] = (float *)calloc(num_malloced[part_num], sizeof(float));
        max_smoothed_acc[part_num] = (float *)calloc(num_malloced[part_num]/args.build_size, sizeof(float));
        running_max_acc[part_num] = (float *)calloc(num_malloced[part_num], sizeof(float));
        running_avg_acc[part_num] = (float *)calloc(args.build_size, sizeof(float));
    }
    
    // Add the most current accuracy to the list
    while (trees >= num_malloced[part_num]) {
        num_malloced[part_num] *= 2;
        //printf("Realloc'ing to %d\n", num_malloced);
        raw_accuracies[part_num] = (float *)realloc(raw_accuracies[part_num], num_malloced[part_num] * sizeof(float));
        avg_accuracies[part_num] = (float *)realloc(avg_accuracies[part_num], num_malloced[part_num] * sizeof(float));
        running_max_acc[part_num] = (float *)realloc(running_max_acc[part_num], num_malloced[part_num] * sizeof(float));
        max_smoothed_acc[part_num] = (float *)realloc(max_smoothed_acc[part_num],
                                                      (num_malloced[part_num]/args.build_size) * sizeof(float));
    }
    raw_accuracies[part_num][trees-1] = raw_accuracy;
    if (args.stopping_algorithm_regtest == TRUE)
        printf("\nREGTEST %d:tree:%d raw_oob_acc:%.5f\n", part_num, trees, raw_accuracy);
    
    // If we're not supposed to check anything, just return
    if (trees % args.build_size != 0)
        return 0;
    
    // Else, check ...
    
    float local_max = 0.0;
    for (i = trees - args.build_size; i < trees; i++) {         // i = 0-19;20-39, ...
        avg_accuracies[part_num][i] = 0.0;
        if (i - args.slide_size + 1 < 0) {
            //continue;
            for (j = 0; j < i+1; j++) {
                avg_accuracies[part_num][i] = ((avg_accuracies[part_num][i] * (float)j) +
                                                                        raw_accuracies[part_num][j]) / (float)(j+1);
            }
        } else {
            for (j = i - args.slide_size + 1; j <= i; j++)   // j = (4)-(0),(3)-1,...,15-19;16-20,...
                avg_accuracies[part_num][i] += raw_accuracies[part_num][j];
            avg_accuracies[part_num][i] /= (float)args.slide_size;
        }
        // This does not keep the values in strict order but it does keep the most recent build_size values
        running_avg_acc[part_num][i%args.build_size] = avg_accuracies[part_num][i];
        running_max_acc[part_num][i] = running_avg_acc[part_num][0];
        for (j = 1; j < args.build_size; j++)
            if (running_avg_acc[part_num][j] > running_max_acc[part_num][i])
                running_max_acc[part_num][i] = running_avg_acc[part_num][j];

        if (args.stopping_algorithm_regtest == TRUE)
            printf("REGTEST %d:tree:%d avg_oob_acc:%.5f\n", part_num, i+1, avg_accuracies[part_num][i]);
        if (avg_accuracies[part_num][i] > local_max)
            local_max = avg_accuracies[part_num][i];
    }
    max_smoothed_acc[part_num][trees/args.build_size] = local_max;
    if (args.stopping_algorithm_regtest == TRUE)
        printf("REGTEST %d:max:%d:%.5f\n", part_num, trees/args.build_size,
                                           max_smoothed_acc[part_num][trees/args.build_size]);
    
    if (args.output_verbose_oob) {
        FILE *oob_file = NULL;
        if ((oob_file = fopen(oob_filename, "a")) == NULL) {
            fprintf(stderr, "Failed to open oob file for saving oob accuracies: '%s'\nExiting ...\n", oob_filename);
            exit(8);
        }
        for (i = trees - args.build_size; i < trees; i++) {
            cumulative_avg_acc[part_num] = ((cumulative_avg_acc[part_num] * (float)i) + raw_accuracies[part_num][i]) /
                                                                                                        (float)(i+1);
            fprintf(oob_file, "%5d %.6f %.6f %.6f %.6f %.6f\n", i+1, raw_accuracies[part_num][i],
                                                                cumulative_avg_acc[part_num],
                                                                avg_accuracies[part_num][i],
                                                                max_smoothed_acc[part_num][trees/args.build_size],
                                                                running_max_acc[part_num][i]);
        }
        
        fclose(oob_file);
    }
    
    // Nothing to compare (first time through) -- return 0
    if (trees/args.build_size == 1)
        return 0;
    // Max smoothed accuracy is still increasing -- return 0
    if (max_smoothed_acc[part_num][trees/args.build_size] > max_smoothed_acc[part_num][trees/args.build_size - 1])
        return 0;
    
    // Time to stop. Find the max raw accuracy
    int optimal_tree = 0;
    *max_raw = 0.0;
    for (i = trees - 2*args.build_size; i < trees - args.build_size; i++) {
        if (raw_accuracies[part_num][i] > *max_raw) {
            *max_raw = raw_accuracies[part_num][i];
            optimal_tree = i+1;
        }
    }

    return optimal_tree;
}

// fminf is not on Sparc so write my own here
// Should figure out how to use the system one if available
// But it's only used once so it's not very urgent
double _fminf(double x, double y) {
    if (x < y)
        return x;
    return y;
}

