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
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <libgen.h>
#include "crossval.h"
#include "reset.h"
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "array.h"
#include "gain.h"
#include "skew.h"
#include "smote.h"
#include "balanced_learning.h"
#include "version_info.h"

//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_tree call
int main (int argc, char **argv) {

  int i, j;
  //int gid;
  CV_Dataset Dataset;
  CV_Subset  Full_Trainset;
  CV_Class   Class;
  AV_SortedBlobArray SortedExamples;
  AV_ReturnCode rc;
  Args_Opts Args;
  int *fold_population;
  FC_Dataset ds, pred_prob;

  // Cosmin: reset structures
  reset_CV_Dataset(&Dataset);
  reset_CV_Subset(&Full_Trainset);
  reset_CV_Class(&Class);

  Args = process_opts(argc, argv);
  Args.caller = CROSSVALFC_CALLER;
  Args.num_test_times = Args.num_train_times;
  Args.test_times = Args.train_times;
  Args.do_testing = TRUE;
  if (! sanity_check(&Args))
    exit(-1);
  set_output_filenames(&Args, FALSE, FALSE);

    if (Args.format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Read .classes file for exodus datasets
        read_classes_file(&Class, &Args);
        // Let command line -V option override class file or set Args value if not set
        if (Args.class_var_name != NULL)
            Class.class_var_name = av_strdup(Args.class_var_name);
        else
            Args.class_var_name = av_strdup(Class.class_var_name);
        Dataset.meta.num_classes = Class.num_classes;
        Dataset.meta.class_names = Class.class_names;
    
        // Initialize some stuff
        Dataset.meta.num_fclib_seq = 0;
        #else
        av_printfErrorMessage("To use EXODUS_FORMAT, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
        #endif
    }    
    Dataset.meta.exo_data.num_seq_meshes = 0;
    Dataset.examples = (CV_Example *)malloc(sizeof(CV_Example)); // Do initial malloc so we can realloc later
    rc = av_initSortedBlobArray(&SortedExamples);
    av_exitIfError(rc);
    
    if (Args.format == EXODUS_FORMAT) {
        #ifdef HAVE_AVATAR_FCLIB
        // Add the data for the requested timesteps
        init_fc(Args);
        open_exo_datafile(&ds, Args.datafile);
        for (i = 0; i < Args.num_train_times; i++) {
            if (! add_exo_data(ds, Args.train_times[i], &Dataset, &Class, (i==0?TRUE:FALSE))) {
                fprintf(stderr, "Error adding data for timestep %d\n", Args.train_times[i]);
                exit(-8);
            }
        }
        // Create the integer-mapped arrays for each attribute
        create_cv_subset(Dataset, &Full_Trainset);
        // Put data into SortedBlobArray and populate distinct_attribute_values in each example
        populate_distinct_values_from_dataset(Dataset, &Full_Trainset, &SortedExamples);
        #endif
    } else {
        if (! read_names_file(&Dataset.meta, &Class, &Args, TRUE)) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
        }
        if (! read_data_file(&Dataset, &Full_Trainset, &Class, &SortedExamples, "data", Args)) {
            fprintf(stderr, "Error reading data file\n");
            exit(-8);
        }
    }
    //cv_class_print(Class);
    //cv_dataset_print(Dataset, "Data");
    
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    display_cv_opts(stdout, "", Args);
    if (Args.num_minority_classes > 0)
        decode_minority_class_names(Full_Trainset.meta, &Args);
    
    //num_log_comps = 0;
    //num_pot_log_comps = 0;
    //max_log_lookup = -1;
    
    if (Args.do_rigorous_strat == TRUE) {
        // Make sure there are enough examples in each class to populate the folds.
        // If there are any examples at all, there should be at least the same as the number of folds for >2 folds.
        // This means at least one example per fold which means at least one for testing and at least 2 for
        // training which is ok for closed SMOTE.
        // For 2 folds, there needs to be at least 4 examples for closed SMOTE and at least 2 for open SMOTE.
        int num_inadequate_classes = 0;
        for (i = 0; i < Full_Trainset.meta.num_classes; i++) {
            int min = Args.num_folds;
            if (Args.num_folds == 2 && Args.do_smote == TRUE && Args.smote_type == CLOSED_SMOTE)
                min = 4;
            if (Full_Trainset.meta.num_examples_per_class[i] > 0 && Full_Trainset.meta.num_examples_per_class[i] < min) {
                num_inadequate_classes++;
                fprintf(stderr, "ERROR: There are not enough examples in the %s class to populate all folds\n",
                                Full_Trainset.meta.class_names[i]);
            }
        }
        if (num_inadequate_classes > 0) {
            fprintf(stderr, "To circumvent these errors, supply the --no-rigorous-strat command line option\n");
            exit(-1);
        }
    }
    
    int iteration = 0;
    int fold;
    
    while (iteration < (Args.do_5x2_cv?5:1)) {
        
        if (Args.output_predictions) {
            if (Args.format == EXODUS_FORMAT) {
                #ifdef HAVE_AVATAR_FCLIB
                init_predictions(Full_Trainset.meta, &pred_prob, Args.do_5x2_cv?iteration:-1, Args);
                if (iteration == 4 && Args.do_5x2_cv == TRUE) {
                    // This is for the overall pred file for 5x2
                    init_predictions(Full_Trainset.meta, &pred_prob, -1, Args);
                }
                #endif
            }
        }
        
        // Set up folds
        // Random sort example numbers for folds
        assign_class_based_folds(Args.num_folds, SortedExamples, &fold_population);

        // Handle each fold
        // The current fold is the testing data and all else is training data
        for (fold = 0; fold < Args.num_folds; fold++) {
            //fold = 1;
            
            // Init the stopping algorithm
            if (Args.do_ivote == TRUE || Args.do_bagging == TRUE)
                check_stopping_algorithm(1, 1, 0.0, 0, NULL, NULL, Args);
            
            // Need a SortedBlobArray associated with the training data
            AV_SortedBlobArray FoldExamples;
            av_exitIfError(av_initSortedBlobArray(&FoldExamples));
            CV_Subset Trainset, Testset;
            //Cosmin added reset functions
            reset_CV_Subset(&Trainset);
            reset_CV_Subset(&Testset);
            
            // Set up train and test sets
            make_train_test_subsets(Full_Trainset, fold_population[fold], fold, &Trainset, &Testset, &FoldExamples);
            Testset.meta.Missing = Trainset.meta.Missing;
            if (Args.do_smote == TRUE) {
                update_actual_att_props(0, Trainset.meta, &Args);
                smote(&Trainset, &Trainset, &FoldExamples, Args);
                // Uncomment the next line to print the data for the current fold
                // This lets you run avatardt on each fold individually
                //__print_smoted_data(fold, Trainset);
            }
            if (Args.do_balanced_learning == TRUE) {
                update_actual_att_props(0, Trainset.meta, &Args);
                assign_bl_clump_numbers(Trainset.meta, FoldExamples, Args);
                // Uncomment the next line to print the data for each clump
                // This lets you run avatardt on each clump individually
                //__print_bl_clumps(-1, Trainset, Args);
            }
            if (Args.majority_bagging == TRUE || Args.majority_ivoting == TRUE)
                update_actual_att_props(0, Trainset.meta, &Args);
            if (Args.majority_bagging == TRUE || Args.majority_ivoting == TRUE ||
                Args.do_balanced_learning == TRUE || Args.do_smote == TRUE) {
                printf("================================================\n");
                printf("For crossval iteration %d:\n", fold+1);
                int num_clumps, num_ex_per_clump;
                compute_number_of_clumps(Trainset.meta, &Args, &num_clumps, &num_ex_per_clump);
                if (Args.majority_bagging == TRUE) {
                    printf("Examples per Bag       : %d\n", num_ex_per_clump);
                    printf("Bagging                : %.2f%%\n", Args.bag_size);
                } else if (Args.majority_ivoting == TRUE) {
                    printf("Majority Ivoting\n");
                    printf("Bite Size              : %d\n", Args.bite_size);
                    if (Args.num_trees > 0)
                        printf("Number of Trees        : %d\n", Args.num_trees);
                    else if (Args.auto_stop == TRUE)
                        printf("Number of Trees        : using stopping algorithm\n");
                } else if (Args.do_balanced_learning == TRUE) {
                    printf("Number of Clumps       : %d\n", num_clumps);
                    printf("Examples per Clump     : %d\n", num_ex_per_clump);
                }
                print_skewed_per_class_stats(stdout, 0, Args, Trainset.meta);
                printf("================================================\n");
            }
            
            // Train
            if (Args.do_ivote) {
                Vote_Cache Cache;
                train_ivote(Trainset, Testset, fold + iteration*Args.num_folds, &Cache, Args);
                test_ivote(Testset, Cache, pred_prob, fold + iteration*Args.num_folds, Args);
                free_Vote_Cache(Cache, Args);
            } else {
                DT_Ensemble Ensemble[1];
                reset_DT_Ensemble(Ensemble);
                train(&Trainset, &Ensemble[0], fold + iteration*Args.num_folds, Args);
                if (Args.save_trees) {
                    write_tree_file_header(Ensemble[0].num_trees, Trainset.meta, fold + iteration*Args.num_folds,
                                           Args.trees_file, Args);
                    for (j = 0; j < Ensemble[0].num_trees; j++)
		                save_tree(Ensemble[0].Trees[j], fold + iteration*Args.num_folds, j+1, Args, Trainset.meta.num_classes);
                    //save_ensemble(Ensemble[0], Trainset, fold, Args);
                }
                // Test
                test(Testset, 1, Ensemble, pred_prob, fold + iteration*Args.num_folds, Args);
                free_DT_Ensemble(Ensemble[0], TRAIN_MODE);
            }
            
            free_CV_Subset_inter(&Trainset, Args, TRAIN_MODE);
            free_CV_Subset_inter(&Testset, Args, TEST_MODE);
            av_freeSortedBlobArray(&FoldExamples);
            
            // Init the stopping algorithm
            if (Args.do_ivote == TRUE || Args.do_bagging == TRUE)
                check_stopping_algorithm(-1, 1, 0.0, 0, NULL, NULL, Args);
        }
        
        free(fold_population);
        
        iteration++;
    }

    // Clean up
    free_CV_Subset(&Full_Trainset, Args, TRAIN_MODE);
    free_CV_Dataset(Dataset, Args);
    free_CV_Class(Class);
    free_Args_Opts(Args);
    if (! Args.read_folds)
        av_freeSortedBlobArray(&SortedExamples);
    // free the log lookup table
    dlog_2_int(-1);
    
    #ifdef HAVE_AVATAR_FCLIB
    if (Args.format == EXODUS_FORMAT && ! Args.read_folds) {
        fc_deleteDataset(ds);
        fc_finalLibrary();
    }
    #endif
    
    return 0;
}

/*
 * Takes the full dataset and creates the training/testing datasets for this iteration.
 * All examples with a containing_fold_num == fold_num is in the testing dataset. All
 * others are in the training dataset.
 */

void make_train_test_subsets(CV_Subset full, int fold_pop, int fold_num, CV_Subset *train_subset, CV_Subset *test_subset, AV_SortedBlobArray *blob) {
    int j, k;
    AV_ReturnCode rc;
    
    copy_subset_meta(full, train_subset, full.meta.num_examples - fold_pop);
    copy_subset_meta(full, test_subset, fold_pop);
    // Copy data into train and test sets
    for (j = 0; j < full.meta.num_examples; j++) {
        // If this example's fold is ==fold_num then it's a test point. Otherwise it's a train point
        // Also update low and high for each attribute
        if (full.examples[j].containing_fold_num == fold_num) {
            // Copy over the example and update the per class population
            test_subset->examples[test_subset->meta.num_examples] = full.examples[j];
            test_subset->meta.num_examples_per_class[full.examples[j].containing_class_num]++;
            for (k = 0; k < full.meta.num_attributes; k++) {
                if (test_subset->meta.num_examples == 0 && test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    test_subset->low[k] = full.low[k];
                    test_subset->high[k] = full.high[k];
                } else if (test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    if (full.low[k] < test_subset->low[k])
                        test_subset->low[k] = full.low[k];
                    if (full.high[k] > test_subset->high[k])
                        test_subset->high[k] = full.high[k];
                }
            }
            test_subset->meta.num_examples++;
        } else {
            // Copy over the example and update the per class population
            
            // When SMOTEing, need to make a copy of the example and not just copy the pointer because the
            // distinct_attribute_values array gets modified and, by pointing, the original data is modified
            // which messes things up the second time through.
            //train_subset->examples[train_subset->meta.num_examples] = full.examples[j];
            copy_example_metadata(full.examples[j], &train_subset->examples[train_subset->meta.num_examples]);
            copy_example_data(full.meta.num_attributes, full.examples[j],
                              &train_subset->examples[train_subset->meta.num_examples]);
            train_subset->meta.num_examples_per_class[full.examples[j].containing_class_num]++;
            for (k = 0; k < full.meta.num_attributes; k++) {
                if (train_subset->meta.num_examples == 0 && test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    train_subset->low[k] = full.low[k];
                    train_subset->high[k] = full.high[k];
                } else if (test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    if (full.low[k] < train_subset->low[k])
                        train_subset->low[k] = full.low[k];
                    if (full.high[k] > train_subset->high[k])
                        train_subset->high[k] = full.high[k];
                }
            }
            rc = av_addBlobToSortedBlobArray(blob, &train_subset->examples[train_subset->meta.num_examples], cv_example_compare_by_seq_id);
            if (rc < 0) {
                av_exitIfErrorPrintf(rc, "Failed to add training example %d to SBA\n", train_subset->meta.num_examples);
            } else if (rc == 0) {
                fprintf(stderr, "Example %d already exists in SBA\n", train_subset->meta.num_examples);
            }
            train_subset->meta.num_examples++;
        }
    }
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added HELLINGER to 's' flag
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added --output-laplacean flag
void display_usage( void ) {
    printf("\ncrossvalfc ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: crossvalfc -N|-X options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME : Data format. FMTNAME is either 'exodus' or 'avatar'\n"); 
    printf("                           Default = 'exodus'\n");
    printf("  Exactly one of the following three options must be specified:\n");
    printf("    -N, --folds=N        : Perform N-fold cross validation\n");
    printf("    -X, --5x2            : Perform 5x2 cross validation\n");
    printf("        --loov           : Perform leave-one-out-validation.\n");
    printf("                           This is N-fold cross validation with the number of folds\n");
    printf("                           set to the number of samples.\n");
    printf("\n");
    printf("required exodus-specific arguments:\n");
    printf("    -d, --datafile=FILE     : Use FILE as the exodus datafile\n");
    printf("    --train-times=R         : Use data from the range of timesteps R (e.g. 1-4,6)\n");
    printf("                              (N.B. Exodus is 1-based for timesteps)\n");
    printf("    -V, --class-var=VARNAME : Use the variable named VARNAME as the class\n");
    printf("                              definition\n");
    printf("    -C, --class-file=FILE   : FILE the gives number of classes and thresholds:\n");
    printf("                              E.g.\n");
    printf("                                  class_var_name Osaliency\n");
    printf("                                  number_of_classes 5\n");
    printf("                                  thresholds 0.2,0.4,0.6,0.8\n");
    printf("                                Will put all values <=0.2 in class 0,\n");
    printf("                                <=0.4 in class 1, etc\n");
    printf("\n");
    printf("required avatar-specific arguments:\n");
    printf("    -f, --filestem=STRING : Use STRING as the filestem\n");
    printf("\n");
    printf("optional avatar arguments:\n");
    printf("    --include=R      : Include the features listed in R (e.g. 1-4,6)\n");
    printf("    --exclude=R      : Exclude the features listed in R (e.g. 1-4,6)\n");
    printf("                       --include and --exclude may be specified multiple times.\n");
    printf("                       They are applied left to right.\n");
    printf("    --truth-column=S : Location of the truth column. S = first or last\n");
    printf("                       Default = last\n");
    printf("\n");
    printf("fold-generation options:\n");
    printf("        --no-rigorous-strat      : Do not use rigorous class stratification across folds.\n");
    printf("                                   This means that folds are allowed to have zero examples of\n");
    printf("                                   one or more classes.\n");
    printf("\n");
    printf("tree-building options:\n");
    printf("    -n, --num-trees=N            : Build N trees. Default = 1\n");
    printf("        --use-stopping-algorithm : Use a stopping algorithm to determine optimal\n");
    printf("                                   number of trees in the ensemble\n");
    printf("        --slide-size             : For the stopping algorithm, the size of the averaging\n");
    printf("                                   window. Default = 5\n");
    printf("        --build-size             : For the stopping algorithm, the number of trees to\n");
    printf("                                   add at each iteration. Default = 20\n");
    printf("    -s, --split-method=METHOD    : Split criteria. METHOD is one of the following:\n");
    printf("                                       C45 (default), INFOGAIN, GAINRATIO, HELLINGER\n");
    printf("    -m, --hard-limit-bounds      : Lower bound for minimum examples to split\n");
    printf("                                   Default = 2\n");
    printf("    -z, --split-zero-gain        : Split on zero information gain or gain ratio\n");
    printf("                                   Off by default\n");
    printf("        --no-split-zero-gain     : Explicitly turn off splitting on zero info gain\n");
    printf("                                   or gain ratio\n");
    printf("    -b, --subsample=N            : Use subsample of n points to find best split, if\n");
    printf("                                   number of points at local node > n. Default: don't\n");
    printf("                                   subsample (use all data).\n");
    printf("        --collapse-subtree       : Allow subtrees to be collapsed\n");
    printf("                                   On by default\n");
    printf("        --no-collapse-subtree    : Do not allow subtrees to be collapsed.\n");
    printf("        --dynamic-bounds         : Use dynamic computation for minimum examples to\n");
    printf("                                   split\n");
    printf("                                   On by default\n");
    printf("        --no-dynamic-bounds      : Use hard lower bound set by -m option.\n");
    printf("        --save-trees             : Write trees to disk\n");
    printf("                                   On by default.\n");
    printf("        --no-save-trees          : Do not write trees to disk\n");
    printf("\n");
    printf("ensemble options:\n");
    printf("    -B, --bagging=N          : Bagging with N%% of training set examples\n");
    printf("                               Default = 100\n");
    printf("    -F, --random-forests=N   : Random Forests splitting on the best N attributes\n");
    printf("                               Default = lg(num_attributes) + 1\n");
    printf("    -S, --random-subspaces=N : Random Subspaces with N%% of attributes\n");
    printf("                               Default = 50\n");
    printf("    -T, --totl-rnd-trees     : Totally random trees\n");
    printf("    -E, --extr-rnd-trees=N   : Extremely random trees splitting on N attributes\n");
    printf("                               Default = lg(num_attributes) + 1\n");
    printf("    -I, --ivoting            : IVoting with defaults of bites of size 50 and 100 iterations\n");
    printf("    --bite-size=N            : Override the default bite size for ivoting\n");
    printf("                               Implies --ivoting\n");
    printf("    --ivote-p-factor=F       : Override the default p-factor (0.75) for ivoting\n");
    printf("\n");
    printf("skew correction:\n");
    printf("    --majority-bagging        : Use majority bagging\n");
    printf("    --majority-ivoting        : Use majority ivoting\n");
    printf("    NOTE: The bag/bite size is set automatically to ensure the desired proportions. In the\n");
    printf("          future, setting the bag/bite size on the command line will be allowed.\n");
    printf("    --balanced-learning       : Use balanced learning\n");
    printf("    --smote=TYPE              : Use SMOTE. Type is OPEN (default) or CLOSED\n");
    printf("    -k, --nearest-neighbors=N : The number of nearest neighbors to consider.\n");
    printf("                                Default = 5\n");
    printf("    --distance-type=N         : The distance type.\n");
    printf("                                N=2 is Euclidean (default). N=1 is Manhattan.\n");
    printf("    --minority-classes=L      : A comma-separated list of the minority class labels.\n");
    printf("                                This option must be specified for all skew correction methods.\n");
    printf("    --proportions=L           : A colon-separated list of the desired proportions for the\n");
    printf("                                minority classes. The order of the proportions must be the\n");
    printf("                                same as the order in the minority-classes list.\n");
    printf("                                The proportions should be entered as percentages.\n");
    printf("                                This option must be specified for all skew correction methods.\n");
    printf("\n");
    printf("alternate filenames:\n");
    printf("    --train-file=FILE       : For avatar format data, use FILE for training data\n");
    printf("    --names-file=FILE       : For avatar format data, use FILE for the names file\n");
    printf("    --test-file=FILE        : For avatar format data, use FILE for testing data\n");
    printf("    --trees-file=FILE       : Write the ensemble to FILE\n");
    printf("    --predictions-file=FILE : For avatar format data, write predictions to FILE\n");
    printf("    --oob-file=FILE         : Write the verbose oob data to FILE\n");
    printf("\n");
    printf("output options:\n");
    printf("    --output-accuracies          : Print average individual and voted accuracies\n");
    printf("                                   on stdout\n");
    printf("                                   On by default\n");
    printf("    --no-output-accuracies       : Turn off the printing of accuracies\n");
    printf("    --output-performance-metrics : Print additional performance metrics on a per\n");
    printf("                                   class and overall basis\n");
    printf("    --output-predictions         : For exodus data: create an exodus file with\n");
    printf("                                   truth and predictions.\n");
    printf("                                   For avatar data: create a text file with a single\n");
    printf("                                   column containing the predicted class\n");
    printf("    --output-probabilities=TYPE  : TYPE is weighted or unweighted (default)\n");
    printf("                                   For exodus data: add class probabilities to the\n");
    printf("                                   predictions file.\n");
    printf("                                   For avatar data: create a text file with the class\n");
    printf("                                   probabilities.\n");
    printf("    --output-confusion-matrix    : Print the confusion matrix on stdout.\n");
    printf("    --verbose-oob                : Print the out-of-bag accuracy for the ensemble after each\n");
    printf("                                   tree is added. The values are printed to a file named\n");
    printf("                                   the same as the input data file but with the extension\n");
    printf("                                   .oob-data\n");
    printf("    --output-margins             : FOR AVATAR FORMAT DATA ONLY. Store the class margin in the\n");
    printf("                                   predictions file. This option turns on --output-predictions.\n");
    printf("\n");
    printf("miscellaneous options:\n");
    printf("    --seed=N      : Use the integer N as the random number generator seed\n");
    printf("    -v, --verbose : Increase verbosity of fclib functions.\n");
    //printf("    --write|-w    : Write the *.itest and *.idata files\n");
    exit(-1);
}

