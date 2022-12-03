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
#include <mpi.h>
#include "crossval.h"
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "array.h"
#include "gain.h"
#include "mpiL.h"
#include "version_info.h"

void make_train_test_subsets2(CV_Subset full, int fold_pop, int fold_num, CV_Subset *train_subset, CV_Subset *test_subset);

#ifdef HAVE_AVATAR_FCLIB
//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_tree call
int main (int argc, char **argv) {
    int i, j;
    //int gid;
    CV_Dataset Dataset;
    CV_Subset Full_Trainset;
    CV_Class Class;
    AV_SortedBlobArray SortedExamples;
    FC_Dataset ds, pred_prob;
    AV_ReturnCode rc;
    Args_Opts Args;
    int *fold_population;
    
    int myrank;
    int mpires;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    derive_MPI_OPTIONS();
    derive_MPI_EXAMPLE();
    
    if (myrank == 0) {
        Args = process_opts(argc, argv);
        Args.caller = CROSSVALFC_CALLER;
        Args.num_test_times = Args.num_train_times;
        Args.test_times = Args.train_times;
        Args.do_testing = TRUE;
        if (! sanity_check(&Args)) {
            Args.go4it = FALSE;
        } else {
            if (Args.format == EXODUS_FORMAT) {
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
            }
            Dataset.meta.exo_data.num_seq_meshes = 0;
            Dataset.examples = (CV_Example *)malloc(sizeof(CV_Example)); // Do initial malloc so we can realloc later
            rc = av_initSortedBlobArray(&SortedExamples);
            av_exitIfError(rc);
            
            if (Args.format == EXODUS_FORMAT) {
                // Add the data for the requested timesteps
                init_fc(Args);
                open_exo_datafile(&ds, Args.datafile);
                for (i = 0; i < Args.num_train_times; i++) {
                    if (! add_exo_data(ds, Args.train_times[i], &Dataset, &Class)) {
                        fprintf(stderr, "Error adding data for timestep %d\n", Args.train_times[i]);
                        exit(-8);
                    }
                }
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
            
            // The functionality of this block is done in read_data_file() for non-exodus data
            if (Args.format == EXODUS_FORMAT) {
                // Create the integer-mapped arrays for each attribute
                create_cv_subset(Dataset, &Full_Trainset);
                // Put data into SortedBlobArray and populate distinct_attribute_values in each example
                populate_distinct_values_from_dataset(Dataset, &Full_Trainset, &SortedExamples);
            }
            
            late_process_opts(Dataset.meta.num_attributes, &Args);
            // Init the stopping algorithm
            if (Args.do_ivote == TRUE || Args.do_bagging == TRUE)
                check_stopping_algorithm(1, 1, 0.0, 0, NULL, Args);
            
        }
    }
    
    broadcast_options(&Args);
    
    if (Args.go4it == FALSE) {
        MPI_Finalize();
        return(-1);
    }

    if (myrank == 0)
        display_opts(Args);
    
    // Set up and broadcast some derived data stored in Args
    if (myrank != 0) {
        free(Args.skipped_features);
        Args.skipped_features = (int *)malloc(Args.num_skipped_features * sizeof(int));
        free(Args.train_times);
        Args.train_times = (int *)malloc(Args.num_train_times * sizeof(int));
    }
    MPI_Bcast(Args.skipped_features, Args.num_skipped_features, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(Args.train_times, Args.num_train_times, MPI_INT, AT_MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myrank != 0) {
        //printf("Rank %d seeding with %ld\n", myrank, Args.random_seed + myrank);
        if (Args.common_mpi_rand48_seed == TRUE)
            srand48(Args.random_seed);
        else
            srand48(Args.random_seed + myrank);
    }
    
    int iteration = 0;
    int fold;
    while (iteration < (Args.do_5x2_cv?5:1)) {
        
        if (myrank == 0) {
            if (Args.output_predictions)
                if (Args.format == EXODUS_FORMAT)
                    init_predictions(Full_Trainset.meta, &pred_prob, Args.do_5x2_cv?iteration:-1, Args);
        
            // Set up folds
            // Random sort example numbers for folds
            assign_class_based_folds(Args.num_folds, SortedExamples, &fold_population);
        }
        
        // Handle each fold
        // The current fold is the testing data and all else is training data
        for (fold = 0; fold < Args.num_folds; fold++) {
            CV_Subset Trainset, Testset;
            // Set up train and test sets
            make_train_test_subsets2(Full_Trainset, fold_population[fold], fold, &Trainset, &Testset);
            Testset.meta.Missing = Trainset.meta.Missing;
            
            // Train
            if (Args.do_ivote) {
                Vote_Cache Cache;
                train_ivote(Trainset, Testset, fold, &Cache, Args);
                test_ivote(Testset, Cache, pred_prob, fold + iteration*Args.num_folds, Args);
                
                free_Vote_Cache(Cache, Args);
            } else {
                DT_Ensemble Ensemble[1];
                train(&Trainset, &Ensemble[0], Args);
                if (Args.save_trees) {
                    write_tree_file_header(Ensemble[0].num_trees, Trainset.meta, fold + iteration*Args.num_folds,
                                           Args.tree_file, Args);
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
        }
        
        free(fold_population);
        
        iteration++;
    }

    // Clean up
    free_CV_Subset(&Full_Trainset, Args, TRAIN_MODE);
    free_CV_Dataset(Dataset, Args);
    free_CV_Class(Class);
    free_Args_Opts(Args);
    av_freeSortedBlobArray(&SortedExamples);
    // free the log lookup table
    dlog_2_int(-1);
    
    if (Args.format == EXODUS_FORMAT) {
        fc_deleteDataset(ds);
        fc_finalLibrary();
    }
    
    return 0;
}
#else
int main (int argc, char **argv)
{
  av_printfErrorMessage("To use MPI Avatar, install the fclib 1.6.1 source in avatar/util/ and then rebuild.");
  return -1;
}
#endif

void make_train_test_subsets2(CV_Subset full, int fold_pop, int fold_num, CV_Subset *train_subset, CV_Subset *test_subset) {
    int j, k;
    
    copy_subset_meta(full, train_subset, full.meta.num_examples - fold_pop);
    copy_subset_meta(full, test_subset, fold_pop);
    // Copy data into train and test sets
    for (j = 0; j < full.meta.num_examples; j++) {
        // If this example's fold is ==i then it's a test point. Otherwise it's a train point
        // Also update low and high for each attribute
        if (full.examples[j].containing_fold_num == fold_num) {
            test_subset->examples[test_subset->meta.num_examples++] = full.examples[j];
            for (k = 0; k < full.meta.num_attributes; k++) {
                if (test_subset->meta.num_examples == 1 && test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    test_subset->low[k] = full.low[k];
                    test_subset->high[k] = full.high[k];
                } else if (test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    if (full.low[k] < test_subset->low[k])
                        test_subset->low[k] = full.low[k];
                    if (full.high[k] > test_subset->high[k])
                        test_subset->high[k] = full.high[k];
                }
            }
        } else {
            train_subset->examples[train_subset->meta.num_examples++] = full.examples[j];
            for (k = 0; k < full.meta.num_attributes; k++) {
                if (train_subset->meta.num_examples == 1 && test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    train_subset->low[k] = full.low[k];
                    train_subset->high[k] = full.high[k];
                } else if (test_subset->meta.attribute_types[k] == CONTINUOUS) {
                    if (full.low[k] < train_subset->low[k])
                        train_subset->low[k] = full.low[k];
                    if (full.high[k] > train_subset->high[k])
                        train_subset->high[k] = full.high[k];
                }
            }
        }
    }
}

//Modified by DACIESL June-03-08: HDDT CAPABILITY
//Added HELLINGER to 's' flag
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added --output-laplacean flag
void display_usage( void ) {
    printf("\ncrossvalmpi ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: crossvalmpi -N|-X options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME : Data format. FMTNAME is either 'exodus' or 'avatar'\n"); 
    printf("                           Default = 'exodus'\n");
    printf("  Exactly one of the following two options must be specified:\n");
    printf("    -N, --folds=N        : Perform N-fold cross validation\n");
    printf("    -X, --5x2            : Perform 5x2 cross validation\n");
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
    printf("    -I, --ivoting            : IVoting with defaults of bites of size 50 and 100 iterations\n");
    printf("    --bite-size=N            : Override the default bite size for ivoting\n");
    printf("                               Implies --ivoting\n");
    printf("    --ivote-p-factor=F       : Override the default p-factor (0.75) for ivoting\n");
    printf("\n");
    printf("output options:\n");
    printf("    --output-accuracies       :   Print average individual and voted accuracies\n");
    printf("                                  on stdout\n");
    printf("                                  On by default\n");
    printf("    --no-output-accuracies    :   Turn off the printing of accuracies\n");
    printf("    --output-performance-metrics : Print additional performance metrics on a per\n");
    printf("                                   class and overall basis\n");
    printf("    --output-predictions      :   For exodus data: create an exodus file with\n");
    printf("                                  truth and predictions.\n");
    printf("                                  For avatar data: create a text file with a single\n");
    printf("                                  column containing the predicted class\n");
    printf("    --output-probabilities=TYPE : TYPE is weighted or unweighted (true)\n");
    printf("                                  For exodus data: add class probabilities to the\n");
    printf("                                  predictions file.\n");
    printf("                                  For avatar data: create a text file with the class\n");
    printf("                                  probabilities.\n");
    printf("    --output-confusion-matrix :   Print the confusion matrix on stdout.\n");
    printf("\n");
    printf("miscellaneous options:\n");
    printf("    --seed=N      : Use the integer N as the random number generator seed\n");
    printf("    -v, --verbose : Increase verbosity of fclib functions.\n");
    //printf("    --write|-w    : Write the *.itest and *.idata files\n");
    exit(-1);
}

//Modified by DACIESL June-03-08: HDDT CAPABILITY
//Added Hellinger to Split Method output
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added Laplacean to Output Probabilities menu
void display_opts(Args_Opts args) {
    printf("\n============================================================\n");
    
    printf("Data Format            : %s\n", args.format==EXODUS_FORMAT?"exodus":
                                            (args.format==AVATAR_FORMAT?"avatar":"unknown"));
    if (args.format == EXODUS_FORMAT) {
        printf("Data Filename          : %s\n", args.datafile);
        printf("Classes Filename       : %s\n", args.classes_filename);
        printf("Class Variable Name    : %s\n", args.class_var_name);
    }
    if (args.format == AVATAR_FORMAT) {
        printf("Filestem               : %s\n", args.base_filestem);
        printf("Truth Column           : %d\n", args.truth_column);
        if (args.num_skipped_features > 0) {
            char *range;
            array_to_range(args.skipped_features, args.num_skipped_features, &range);
            printf("Skipped Feature Numbers: %s\n", range);
            free(range);
        }
    }
    if (args.do_nfold_cv)
        printf("Number of Folds        : %d\n", args.num_folds);
    if (! args.do_ivote) {
        if (args.num_trees > 0)
            printf("Number of Trees/Fold   : %d\n", args.num_trees);
        else if (args.auto_stop == TRUE)
            printf("Number of Trees/Fold   : using stopping algorithm\n");
    }
    printf("Crossval Type          : %s\n", args.do_5x2_cv?"5x2":
                                           (args.do_nfold_cv?"N-fold":"N/A"));
    if (args.random_seed != 0)
        printf("Seed for *rand48       : %ld\n", args.random_seed);
    if (args.format == EXODUS_FORMAT) {
        char *range;
        array_to_range(args.train_times, args.num_train_times, &range);
        printf("Timeteps               : %s\n", range);
        free(range);
    } else {
        printf("Training Data          : %s/%s.data\n", args.data_path, args.base_filestem);
    }
    //printf("Write Fold Files       : %s\n", args.write_folds?"TRUE":"FALSE");
    printf("Verbosity              : %d\n", args.verbosity);
    printf("Split On Zero Gain     : %s\n", args.split_on_zero_gain?"TRUE":"FALSE");
    printf("Collapse Subtrees      : %s\n", args.collapse_subtree?"TRUE":"FALSE");
    printf("Dynamic Bounds         : %s\n", args.dynamic_bounds?"TRUE":"FALSE");
    printf("Minimum Examples       : %d\n", args.minimum_examples);
    printf("Split Method           : %s\n", args.split_method==C45STYLE?"C45 Style":
                                           (args.split_method==INFOGAIN?"Information Gain":
                                           (args.split_method==GAINRATIO?"Gain Ratio":
	       				   (args.split_method==HELLINGER?"Hellinger Distance":"N/A"))));
    printf("Save Trees             : %s\n", args.save_trees?"TRUE":"FALSE");
    if (args.random_forests > 0)
        printf("Random Forests         : %d\n", args.random_forests);
    //printf("Random Attributes      : %d\n", args.random_attributes);
    if (args.random_subspaces > 0.0)
        printf("Random Subspaces       : %f\n", args.random_subspaces);
    if (args.bag_size > 0.0)
        printf("Bagging                : %.2f%%\n", args.bag_size);
    if (args.do_ivote) {
        printf("IVoting                : %s\n", args.do_ivote?"TRUE":"FALSE");
        printf("Bite Size              : %d\n", args.bite_size);
        if (args.num_trees > 0)
            printf("Number of Trees        : %d\n", args.num_trees);
        else if (args.auto_stop == TRUE)
            printf("Number of Trees        : using stopping algorithm\n");
    }
    printf("Output Accuracies      : %s\n", args.output_accuracies==ON?"TRUE":"FALSE");
    printf("Output Predictions     : %s\n", args.output_predictions?"TRUE":"FALSE");
    printf("Output Probabilities   : %s\n", args.output_laplacean?"Weighted":(args.output_probabilities?"Unweighted":"None"));
    if (args.output_probabilities_warning) {
        printf("                         Warning: Old style input tree\n");
        printf("                         Unweighted calculation used\n");
    }
    printf("Output Confusion Matrix: %s\n", args.output_confusion_matrix?"TRUE":"FALSE");
    
    printf("============================================================\n");
}

