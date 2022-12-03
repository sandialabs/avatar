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
#include <sys/errno.h>
#include <time.h>
#include "crossval.h"
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "array.h"
#include "gain.h"
#include "smote.h"
#include "balanced_learning.h"
#include "skew.h"
#include "version_info.h"
#include "attr_stats.h"
#include "reset.h"

// Included for cleanup purposes
#include "evaluate.h"
#include "bagging.h"
#include "crossval.h"

//-------------------------------------------------------------------------------------------
// 2020/08/03: Argument list changed by Cosmin Safta
// the original longer list lead to a compiler error with clang as the number of arguments
// is larger than the standard C allows. The extra three arguments are not currently
// needed and are thus commented out both here as well as in avatar_dt.h and dt.c
//-------------------------------------------------------------------------------------------
//int avatar_dt (int argc, char **argv, char* names_file, char* tree_file, char* test_data_file) {
int avatar_dt (int argc, char **argv) {
    int i;
    CV_Dataset Train_Dataset = {0}, Test_Dataset = {0};
    CV_Subset Train_Subset = {0}, Test_Subset = {0};
    Vote_Cache Cache = {0};
    AV_SortedBlobArray Train_Sorted_Examples = {0}, Test_Sorted_Examples = {0};
    CV_Partition Partitions = {0};
    DT_Ensemble *Test_Ensembles = NULL;
    Test_Ensembles = (DT_Ensemble *)calloc(1, sizeof(DT_Ensemble));
    Args_Opts Args = {0};
    reset_Args_Opts(&Args);

    FC_Dataset ds = {0}, pred_prob = {0};

    Args = process_opts(argc, argv);
    Args.caller = AVATARDT_CALLER;

    if (Args.names_file_is_a_string == TRUE) {
        //Args.names_string = names_file;
        fprintf(stderr, "ERROR: Undefined behavior in avatar_dt: names_file_is_a_string should be FALSE\n");
        exit(0);
    }

    if (Args.test_file_is_a_string == TRUE) {
        //Args.test_string = test_data_file;
        fprintf(stderr, "ERROR: Undefined behavior in avatar_dt: test_file_is_a_string should be FALSE\n");
        exit(0);
    }

    if (Args.trees_file_is_a_string == TRUE) {
	      //Args.trees_string = tree_file;
        fprintf(stderr, "ERROR: Undefined behavior in avatar_dt: trees_file_is_a_string should be FALSE\n");
        exit(0);
    }

    if (! sanity_check(&Args))
        exit(-1);
    set_output_filenames(&Args, FALSE, FALSE);

    srand48(Args.random_seed);

    if (Args.do_training) {
        if (Args.format == EXODUS_FORMAT) {
#ifdef HAVE_AVATAR_FCLIB
          init_fc(Args);
          open_exo_datafile(&ds, Args.datafile);
#else
          av_missingFCLIB();
#endif
        }
        av_exitIfError(av_initSortedBlobArray(&Train_Sorted_Examples));
        read_training_data(&ds, &Train_Dataset, &Train_Subset, &Train_Sorted_Examples, &Args);
        if (Args.num_minority_classes > 0)
            decode_minority_class_names(Train_Subset.meta, &Args);
        if (Args.do_smote)
            smote(&Train_Subset, &Train_Subset, &Train_Sorted_Examples, Args);
        if (Args.do_balanced_learning) {
            assign_bl_clump_numbers(Train_Subset.meta, Train_Sorted_Examples, Args);
            // Uncomment the next line to print the data for each clump
            // This lets you run avatardt on each clump individually
            //__print_bl_clumps(-1, Train_Subset, Args);
        }
    }

    if (Args.do_testing) {
        if (Args.format == EXODUS_FORMAT && ! Args.do_training) {
#ifdef HAVE_AVATAR_FCLIB
            init_fc(Args);
            open_exo_datafile(&ds, Args.datafile);
#else
            av_missingFCLIB();
#endif
        }
        av_exitIfError(av_initSortedBlobArray(&Test_Sorted_Examples));
        read_testing_data(&ds, Train_Subset.meta, &Test_Dataset, &Test_Subset, &Test_Sorted_Examples, &Args);
        if (Args.partitions_filename != NULL) {
            // Read parition file if it was specified
            read_partition_file(&Partitions, &Args);
        } else {
            // Else use the single datafile as the single partition
            Partitions.num_partitions = 1;
            Partitions.partition_datafile = (char **)malloc(sizeof(char *));
            Partitions.partition_data_path = (char **)malloc(sizeof(char *));
            Partitions.partition_base_filestem = (char **)malloc(sizeof(char *));
            Partitions.partition_datafile[0] = av_strdup(Args.datafile);
            Partitions.partition_data_path[0] = av_strdup(Args.data_path);
            Partitions.partition_base_filestem[0] = av_strdup(Args.base_filestem);
        }
        if (Partitions.num_partitions > 1)
            Test_Ensembles = (DT_Ensemble *)realloc(Test_Ensembles, Partitions.num_partitions * sizeof(DT_Ensemble));
    }

    if (Args.do_training)
        late_process_opts(Train_Dataset.meta.num_attributes, Train_Dataset.meta.num_examples, &Args);
    else if (Args.do_testing)
        late_process_opts(Test_Dataset.meta.num_attributes, Test_Dataset.meta.num_examples, &Args);

    if ((!Args.do_training) && Args.output_laplacean)
        check_tree_version(-1, &Args);

    // Init the stopping algorithm
    // We are now reporting OOB accuracy anytime bagging or ivoting is used
    if (Args.do_ivote == TRUE || Args.do_bagging == TRUE)
        check_stopping_algorithm(1, 1, 0.0, 0, NULL, NULL, Args);
    display_dt_opts(stdout, "", Args, Train_Subset.meta);

    //num_log_comps = 0;
    //num_pot_log_comps = 0;
    //max_log_lookup = -1;

#ifdef HAVE_AVATAR_FCLIB
    if (Args.output_predictions)
        if (Args.format == EXODUS_FORMAT)
            init_predictions(Test_Subset.meta, &pred_prob, -1, Args);
#endif

    if (Args.do_training) {
        // Train
        if (Args.do_ivote) {
            // If we're not testing now, create a stub testing subset
            if (! Args.do_testing)
                Test_Subset.meta.num_examples = 0;
            //printf("Picks %ld\n", lrand48());
            train_ivote(Train_Subset, Test_Subset, -1, &Cache, Args);
        } else {
            DT_Ensemble Train_Ensemble;
            reset_DT_Ensemble(&Train_Ensemble);
            train(&Train_Subset, &Train_Ensemble, -1, Args);

            // May need to re-read the ensemble for testing.
            // We'll delete the ensemble file later if --no-save-trees was requested
            save_ensemble(Train_Ensemble, Train_Subset.meta, -1, Args, Train_Subset.meta.num_classes);
            free_DT_Ensemble(Train_Ensemble, TRAIN_MODE);
            memset(&Train_Ensemble, 0, sizeof(DT_Ensemble));
        }
    }

    if (Args.do_testing) {
        // Test
        if (Args.do_ivote && Args.do_training) {
            test_ivote(Test_Subset, Cache, pred_prob, -1, Args);
        } else {
            // Save some parameters
            char *df;
            char *bf;
            char *dp;
            char *tf;
            df = av_strdup(Args.datafile);
            bf = av_strdup(Args.base_filestem);
            dp = av_strdup(Args.data_path);
            tf = av_strdup(Args.test_file);
            // Read all partition tree files
            for (i = 0; i < Partitions.num_partitions; i++) {
                Args.datafile = av_strdup(Partitions.partition_datafile[i]);
                Args.base_filestem = av_strdup(Partitions.partition_base_filestem[i]);
                Args.data_path = av_strdup(Partitions.partition_data_path[i]);
                if (Args.partitions_filename != NULL)
                    set_output_filenames(&Args, TRUE, TRUE);
                else
                    set_output_filenames(&Args, FALSE, FALSE);
                reset_DT_Ensemble(&Test_Ensembles[i]);
                read_ensemble(&Test_Ensembles[i], -1, 0, &Args);
                //check_ensemble_validity("Test_Ensemble",&Test_Ensembles[i]);
            }
            // Restore
            Args.datafile = av_strdup(df);
            Args.base_filestem = av_strdup(bf);
            Args.data_path = av_strdup(dp);
            Args.test_file = av_strdup(tf);
            free(df);
            free(bf);
            free(dp);
            free(tf);
            test(Test_Subset, Partitions.num_partitions, Test_Ensembles, pred_prob, -1, Args);
        }
    }

    //printf("Size of log lookup array: %d\n", max_log_lookup);
    //printf("Actual/Potential Log Computations: %d/%d (%.4f%%)\n", num_log_comps, num_pot_log_comps, 100.0*num_log_comps/num_pot_log_comps);

    // Clean up
    if (Args.do_training) {
        free_CV_Subset(&Train_Subset, Args, TRAIN_MODE);
        free_CV_Dataset(Train_Dataset, Args);
        av_freeSortedBlobArray(&Train_Sorted_Examples);
        if (! Args.save_trees && remove(Args.trees_file) < 0) {
            if (errno == EACCES)
                fprintf(stderr, "WARNING: Ensemble file '%s' not removed due to permission problems\n", Args.trees_file);
            else if (errno != ENOENT) // The file wasn't there; perhaps we are ivoting so it wasn't saved
                fprintf(stderr, "WARNING: Ensemble file '%s' not removed (%d)\n", Args.trees_file, errno);
        }
    }

    if (Args.do_testing) {
        free_CV_Subset(&Test_Subset, Args, TEST_MODE);
        free_CV_Dataset(Test_Dataset, Args);
        av_freeSortedBlobArray(&Test_Sorted_Examples);
    }
    if (Args.do_ivote) {
        free_Vote_Cache(Cache, Args);
    } else {
        //free_DT_Ensemble(Train_Ensemble, (Args.do_training ? TRAIN_MODE : TEST_MODE));
    }

    // Cleanup static allocations
    CV_Matrix m = {0};
    printf("Cleaning up static allocations\n");
    make_bag(NULL, NULL, Args, 1);
    find_best_class_from_matrix(0, m, Args, 0, 1);

    free_Args_Opts(Args);
    // free the log lookup table
    dlog_2_int(-1);

    #ifdef HAVE_AVATAR_FCLIB
    if (Args.format == EXODUS_FORMAT) {
        fc_deleteDataset(ds);
        if (Args.output_predictions)
            fc_deleteDataset(pred_prob);
        fc_finalLibrary();
    }
    #endif
    return 0;
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added HELLINGER to 's' flag
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added --output-laplacean flag
//Modified by MAMUNSO Sept'12: added --subsample option
void display_usage( void ) {
    printf("\navatardt ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: avatardt --train --test options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME  : Data format. FMTNAME is either 'exodus' or 'avatar'\n");
    printf("                            Default = 'exodus'\n");
    printf("    --train and/or --test : Perform training and/or testing\n");
    printf("\n");
    printf("required exodus-specific arguments:\n");
    printf("    -d, --datafile=FILE     : Use FILE as the exodus datafile\n");
    printf("    --train-times=R         : Use data from the range of times R to train (e.g. 1-4,6)\n");
    printf("                              Implies --train option\n");
    printf("    --test-times=R          : Use data from the range of times R to test (e.g. 5,7-10)\n");
    printf("                              Implies --test option\n");
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
    printf("exodus-specific options:\n");
    printf("    -P, --partition-file=FILE : Use FILE as the partition file\n");
    printf("                                The partition file contains one exodus filename per line\n");
    printf("required avatar-specific arguments:\n");
    printf("    -f, --filestem=STRING : Use STRING as the filestem\n");
    printf("\n");
    printf("avatar-specific options:\n");
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
    printf("    -B, --bagging=N           : Bagging with N%% of training set examples\n");
    printf("                                Default = 100\n");
    printf("    -F, --random-forests=N    : Random Forests splitting on the best N attributes\n");
    printf("                                Default = lg(num_attributes) + 1\n");
    printf("    -S, --random-subspaces=N  : Random Subspaces with N%% of attributes\n");
    printf("                                Default = 50\n");
    printf("    -T, --totl-random-trees   : Totally random trees\n");
    printf("    -E, --extr-random-trees=N : Extremely random trees splitting on N attributes\n");
    printf("                                Default = lg(num_attributes) + 1\n");
    printf("    -I, --ivoting             : IVoting with defaults of bites of size 50 and 100 iterations\n");
    printf("    --bite-size=N             : Override the default bite size for ivoting\n");
    printf("                                Implies --ivoting\n");
    printf("    --ivote-p-factor=F        : Override the default p-factor (0.75) for ivoting\n");
    printf("\n");
    printf("ensemble combination options:\n");
    printf("    --do-mmv  : Do mass majority voting\n");
    printf("    --do-emv  : Do ensemble majority voting\n");
    printf("    --do-memv : Do margin ensemble majority voting\n");
    printf("    --do-pmv  : Do probability majority voting\n");
    printf("    --do-spmv : Do scaled probability majority voting\n");
    printf("\n");
    printf("skew correction:\n");
    printf("    --majority-bagging        : Use majority bagging\n");
    printf("    --majority-ivoting        : Use majority ivoting\n");
    printf("    NOTE: The bag/bite size is set automatically to ensure the desired proportions. In the\n");
    printf("          future, setting the bag/bite size on the command line will be allowed.\n");
    printf("    --balanced-learning       : Use balanced learning\n");
    printf("    --boosting                : Use AdaBoost\n");
    printf("    --smote=TYPE              : Use SMOTE. Type is OPEN (default) or CLOSED\n");
    printf("    --smoteboost              : Use AdaBoost and closed SMOTE\n");
    printf("    -k, --nearest-neighbors=N : The number of nearest neighbors to consider.\n");
    printf("                                Default = 5\n");
    printf("    --distance-type=N         : The distance type.\n");
    printf("                                N=2 is Euclidean (default). N=1 is Manhattan.\n");
    printf("    NOTE: For all skew correction methods except boosting, the following two options\n");
    printf("          are required.\n");
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
