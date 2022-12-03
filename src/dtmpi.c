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
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <errno.h>
#include "av_utils.h"
#include "crossval.h"
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "treempi.h"
#include "array.h"
#include "gain.h"
#include "mpiL.h"
#include "reset.h"
#include "version_info.h"

//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_tree call
int main (int argc, char **argv) {
    int i, j;
    CV_Metadata Dataset_Meta;
    CV_Partition Partitions;
    DT_Ensemble *Ensemble;
    Args_Opts Args;
    reset_Args_Opts(&Args);
    
    FC_Dataset ds;
    FC_Dataset pred_prob;
    
    int myrank;
    int mpires;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    //derive_MPI_OPTION();
    derive_MPI_OPTIONS();
    derive_MPI_EXAMPLE();
    derive_MPI_TREENODE();
    
    reset_CV_Metadata(&Dataset_Meta);

    //printf("Running as rank %d\n", myrank);
    // Root process reads all the data and creates the training and testing data subsets
    
    if (myrank == 0)
        Args = process_opts(argc, argv);
    
    // We only do training ...
    Args.do_training = TRUE;
    
    if (myrank == 0) {
        Args.caller = AVATARMPI_CALLER;
        if (! sanity_check(&Args)) {
            Args.go4it = FALSE;
        } else if (Args.do_testing) {
            Args.go4it = FALSE;
            fprintf(stderr, "\navatarmpi is used only for training. Use avatardt for testing\n");
        } else {
            if (Args.partitions_filename != NULL) {
                // Read parition file if it was specified
                //printf("Reading the partitions file\n");
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
            set_output_filenames(&Args, FALSE, FALSE);
            Ensemble = (DT_Ensemble *)malloc(Partitions.num_partitions * sizeof(DT_Ensemble));
            for(i=0; i<Partitions.num_partitions; i++)
              reset_DT_Ensemble(&Ensemble[i]);
            //printf("Using %d partition with filename 0 '%s'\n", Partitions.num_partitions, Partitions.partition_datafile[0]);
            
            // Read the training data so that we can run late_process_opts and get num_attributes for later
            if (Args.do_training) {
                if (Args.format == EXODUS_FORMAT) {
                    init_fc(Args);
                    open_exo_datafile(&ds, Partitions.partition_datafile[0]);
                }
                read_metadata(&ds, &Dataset_Meta, &Args);
            }
            
            late_process_opts(Dataset_Meta.num_attributes, Dataset_Meta.num_examples, &Args);
            // Init the stopping algorithm
            if (Args.do_ivote == TRUE || Args.do_bagging == TRUE)
                check_stopping_algorithm(1, Partitions.num_partitions, 0.0, 0, NULL, NULL, Args);
        }
    } else {
        // Non-root ranks only need to worry about a single DT_Ensemble
        Ensemble = (DT_Ensemble *)malloc(sizeof(DT_Ensemble));
        reset_DT_Ensemble(&Ensemble[0]);
    }
    
    // Send Args to all the other processes
    // Need to do this so all processes can gracefully exit by checking Args.go4it
    broadcast_options(&Args);
    
    if (Args.go4it == FALSE) {
        MPI_Finalize();
        return(-1);
    }
    // If we're still here, set Args.mpi_rank
    Args.mpi_rank = myrank;
    if (myrank == 0)
        display_dtmpi_opts(stdout, "", Args);
    
    //printf("debug for %d = %d\n", myrank, Args.debug);
    //printf("majority_bagging for %d = %d\n", myrank, Args.majority_bagging);
    //printf("do_bagging for %d = %d\n", myrank, Args.do_bagging);
    //printf("bag_size for %d = %f\n", myrank, Args.bag_size);
    //printf("random_seed for %d = %ld\n", myrank, Args.random_seed);
    
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
    
    //num_log_comps = 0;
    //num_pot_log_comps = 0;
    //max_log_lookup = -1;
    
    if (myrank == 0)
        if (Args.output_predictions)
            if (Args.format == EXODUS_FORMAT)
                init_predictions(Dataset_Meta, &pred_prob, -1, Args);
    
    if (Args.do_training) {
        // Train
        if (Args.do_ivote) {
            // If we're not testing now, create a stub testing subset
            //if (! Args.do_testing)
            //    Test_Subset.num_examples = 0;
            //printf("Rank %d picks %ld\n", myrank, lrand48());
            train_ivote_mpi(-1, Partitions, myrank, Args);
/* DON'T WORRY ABOUT THIS FOR NOW
            // Since the root process doesn't store all the Vote_Cache details and just writes the trees to disk,
            // to test we need to read the just-written ensemble
            if (myrank == 0) {
                for (i = 0; i < Partitions.num_partitions; i++) {
                    Args.datafile = av_strdup(Partitions.partition_datafile[i]);
                    Args.base_filestem = av_strdup(Partitions.partition_base_filestem[i]);
                    Args.data_path = av_strdup(Partitions.partition_data_path[i]);
                    if (Partitions.num_partitions > 1)
                        set_output_filenames(&Args, FALSE, TRUE);
                    else
                        set_output_filenames(&Args, FALSE, FALSE);
                    read_ensemble(&Ensemble[i], -1, 0, &Args);
                    // Set the missing values to the values in the ensemble file
                }
            }
 */
        } else {
            train_mpi(Partitions, &Ensemble, myrank, Args);
            if (myrank == 0 && Args.save_trees) {
                for (i = 0; i < Partitions.num_partitions; i++) {
                    // Set up Args for this ensemble
                    Args.datafile = av_strdup(Partitions.partition_datafile[i]);
                    Args.base_filestem = av_strdup(Partitions.partition_base_filestem[i]);
                    Args.data_path = av_strdup(Partitions.partition_data_path[i]);
                    // Set up Metadata for this ensemble
                    Dataset_Meta.num_examples = Ensemble[i].num_training_examples;
                    Dataset_Meta.num_examples_per_class = (int *)malloc(Ensemble[i].num_classes * sizeof(int));
                    for (j = 0; j < Ensemble[i].num_classes; j++)
                        Dataset_Meta.num_examples_per_class[j] = Ensemble[i].num_training_examples_per_class[j];
                    Dataset_Meta.Missing = (union data_point_union *)malloc(Ensemble[i].num_attributes * sizeof(union data_point_union));
                    for (j = 0; j < Ensemble[i].num_attributes; j++) {
                        if (Ensemble[i].attribute_types[j] == DISCRETE)
                            Dataset_Meta.Missing[j].Discrete = Ensemble[i].Missing[j].Discrete;
                        else if (Ensemble[i].attribute_types[j] == CONTINUOUS)
                            Dataset_Meta.Missing[j].Continuous = Ensemble[i].Missing[j].Continuous;
                    }
                    if (Partitions.num_partitions > 1)
                        set_output_filenames(&Args, FALSE, TRUE);
                    else
                        set_output_filenames(&Args, FALSE, FALSE);
                    save_ensemble(Ensemble[i], Dataset_Meta, -1, Args, Ensemble[i].num_classes);
                }
            }
        }
    }
    
    if (myrank == 0) {
        if (Args.do_ivote) {
            //free_Vote_Cache(Cache, Args);
        } else {
            //free_DT_Ensemble(Ensemble, (Args.do_training ? TRAIN_MODE : TEST_MODE));
        }
        
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
        if (! Args.save_trees && remove(Args.trees_file) < 0) {
            if (errno == EACCES) 
                fprintf(stderr, "WARNING: Ensemble file '%s' not removed due to permission problems\n", Args.trees_file);
            else if (errno != ENOENT) // The file wasn't there; perhaps we are ivoting so it wasn't saved
                fprintf(stderr, "WARNING: Ensemble file '%s' not removed (%d)\n", Args.trees_file, errno);
        }
    } else {
    }

    MPI_Finalize();
    
    return 0;
}

//Modified by DACIESL June-03-08: HDDT CAPABILITY
//Added HELLINGER to 's' flag
void display_usage( void ) {
    printf("\navatarmpi ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: avatarmpi --format=FMTNAME format-specific-options [tree-building-options] [ensemble-options]\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME  : Data format. FMTNAME is either 'exodus' or 'avatar'\n");
    printf("                            Default = 'exodus'\n");
    printf("\n");
    printf("required exodus-specific arguments:\n");
    printf("    -d, --datafile=FILE       : Use FILE as the exodus datafile\n");
    printf("    -P, --partition-file=FILE : Use FILE as the partition file\n");
    printf("                                The partition file contains one exodus filename per line\n");
    printf("  NOTE: --partition-file overrides --datafile if both are specified\n");
    printf("    --train-times=R           : Use data from the range of times R to train (e.g. 1-4,6)\n");
    printf("                                Implies --train option\n");
    printf("    -V, --class-var=VARNAME   : Use the variable named VARNAME as the class\n");
    printf("                                definition\n");
    printf("    -C, --class-file=FILE     : FILE the gives number of classes and thresholds:\n");
    printf("                                  E.g.\n");
    printf("                                    class_var_name Osaliency\n");
    printf("                                    number_of_classes 5\n");
    printf("                                    thresholds 0.2,0.4,0.6,0.8\n");
    printf("                                  Will put all values <=0.2 in class 0,\n");
    printf("                                  <=0.4 in class 1, etc\n");
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
    printf("alternate filenames:\n");
    printf("    --train-file=FILE       : For avatar format data, use FILE for training data\n");
    printf("    --names-file=FILE       : For avatar format data, use FILE for the names file\n");
    printf("    --test-file=FILE        : For avatar format data, use FILE for testing data\n");
    printf("    --trees-file=FILE       : Write the ensemble to FILE\n");
    printf("    --predictions-file=FILE : For avatar format data, write predictions to FILE\n");
    printf("    --oob-file=FILE         : Write the verbose oob data to FILE\n");
    printf("\n");
    printf("miscellaneous options:\n");
    printf("    --seed=N      : Use the integer N as the random number generator seed\n");
    printf("    -v, --verbose : Increase verbosity of fclib functions.\n");
    //printf("    --write|-w    : Write the *.itest and *.idata files\n");
    exit(-1);
}

