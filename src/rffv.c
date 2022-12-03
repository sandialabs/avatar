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
#include "options.h"
#include "rw_data.h"
#include "util.h"
#include "memory.h"
#include "distinct_values.h"
#include "tree.h"
#include "array.h"
#include "gain.h"
#include "ivote.h"
#include "version_info.h"

//Modified by DACIESL June-04-08: Laplacean Estimates
//uses new save_ensembles calls
int main (int argc, char **argv) {
    
    int i;
    //int gid;
    CV_Dataset Dataset;
    CV_Subset Trainset, Testset;
    CV_Class Class;
    AV_SortedBlobArray SortedExamples;
    AV_ReturnCode rc;
    Vote_Cache Cache;
    DT_Ensemble Ensemble;
    Args_Opts Args;
    #ifdef HAVE_AVATAR_FCLIB
    FC_Dataset ds;
    #endif
    
    Args = process_opts(argc, argv);
    Args.caller = RFFEATUREVALUE_CALLER;
    Args.do_training = TRUE;
    // Turn on do_noising
    Args.do_noising = TRUE;
    // Turn on random forests if not on from command line
    if (Args.random_forests == 0)
        Args.random_forests = -1;
    
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
        #endif
    } else {
        if (! read_names_file(&Dataset.meta, &Class, &Args, TRUE)) {
            fprintf(stderr, "Error reading names file\n");
            exit(-8);
        }
        if (! read_data_file(&Dataset, &Trainset, &Class, &SortedExamples, "data", &Args)) {
            fprintf(stderr, "Error reading data file\n");
            exit(-8);
        }
    }
    
    #ifdef HAVE_AVATAR_FCLIB
    if (Args.format == EXODUS_FORMAT) {
        // Create the integer-mapped arrays for each attribute
        create_cv_subset(Dataset, &Trainset);
        populate_distinct_values_from_dataset(Dataset, &Trainset, &SortedExamples);
    }
    #endif
    
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    // Init the stopping algorithm
    if (Args.bag_size > 0.0 || Args.do_ivote == TRUE)
        check_stopping_algorithm(1, 1, 0.0, 0, NULL, NULL, Args);
    display_fv_opts(stdout, "", Args);
    
    //num_log_comps = 0;
    //num_pot_log_comps = 0;
    //max_log_lookup = -1;
    
    // Train
    if (Args.do_ivote) {
        train_ivote(Trainset, Testset, -1, &Cache, Args);
    } else {
        train(&Trainset, &Ensemble, -1, Args);
        if (Args.save_trees)
	    save_ensemble(Ensemble, Trainset.meta, -1, Args, Trainset.meta.num_classes);
        free_DT_Ensemble(Ensemble, TRAIN_MODE);
    }
    
    //free_CV_Subset_inter(&Trainset, Args, TRAIN_MODE);
    //free_CV_Subset_inter(&Testset, Args, TEST_MODE);

    // Clean up
    free_CV_Subset(&Trainset, Args, TRAIN_MODE);
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

void display_usage( void ) {
    printf("\nrfFeatureValue ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: rfFeatureValue options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME : Data format. FMTNAME is either 'exodus' or 'avatar'\n"); 
    printf("                           Default = 'exodus'\n");
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
    printf("                                       C45 (default), INFOGAIN, GAINRATIO\n");
    printf("    -m, --hard-limit-bounds      : Lower bound for minimum examples to split\n");
    printf("                                   Default = 2\n");
    printf("    -z, --split-zero-gain        : Split on zero information gain or gain ratio\n");
    printf("                                   Off by default\n");
    printf("        --no-split-zero-gain     : Explicitly turn off splitting on zero info gain\n");
    printf("                                   or gain ratio\n");
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
    printf("                               This option is always on with the default value but\n");
    printf("                               a different value may be specified on the command line.\n");
    printf("                               Default = lg(num_attributes) + 1\n");
//    printf("    -S, --random-subspaces=N : Random Subspaces with N%% of attributes\n");
//    printf("                               Default = 50\n");
    printf("    -I, --ivoting            : IVoting with defaults of bites of size 50 and 100 iterations\n");
    printf("    --bite-size=N            : Override the default bite size for ivoting\n");
    printf("                               Implies --ivoting\n");
    printf("    --ivote-p-factor=F       : Override the default p-factor (0.75) for ivoting\n");
    printf("\n");
    printf("output options:\n");
    printf("    --output-accuracies       : Print average individual and voted accuracies\n");
    printf("                                on stdout\n");
    printf("                                On by default\n");
    printf("    --no-output-accuracies    : Turn off the printing of accuracies\n");
    printf("    --output-predictions      : For exodus data: create an exodus file with\n");
    printf("                                truth and predictions.\n");
    printf("                                For avatar data: create a text file with a single\n");
    printf("                                column containing the predicted class\n");
    printf("    --output-probabilities    : For exodus data: add class probabilities to the\n");
    printf("                                predictions file.\n");
    printf("                                For avatar data: create a text file with the class\n");
    printf("                                probabilities.\n");
    printf("    --output-confusion-matrix : Print the confusion matrix on stdout.\n");
    printf("\n");
    printf("miscellaneous options:\n");
    printf("    --seed=N      : Use the integer N as the random number generator seed\n");
    printf("    -v, --verbose : Increase verbosity of fclib functions.\n");
    //printf("    --write|-w    : Write the *.itest and *.idata files\n");
    exit(-1);
}

