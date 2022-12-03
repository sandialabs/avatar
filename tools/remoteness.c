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
#ifndef _GNU_SOURCE
  #include "getopt.h"
#else
  #include <getopt.h>
#endif

#include "../src/crossval.h"
#include "../src/version_info.h"
#include "../src/rw_data.h"
#include "../src/tree.h"
#include "../src/evaluate.h"
#include "../src/array.h"
#include "../src/safe_memory.h"
#include "../src/util.h"
#include "proximity_utils.h"

struct Global_Args_t {
    char* namesfile;  // file specifying data columns and their format
    char* modelfile;  // file containing the tree ensemble
    char* datafile;   // file with data points to test for remoteness
    char* ref_datafile;// reference data that is presumed to be clean
    Boolean print_prox_progress;
    int truth_column;
    int num_skipped_features;
    int* skipped_features;
} MyArgs;

// Prototypes for helper functions.
void _display_usage(Boolean show_details);
void _process_opts(int argc, char** argv);
void _free_options(struct Global_Args_t* options);

void
display_usage( void ) 
{
    _display_usage(FALSE);
}

int 
main(int argc, char **argv) 
{
    // Parse arguments and options.
    _process_opts(argc, argv);

    // Read names file.
    // This could be simplified if we didn't need Args_Opts for read_names_file().
    Args_Opts ens_opts;  memset(&ens_opts, 0, sizeof(Args_Opts));
    CV_Metadata meta;    memset(&meta, 0, sizeof(CV_Metadata));
    CV_Class classmeta;  memset(&classmeta, 0, sizeof(CV_Class));
    ens_opts.truth_column = MyArgs.truth_column;
    ens_opts.num_skipped_features = MyArgs.num_skipped_features;
    if (ens_opts.num_skipped_features > 0) {
        int i;
        ens_opts.skipped_features = e_calloc(ens_opts.num_skipped_features, sizeof(int));
        for (i = 0; i < ens_opts.num_skipped_features; ++i) {
            ens_opts.skipped_features[i] = MyArgs.skipped_features[i];
        }
    }
    ens_opts.trees_file = e_strdup(MyArgs.modelfile);
    ens_opts.names_file = e_strdup(MyArgs.namesfile);
    ens_opts.format = AVATAR_FORMAT;
    if (!read_names_file(&meta, &classmeta, &ens_opts, TRUE)) {
        fprintf(stderr, "Error reading names file %s\n", ens_opts.names_file);
        exit(-1);
    }

    // REVIEW-2012-04-04-ArtM: short cut to be implemented later
    if (MyArgs.ref_datafile == NULL) {
        fprintf(stderr, "sorry, currently --ref-data is required\n");
        exit(1);
    }

    // Read the reference dataset to contrast with.  This data will be
    // used to set landmarks for measuring remoteness.
    CV_Dataset dataset;  memset(&dataset, 0, sizeof(CV_Dataset));
    dataset.examples = e_calloc(1, sizeof(CV_Example));
    dataset.meta = meta;
    CV_Subset subset;  memset(&subset, 0, sizeof(CV_Subset));
    AV_SortedBlobArray sorted_examples;  memset(&sorted_examples, 0, sizeof(AV_SortedBlobArray));
    av_exitIfError(av_initSortedBlobArray(&sorted_examples));
    FC_Dataset ds;  memset(&ds, 0, sizeof(FC_Dataset));
    ens_opts.caller = UNKNOWN_CALLER;
    ens_opts.do_testing = TRUE;
    ens_opts.test_file = MyArgs.ref_datafile;
    if (!read_data_file(&dataset, &subset, &classmeta, &sorted_examples, "test", &ens_opts)) {
        fprintf(stderr, "Error reading data file: %s\n", MyArgs.ref_datafile);
        exit(-1);
    }
    subset.meta.exo_data.num_seq_meshes = 0;
    ens_opts.test_file = NULL;

    // Read ensemble model.
    DT_Ensemble ensemble;  memset(&ensemble, 0, sizeof(DT_Ensemble));
    read_ensemble(&ensemble, -1, 0, &ens_opts);

    // Establish landmarks.
    Landmarks* landmarks = create_landmarks(&ensemble, &subset, MyArgs.print_prox_progress);
    
    // No longer need reference data.
    //
    // REVIEW-2012-04-06-ArtM: Memory leak here.  Because of aliasing
    // b/w meta, dataset,
    // and subset, freeing the datasets while keeping meta is not
    // currently possible.
/*     meta = dataset.meta; */
/*     memset(&(dataset.meta), 0, sizeof(CV_Metadata)); */
/*     free_CV_Subset(subset, ens_opts); */
/*     free_CV_Dataset(dataset, ens_opts); */

    // Read data points for which we want remoteness scores.
    memset(&dataset, 0, sizeof(CV_Dataset));
    dataset.examples = e_calloc(1, sizeof(CV_Example));
    dataset.meta = meta;
    memset(&subset, 0, sizeof(CV_Subset));
    memset(&sorted_examples, 0, sizeof(AV_SortedBlobArray));
    av_exitIfError(av_initSortedBlobArray(&sorted_examples));
    memset(&ds, 0, sizeof(FC_Dataset));
    ens_opts.test_file = MyArgs.datafile;
    if (!read_data_file(&dataset, &subset, &classmeta, &sorted_examples, "test", &ens_opts)) {
        fprintf(stderr, "Error reading data file: %s\n", MyArgs.datafile);
        exit(-1);
    }
    subset.meta.exo_data.num_seq_meshes = 0;
    ens_opts.test_file = NULL;

    // Score the remoteness of the data points.
    float* scores = e_calloc(subset.meta.num_examples, sizeof(float));
    measure_remoteness(&ensemble, landmarks, &subset, scores);

    // Print the scores to stdout.
    uint i = 0;
    for (i = 0; i < (uint)subset.meta.num_examples; ++i) {
        fprintf(stdout, "%g\n", scores[i]);
    }

    // Clean up.
    free(scores);  scores = NULL;
    free_landmarks(landmarks);  landmarks = NULL;

    return 0;
}

void 
_display_usage(Boolean show_details) {
    printf("usage: remoteness [options] namesfile modelfile datafile\n");
    printf("%s\n", get_version_string());
    printf("--------------------------------------------------------------------------------\n");
    printf("Compute how remote examples in datafile are, as measured by the forest\n" 
           "proximity (using tree ensemble in modelfile).\n"
           "\n"
           "Remoteness scores are printed to stdout, in the same line order as datafile.\n"
           "\n"
           "OPTIONS\n"
           "\n"
           "  --help              : Show this help message.\n"
           "  --exclude=R         : Exclude features listed in R (e.g., 1-4,6).  May be\n"
           "                      : specified multiple times.\n"
           "  --manual            : Show tool manual.\n"
           "  --print-proximity-progress  : Print progress while computing prox. matrix.\n"
           "  --ref-data=file     : base remoteness scores on proximity to reference data\n"
           "  --truth-column=N    : Use column N as truth column.  Defaults to column marked\n"
           "                        as 'class' in namesfile, or the last column.\n"
        );
    if (!show_details) {
        return;
    }
    printf("\n\n"
           "DETAILS\n"
           "\n"
           "There are multiple ways to derive outlier scores based on Leo Breiman's forest\n"
           "proximity.  The proximity of points i and j to each other in a tree ensemble\n"
           "is the fraction of trees in which they land in the same leaf:\n"
           "\n"
           "                 sum_t I[t.leaf(i) == t.leaf(j)] \n"
           "    prox(i,j) = --------------------------------\n"
           "                         # trees\n"
           "\n"
           "where I[ ] is the indicator function that returns 1 if true and 0 otherwise.\n"
           "Breiman derived an outlier score (for checking for mislabeled data) as\n"
           "follows.  The outlierness of point i, with respect to the other points j in the\n"
           "same class as i, is:\n"
           "\n"
           "    rawOUT(i) = 1 / (sum_j prox(i,j)^2)\n"
           "\n"
           "The raw outlier scores are then normalized to be comparable across classes.\n"
           "\n"
           "This tool supports the following variations.\n"
           "\n"
           "Extrapolation-Check      (option --ref-data)\n"
           "  The points i in the test data are compared to landmark points j from the\n"
           "  reference data.  Multiple outlier scores for i are computed, as if it\n"
           "  belonged to each of the possible classes c.  The remoteness of i is:\n"
           "\n"
           "      remoteness(i) = min_c outlier(i | label(i) = c)\n"
           "\n"
           "  Note that extrapolation-check ignores the labels in the test data, and\n"
           "  calibrates the raw outlier scores using statistics from the distribution\n"
           "  of outlier scores on the reference data.\n"
        );
}

enum {
    no_option,
    option_help,
    option_manual,
    option_exclude,
    option_print_prox_progress,
    option_ref_data,
    option_truth_column,
};

static const struct option long_opts[] = {
    {"help", no_argument, NULL, option_help},
    {"manual", no_argument, NULL, option_manual},
    {"exclude", required_argument, NULL, option_exclude},
    {"print-proximity-progress", no_argument, NULL, option_print_prox_progress},
    {"ref-data", required_argument, NULL, option_ref_data},
    {"truth-column", required_argument, NULL, option_truth_column},
    {NULL, no_argument, NULL, 0}
};

void
_process_opts(int argc, char** argv)
{
    int opt = 0;
    int long_index = 0;
    Boolean found_opt = 0;

    // Initialize to defaults.
    memset(&MyArgs, 0, sizeof(struct Global_Args_t));
    MyArgs.truth_column = -1;

    // Grab options.
    found_opt = -1 != (opt = getopt_long(argc, argv, "", long_opts, &long_index));
    while (found_opt) {
        switch (opt) {
        case option_help:
            _display_usage(FALSE);
            exit(0);
            break;
        case option_manual:
            _display_usage(TRUE);
            exit(0);
            break;
        case option_exclude: {
            int count = 0;
            int* to_exclude = NULL;
            parse_int_range(optarg, 1, &count, &to_exclude);

            if (MyArgs.num_skipped_features > 0) {
                fprintf(stderr, "error: handling multple --exclude is not implemented yet. sorry\n");
                exit(-1);
            }

            MyArgs.num_skipped_features = count;
            MyArgs.skipped_features = to_exclude;
            break;
        }
        case option_print_prox_progress:
            MyArgs.print_prox_progress = TRUE;
            break;
        case option_ref_data:
            MyArgs.ref_datafile = e_strdup(optarg);
            break;
        case option_truth_column:
            MyArgs.truth_column = atoi(optarg);
            break;
        default:
            break;
        }
        found_opt = -1 != (opt = getopt_long(argc, argv, "", long_opts, &long_index));
    }

    // Grab required arguments.
    argc -= optind;
    if (argc != 3) {
        fprintf(stderr, "Missing required arguments.\n");
        display_usage();
        exit(1);
    }
    argv += optind;
    MyArgs.namesfile = e_strdup(argv[0]);
    MyArgs.modelfile = e_strdup(argv[1]);
    MyArgs.datafile = e_strdup(argv[2]);

    //
    // Some basic sanity checks.
    //

    // Truth column should not have been excluded.
    if (MyArgs.truth_column != -1) {
        int i;
        for (i = 0; i < MyArgs.num_skipped_features; ++i) {
            if (MyArgs.skipped_features[i] == MyArgs.truth_column) {
                fprintf(stderr, "error: the truth column cannot be excluded\n");
                exit(1);
            }
        }
    }
}

void _free_options(struct Global_Args_t* options)
{
    if (options == NULL) {
        free(options->namesfile);     options->namesfile = NULL;
        free(options->modelfile);     options->modelfile = NULL;
        free(options->datafile);      options->datafile = NULL;
        free(options->ref_datafile);  options->ref_datafile = NULL;
        options->print_prox_progress = FALSE;
        options->truth_column = 0;
        free(options->skipped_features); options->skipped_features = NULL;
        options->num_skipped_features = 0;
    }
}
