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
#include <libgen.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include "crossval.h"
#include "options.h"
#include "util.h"
#include "array.h"
#include "gain.h"
#include "distinct_values.h"
#include "version_info.h"
#include "skew.h"
#ifndef __GNU_LIBRARY__
  #include "getopt.h"
#else
  #include <getopt.h>
#endif

Args_Opts Args;

//Modified by MEGOLDS August, 2012: subsampling
//Included switch 'b'
static const char *opt_string = "+C:d:f:hN:o:P:V:vXA:B::S::F::E::TIk:p:m:n:s:zb:";

enum {
    no_option,
    option_random_seed,
    option_test_times,
    option_train_times,
    option_truth_column,
    option_exclude,
    option_include,
    option_majority_bagging,
    option_bite_size,
    option_num_iterations,
    option_ivote_p_factor,
    option_majority_ivoting,
    option_smote_Ln,
    option_smote,
    option_boosting,
    option_smoteboost,
    option_minority_classes,
    option_auto_stop,
    option_slide_size,
    option_build_size,
    option_stopping_algorithm_regtest,
    option_train_file,
    option_names_file,
    option_test_file,
    option_trees_file,
    option_predictions_file,
    option_oob_file,
    option_prox_sorted_file,
    option_prox_matrix_file,
    option_sort,
    option_probability_type,
};

//Modified by DACIESL June-04-08: Laplacean Estimates
//Added --output-laplacean
//Modified by MEGOLDS August, 2012: subsampling
//Added --subsample 
static const struct option long_opts[] = {
    {"class-file", required_argument, NULL, 'C'},
    {"partition-file", required_argument, NULL, 'P'},
    {"datafile", required_argument, NULL, 'd'},
    {"filestem", required_argument, NULL, 'f'},
    {"help", no_argument, NULL, 'h'},
    {"folds", required_argument, NULL, 'N'},
    {"format", required_argument, NULL, 'o'},
    {"class-var", required_argument, NULL, 'V'},
    {"verbose", no_argument, NULL, 'v'},
    {"version", no_argument, (int *)&Args.print_version, TRUE},
    {"5x2", no_argument, NULL, 'X'},
    {"loov", no_argument, (int *)&Args.do_loov, TRUE},
    
    {"train", no_argument, (int *)&Args.do_training, TRUE},
    {"test", no_argument, (int *)&Args.do_testing, TRUE},
    
    {"seed", required_argument, NULL, option_random_seed},
    {"test-times", required_argument, NULL, option_test_times},
    {"train-times", required_argument, NULL, option_train_times},
    {"write-folds", no_argument, (int *)&Args.write_folds, TRUE},
    
    // Fold Generation options
    {"no-rigorous-strat", no_argument, (int *)&Args.do_rigorous_strat, FALSE},
    
    // Decision Tree Generation options
    {"num-trees", required_argument, NULL, 'n'},
    {"use-stopping-algorithm", no_argument, NULL, option_auto_stop},
    {"slide-size", required_argument, NULL, option_slide_size},
    {"build-size", required_argument, NULL, option_build_size},
    {"random-attributes", required_argument, NULL, 'A'},
    {"bagging", optional_argument, NULL, 'B'},
    {"random-forests", optional_argument, NULL, 'F'},
    {"totl-random-trees", no_argument, NULL, 'T'},
    {"extr-random-trees", optional_argument, NULL, 'E'},
    {"majority-bagging", no_argument, NULL, option_majority_bagging},
    {"hard-limit-bounds", required_argument, NULL, 'm'},
    {"trees", required_argument, NULL, 'n'},
    {"random-subspaces", optional_argument, NULL, 'S'},
    {"split-method", required_argument, NULL, 's'},
    {"split-zero-gain", no_argument, NULL, 'z'},
    {"no-split-zero-gain", no_argument, (int *)&Args.split_on_zero_gain, FALSE},
    {"subsample", required_argument, NULL, 'b'},
    
    {"collapse-subtree", no_argument, (int *)&Args.collapse_subtree, TRUE},
    {"no-collapse-subtree", no_argument, (int *)&Args.collapse_subtree, FALSE},
    {"dynamic-bounds", no_argument, (int *)&Args.dynamic_bounds, TRUE},
    {"no-dynamic-bounds", no_argument, (int *)&Args.dynamic_bounds, FALSE},
    {"save-trees", no_argument, (int *)&Args.save_trees, TRUE},
    {"no-save-trees", no_argument, (int *)&Args.save_trees, FALSE},
    
    // User Customizations
    {"exclude", required_argument, NULL, option_exclude},
    {"include", required_argument, NULL, option_include},
    {"truth-column", optional_argument, NULL, option_truth_column},

    // ivoting options
    {"ivoting", no_argument, NULL, 'I'},
    {"bite-size", required_argument, NULL, option_bite_size},
    {"iterations", required_argument, NULL, option_num_iterations},
    {"ivote-p-factor", required_argument, NULL, option_ivote_p_factor},
    {"majority-ivoting", no_argument, NULL, option_majority_ivoting},
    
    // smote options
    {"smote", optional_argument, NULL, option_smote},
    {"nearest-neighbors", required_argument, NULL, 'k'},
    {"distance-type", required_argument, NULL, option_smote_Ln},
    {"proportions", required_argument, NULL, 'p'},
    {"minority-classes", required_argument, NULL, option_minority_classes},
    
    // Boosting and SMOTEBoost opeions
    {"boosting", no_argument, NULL, option_boosting},
    {"smoteboost", optional_argument, NULL, option_smoteboost},
    
    // balanced learning options
    {"balanced-learning", no_argument, (int *)&Args.do_balanced_learning, TRUE},
    
    // diversity options
    {"output-kappa-plot-data", no_argument, (int *)&Args.kappa_plot_data, TRUE},
    
    // proximity options
    {"use-standard-deviation", no_argument, (int *)&Args.deviation_type, STANDARD_DEVIATION},
    {"use-absolute-deviation", no_argument, (int *)&Args.deviation_type, ABSOLUTE_DEVIATION},
    {"sort", optional_argument, NULL, option_sort},
    {"print-proximity-progress", no_argument, (int *)&Args.print_prox_prog, TRUE},
    {"save-matrix", no_argument, (int *)&Args.save_prox_matrix, TRUE},
    {"no-save-matrix", no_argument, (int *)&Args.save_prox_matrix, FALSE},
    {"load-matrix", no_argument, (int *)&Args.load_prox_matrix, TRUE},
    
    // Alternate filenames
    {"train-file", required_argument, NULL, option_train_file},
    {"names-file", required_argument, NULL, option_names_file},
    {"test-file", required_argument, NULL, option_test_file},
    {"trees-file", required_argument, NULL, option_trees_file},
    {"predictions-file", required_argument, NULL, option_predictions_file},
    {"oob-file", required_argument, NULL, option_oob_file},
    {"prox-sorted-file", required_argument, NULL, option_prox_sorted_file},
    {"prox-matrix-file", required_argument, NULL, option_prox_matrix_file},
    {"test-file-string", no_argument, NULL, 'i'},
    
    // Output options
    {"output-accuracies", no_argument, (int *)&Args.output_accuracies, ON},
    {"no-output-accuracies", no_argument, (int *)&Args.output_accuracies, OFF},
    {"output-performance-metrics", no_argument, (int *)&Args.output_accuracies, VERBOSE},
    {"output-predictions", no_argument, (int *)&Args.output_predictions, TRUE},
    {"no-output-predictions", no_argument, (int *)&Args.output_predictions, FALSE},
    {"output-probabilities", optional_argument, NULL, option_probability_type},
    {"no-output-probabilities", no_argument, (int *)&Args.output_probabilities, FALSE},
    {"output-confusion-matrix", no_argument, (int *)&Args.output_confusion_matrix, TRUE},
    {"no-output-confusion-matrix", no_argument, (int *)&Args.output_confusion_matrix, FALSE},
    {"verbose-oob", no_argument, (int *)&Args.output_verbose_oob, TRUE},
    {"output-margins", no_argument, (int *)&Args.output_margins, TRUE},
    
    // Ensemble handling options
    {"do-mmv", no_argument, (int *)&Args.do_mass_majority_vote, TRUE},
    {"do-emv", no_argument, (int *)&Args.do_ensemble_majority_vote, TRUE},
    {"do-memv", no_argument, (int *)&Args.do_margin_ensemble_majority_vote, TRUE},
    {"do-pmv", no_argument, (int *)&Args.do_probabilistic_majority_vote, TRUE},
    {"do-spmv", no_argument, (int *)&Args.do_scaled_probabilistic_majority_vote, TRUE},
    
    // Unpublished options
    {"debug", no_argument, (int *)&Args.debug, TRUE},
    {"read-folds", no_argument, (int *)&Args.read_folds, TRUE},
    {"run-regression-test", no_argument, (int *)&Args.run_regression_test, TRUE},
    {"use-opendt-shuffle", no_argument, (int *)&Args.use_opendt_shuffle, TRUE},
    {"no-break-ties-randomly", no_argument, (int *)&Args.break_ties_randomly, FALSE},
    {"stopping-algorithm-regtest", no_argument, (int *)&Args.stopping_algorithm_regtest, TRUE},
    {"show-per-process-stats", no_argument, (int *)&Args.show_per_process_stats, TRUE},
    {"common-mpi-rand48-seed", no_argument, (int *)&Args.common_mpi_rand48_seed, TRUE},

    {NULL, no_argument, NULL, 0}
};

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added HELLINGER to 's' flag
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added default for Args.output_laplacean = FALSE
//Modified by MEGOLDS August, 2012: subsampling
//Added default for Args.subsampling = 0
Args_Opts process_opts(int argc, char **argv) {//, Args_Opts *args) {
    
    int i;
    int opt = 0;
    int long_index = 0;
    Boolean *included_features = NULL;
    int sizeof_included_features = 0;
    int *features_in_out;
    int num_features_in_out, biggest, t_count, num_p;
    char *t;
    char **tokens;
    
    // Initialize
    Args.go4it = TRUE;
    
    Args.caller = UNKNOWN_CALLER;
    Args.format = EXODUS_FORMAT;
    Args.datafile = NULL;
    Args.data_path = NULL;
    Args.class_var_name = NULL;
    Args.classes_filename = NULL;
    Args.partitions_filename = NULL;
    Args.num_folds = 10;
    Args.do_5x2_cv = FALSE;
    Args.do_nfold_cv = FALSE;
    Args.do_rigorous_strat = TRUE;
    Args.do_training = FALSE;
    Args.num_train_times = 0;
    Args.do_testing = FALSE;
    Args.num_test_times = 0;
    Args.write_folds = FALSE;
    Args.random_seed = time(NULL) + getpid();
    srand48(Args.random_seed);
    Args.base_filestem = NULL;
    Args.verbosity = 0;
    Args.print_version = FALSE;
    
    // User customization options
    Args.truth_column = -1;
    Args.exclude_all_features_above = -1;
    Args.num_skipped_features = 0;
    Args.skipped_features = NULL;
    Args.num_explicitly_skipped_features = 0;
    Args.explicitly_skipped_features = NULL;

    // Decision Tree Generation options
    Args.num_trees = 0;
    Args.auto_stop = FALSE;
    Args.build_size = 20;
    Args.slide_size = 5;
    Args.split_on_zero_gain = FALSE;
    Args.subsample = 0;
    Args.dynamic_bounds = TRUE;
    Args.minimum_examples = 2;
    Args.split_method = C45STYLE;
    Args.save_trees = TRUE;
    Args.random_forests = 0;
    Args.extr_random_trees = 0;
    Args.totl_random_trees = 0;
    Args.random_attributes = 0;
    Args.random_subspaces = 0.0;
    Args.collapse_subtree = TRUE;
    
    // bagging options
    Args.do_bagging = FALSE;
    Args.bag_size = 0.0;
    Args.majority_bagging = FALSE;
    
    // ivote options
    Args.do_ivote = FALSE;
    Args.bite_size = 0;
    Args.ivote_p_factor = 0.75;
    Args.majority_ivoting = FALSE;
    
    // SMOTE options
    Args.do_smote = FALSE;
    Args.smote_knn = 5;
    Args.smote_Ln = 2;
    Args.smote_type = OPEN_SMOTE;
    
    // Boosting and SMOTEBoost options
    Args.do_boosting = FALSE;
    Args.do_smoteboost = FALSE;
    Args.smoteboost_type = ALL_MINORITY_CLASSES;
    
    // balanced learning options
    Args.do_balanced_learning = FALSE;
    
    // skew data handling options (SMOTE, Balanced Learning, Majority Bagging)
    Args.num_minority_classes = 0;
    Args.minority_classes = NULL;
    Args.minority_classes_char = NULL;
    Args.proportions = NULL;
    
    // rfFeatureValue options
    Args.do_noising = FALSE;
    
    // diversity options
    Args.kappa_plot_data = FALSE;
    
    // proximity options
    Args.deviation_type = STANDARD_DEVIATION;
    Args.sort_line_num = -1;
    Args.save_prox_matrix = TRUE;
    Args.load_prox_matrix = FALSE;
    
    // Ensemble handling options
    Args.do_mass_majority_vote = FALSE;
    Args.do_ensemble_majority_vote = FALSE;
    Args.do_margin_ensemble_majority_vote = FALSE;
    Args.do_probabilistic_majority_vote = FALSE;
    Args.do_scaled_probabilistic_majority_vote = FALSE;
    
    // Alternate filenames]
    Args.train_file = NULL;
    Args.train_file_is_a_string = FALSE;
    Args.train_string = NULL;
    Args.names_file = NULL;
    Args.names_file_is_a_string = FALSE;
    Args.names_string = NULL;
    Args.test_file = NULL;
    Args.test_file_is_a_string = FALSE;
    Args.test_string = NULL;
    Args.trees_file = NULL;
    Args.trees_file_is_a_string = FALSE;
    Args.trees_string = NULL;
    Args.predictions_file = NULL;
    Args.oob_file = NULL;
    Args.prox_sorted_file = NULL;
    
    // Output options
    Args.output_accuracies = LIMBO;
    Args.output_predictions = FALSE;
    Args.output_probabilities = FALSE;
    Args.output_probabilities_warning = FALSE;
    Args.output_laplacean = FALSE;
    Args.output_confusion_matrix = FALSE;
    Args.output_verbose_oob = FALSE;
    Args.output_margins = FALSE;

    // Unpublished options
    Args.debug = FALSE;
    Args.read_folds = FALSE;
    Args.use_opendt_shuffle = FALSE;
    Args.run_regression_test = FALSE;
    Args.break_ties_randomly = TRUE;
    Args.stopping_algorithm_regtest = FALSE;
    Args.show_per_process_stats = FALSE;
    Args.common_mpi_rand48_seed = FALSE;
    
    // Derived element for MPI
    Args.mpi_rank = 0; // Defaults to root node
    
    //Modified by MEGOLDS August, 2012: subsampling
    //Added case 'b' 
    int num_argopt_errors = 0;
    while( (opt = getopt_long( argc, argv, opt_string, long_opts, &long_index )) != -1 ) {
        switch(opt) {
            case 'b':
                Args.subsample = atoi(optarg);
                break;
            case 'C':
                Args.classes_filename = av_strdup(optarg);
                break;
            case 'd':
                Args.datafile = av_strdup(optarg);
                File_Bits bits = explode_filename(optarg);
                Args.data_path = av_strdup(bits.dirname);
                Args.base_filestem = av_strdup(bits.basename);
                break;
            case 'f':
                t = av_strdup(optarg);
                Args.base_filestem = av_strdup(basename(t));
                free(t);
                t = av_strdup(optarg);
                Args.data_path = av_strdup(dirname(t));
                Args.datafile = (char *)malloc((strlen(Args.data_path) + strlen(Args.base_filestem) + 7) * sizeof(char));
                sprintf(Args.datafile, "%s/%s.data", Args.data_path, Args.base_filestem);
                free(t);
                break;
            case 'h':
                display_usage();
                break;
            case 'N':
                Args.num_folds = atoi(optarg);
                Args.do_nfold_cv = 1;
                break;
            case 'o':
                if (! strcasecmp(optarg, "avatar"))
                    Args.format = AVATAR_FORMAT;
                else if (! strcasecmp(optarg, "exodus"))
                    Args.format = EXODUS_FORMAT;
                else
                    Args.format = UNKNOWN_FORMAT;
                break;
            //case 't':
            //    Args.timestep = atoi(optarg);
            //    break;
            case 'P':
                Args.partitions_filename = av_strdup(optarg);
                break;
            case 'V':
                Args.class_var_name = av_strdup(optarg);
                break;
            case 'v':
                Args.verbosity++;
                break;
            case 'X':
                Args.num_folds = 2;
                Args.do_5x2_cv = 1;
                break;
                
            case option_random_seed:
                Args.random_seed = atoi(optarg);
                srand48(Args.random_seed);
                break;
            case option_test_times:
                Args.do_testing = TRUE;
                parse_int_range(optarg, 1, &Args.num_test_times, &Args.test_times);
                break;
            case option_train_times:
                Args.do_training = TRUE;
                parse_int_range(optarg, 1, &Args.num_train_times, &Args.train_times);
                break;
                
            case option_truth_column:
                Args.truth_column = atoi(optarg);
                break;
            case option_exclude:
                parse_int_range(optarg, 1, &num_features_in_out, &features_in_out);
                // So check to make sure our included_features array is big enough and set new elements to TRUE
                biggest = features_in_out[num_features_in_out-1];
                if (biggest > sizeof_included_features) {
                    int diff = biggest - sizeof_included_features;
                    sizeof_included_features = biggest;
                    included_features =
                                    (Boolean *)realloc(included_features, sizeof_included_features * sizeof(Boolean));
                    for (i = biggest - diff; i < biggest; i++)
                        included_features[i] = TRUE;
                }
                // Turn off all features explicitly excluded from the command line
                for (i = 0; i < num_features_in_out; i++)
                    included_features[features_in_out[i]-1] = FALSE; // Convert user's 1-based feature id to 0-based
                
                // Update excluded feature list
                free(Args.skipped_features);
                Args.skipped_features = NULL;
                free(Args.explicitly_skipped_features);
                Args.explicitly_skipped_features = NULL;
                Args.num_skipped_features = 0;
                for (i = 0; i < sizeof_included_features; i++)
                    if (included_features[i] == FALSE)
                        Args.num_skipped_features++;
                Args.num_explicitly_skipped_features = Args.num_skipped_features;
                Args.skipped_features = (int *)malloc(Args.num_skipped_features * sizeof(int *));
                Args.explicitly_skipped_features = (int *)malloc(Args.num_skipped_features * sizeof(int *));
                t_count = 0;
                for (i = 0; i < sizeof_included_features; i++) {
                    if (included_features[i] == FALSE) {
                        Args.explicitly_skipped_features[t_count] = i+1;
                        Args.skipped_features[t_count++] = i+1; // Convert back to 1-based feature id
                    }
                }
                //printf("Interim explicitly excluded feature list: ");
                //if (Args.num_explicitly_skipped_features > 0) {
                //    char *range;
                //    array_to_range(Args.explicitly_skipped_features, Args.num_explicitly_skipped_features, &range);
                //    printf("%d Skipped Feature Numbers: %s\n", Args.num_explicitly_skipped_features, range);
                //    free(range);
                //}
                break;
            case option_include:
                parse_int_range(optarg, 1, &num_features_in_out, &features_in_out);
                // So check to make sure our included_features array is big enough and set new elements to TRUE
                biggest = features_in_out[num_features_in_out-1];
                
                if (biggest > Args.exclude_all_features_above)
                    Args.exclude_all_features_above = biggest; // This is the biggest feature id to include
                
                if (biggest > sizeof_included_features) {
                    int diff = biggest - sizeof_included_features;
                    sizeof_included_features = biggest;
                    included_features =
                                    (Boolean *)realloc(included_features, sizeof_included_features * sizeof(Boolean));
                    for (i = biggest - diff; i < biggest; i++)
                        included_features[i] = FALSE;
                }
                for (i = 0; i < num_features_in_out; i++)
                    included_features[features_in_out[i]-1] = TRUE; // Convert user's 1-based feature id to 0-based
                
                // Update excluded feature list
                if (Args.num_skipped_features > 0)
                    free(Args.skipped_features);
                Args.num_skipped_features = 0;
                for (i = 0; i < sizeof_included_features; i++)
                    if (included_features[i] == FALSE)
                        Args.num_skipped_features++;
                Args.skipped_features = (int *)malloc(Args.num_skipped_features * sizeof(int *));
                t_count = 0;
                for (i = 0; i < sizeof_included_features; i++)
                    if (included_features[i] == FALSE)
                        Args.skipped_features[t_count++] = i+1; // Convert back to 1-based feature if
                //printf("After processing '--include %s': ", optarg);
                //if (Args.num_skipped_features > 0) {
                //    char *range;
                //    array_to_range(Args.skipped_features, Args.num_skipped_features, &range);
                //    printf("%d Skipped Feature Numbers: %s\n", Args.num_skipped_features, range);
                //    free(range);
                //}
                break;

            case 'A':
                Args.random_attributes = atoi(optarg);
                break;
            case 'B':
                if (optarg)
                    Args.bag_size = atof(optarg);
                Args.do_bagging = TRUE;
                break;
            case option_majority_bagging:
                Args.majority_bagging = TRUE;
                Args.do_bagging = TRUE;
                break;
            case 'F':
                if (optarg)
                    Args.random_forests = atoi(optarg);
                else
                    Args.random_forests = -1;
                break;
            case 'E':
                if (optarg)
                    Args.extr_random_trees = atoi(optarg);
                else
                    Args.extr_random_trees = -1;
                break;
            case 'T':
                Args.totl_random_trees = 1;
                break;
            case 'm':
                Args.minimum_examples = atoi(optarg);
                break;
            case 'n':
                Args.num_trees = atoi(optarg);
                if (Args.auto_stop == TRUE)
                    fprintf(stderr, "Overriding the --auto-stop setting and generating %d trees\n", Args.num_trees);
                Args.auto_stop = FALSE;
                break;
            case option_auto_stop:
                Args.auto_stop = TRUE;
                if (Args.num_trees > 1)
                    fprintf(stderr, "Overriding the --num-trees setting and using stopping algorithm\n");
                Args.num_trees = -1;
                break;
            case option_slide_size:
                Args.slide_size = atoi(optarg);
                break;
            case option_build_size:
                Args.build_size = atoi(optarg);
                break;
            case 'S':
                if (optarg)
                    Args.random_subspaces = atof(optarg);
                else
                    Args.random_subspaces = 50.0;
                break;
            case 's':
                if (! strcmp(optarg, "C45"))
                    Args.split_method = C45STYLE;
                else if (! strcmp(optarg, "GAINRATIO"))
                    Args.split_method = GAINRATIO;
                else if (! strcmp(optarg, "INFOGAIN"))
                    Args.split_method = INFOGAIN;
                else if (! strcmp(optarg, "HELLINGER"))
                    Args.split_method = HELLINGER;
                else {
                    fprintf(stderr, "Invalid split method. Must be one of C45, GAINRATIO, INFOGAIN, HELLINGER\n");
                    display_usage();
                    break;
                }
                break;
            case 'z':
                Args.split_on_zero_gain = TRUE;
                break;
                
            case 'I':
                Args.do_ivote = TRUE;
                break;
            case option_bite_size:
                Args.do_ivote = TRUE;
                Args.bite_size = atoi(optarg);
                break;
            case option_ivote_p_factor:
                Args.ivote_p_factor = atof(optarg);
                break;
            case option_majority_ivoting:
                Args.majority_ivoting = TRUE;
                Args.do_ivote = TRUE;
                break;
            case 'k':
                Args.smote_knn = atoi(optarg);
                break;
            case option_smote_Ln:
                Args.smote_Ln = atoi(optarg);
                break;
            case option_boosting:
                Args.do_boosting = TRUE;
                break;
            case option_smoteboost:
                if (optarg) {
                    if (! strcasecmp(optarg, "all")) {
                        Args.smoteboost_type = ALL_MINORITY_CLASSES;
                    } else if (! strcasecmp(optarg, "subset")) {
                        Args.smoteboost_type = FROM_BOOSTED_SET_ONLY;
                    } else {
                        fprintf(stderr, "Optional argument to --smoteboost must be all|subset\n");
                        display_usage();
                        break;
                    }
                }
                Args.do_smoteboost = TRUE;
                Args.do_boosting = TRUE;
                Args.smote_type = CLOSED_SMOTE;
                break;
            case option_smote:
                if (optarg) {
                    if (! strcasecmp(optarg, "open")) {
                        Args.smote_type = OPEN_SMOTE;
                    } else if (! strcasecmp(optarg, "closed")) {
                        Args.smote_type = CLOSED_SMOTE;
                    } else {
                        fprintf(stderr, "Optional argument to --smote must be open|closed\n");
                        display_usage();
                        break;
                    }
                }
                Args.do_smote = TRUE;
                break;
            case 'p':
                parse_delimited_string(':', optarg, &num_p, &tokens);
                if (Args.num_minority_classes > 0 && num_p != Args.num_minority_classes) {
                    fprintf(stderr, "The number of proportions should be the same as the number of minority classes\n");
                    display_usage();
                    break;
                }
                Args.proportions = (float *)malloc(num_p * sizeof(float));
                for (i = 0; i < num_p; i++)
                    Args.proportions[i] = atof(tokens[i])/100.0;
                Args.num_minority_classes = num_p;
                break;
            case option_minority_classes:
                parse_delimited_string(',', optarg, &num_p, &Args.minority_classes_char);
                if (Args.num_minority_classes > 0 && num_p != Args.num_minority_classes) {
                    fprintf(stderr, "The number of minority classes should be the same as the number of proportions\n");
                    display_usage();
                    break;
                }
                Args.num_minority_classes = num_p;
                break;
            case option_sort:
                if (optarg)
                    Args.sort_line_num = atoi(optarg);
                else
                    Args.sort_line_num = 0;
                break;
            case option_train_file:
                Args.train_file = av_strdup(optarg);
                break;
            case option_names_file:
                Args.names_file = av_strdup(optarg);
                break;
            case option_test_file:
                Args.test_file = av_strdup(optarg);
                break;
            case option_trees_file:
                Args.trees_file = av_strdup(optarg);
                break;
            case option_predictions_file:
                Args.predictions_file = av_strdup(optarg);
                break;
	    case option_probability_type:
                if (optarg) {
                    if (! strcasecmp(optarg, "weighted")) {
                        Args.output_laplacean = TRUE;
                    } else if (! strcasecmp(optarg, "unweighted")) {
	 	        Args.output_probabilities = TRUE;
                    } else {
                        fprintf(stderr, "Optional argument to --output_probabilities must be weighted|unweighted\n");
                        display_usage();
                        break;
                    }
                }
		else {
		  Args.output_probabilities = TRUE;
		}
                break;
            case option_oob_file:
                Args.oob_file = av_strdup(optarg);
                break;
            case option_prox_sorted_file:
                Args.prox_sorted_file = av_strdup(optarg);
                break;
            case option_prox_matrix_file:
                Args.prox_matrix_file = av_strdup(optarg);
                break;

            default:
                if (opt > 0)
                    num_argopt_errors++;
                break;
        }
    }
    
    // If there were any arg/opt errors, quit
    if (num_argopt_errors > 0)
        exit(-1);
    
    return Args;
}

// 

void late_process_opts(int num_atts, int num_examples, Args_Opts *args) {
    
    // Set default for random subspaces
    if (args->random_forests == -1) {
        args->random_forests = (int)ceil(dlog_2_int(num_atts)) + 1;
    }
    
    // Set default for random forests
    if (args->extr_random_trees == -1) {
        args->extr_random_trees = (int)ceil(dlog_2_int(num_atts)) + 1;
    }

    // output-probabilities implies output-predictions
    if (args->output_probabilities)
        args->output_predictions = TRUE;
    
    // Set num-trees if it wasn't specified on command line
    if (args->num_trees == 0) {
        if (args->do_ivote)
            args->num_trees = 100;
        else
            args->num_trees = 1;
    }
    
    // Set up LOOV
    if (args->do_loov == TRUE) {
        args->do_rigorous_strat = FALSE;
        args->do_nfold_cv = TRUE;
        args->do_5x2_cv = FALSE;
        args->num_folds = num_examples;
    }
    
}

void set_output_filenames(Args_Opts *args, Boolean force_input, Boolean force_output) {
    // Set up all output file names
    if (args->format == AVATAR_FORMAT) {
        // If train_file was given, override datafile
        if (args->train_file != NULL)
            args->datafile = av_strdup(args->train_file);
        // If other files were NOT given, assign defaults
        if (force_input || args->names_file == NULL) {
            args->names_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 8) * sizeof(char));
            sprintf(args->names_file, "%s/%s.names", args->data_path, args->base_filestem);
        }
        if (force_input || args->test_file == NULL) {
            args->test_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 7) * sizeof(char));
            sprintf(args->test_file, "%s/%s.test", args->data_path, args->base_filestem);
        }
    }
    if (force_output || args->trees_file == NULL) {
        if (args->save_trees == FALSE) {
            // If --no-save-trees was specified, use a dotted temp name
            char pid[128];
            sprintf(pid, "%d", getpid());
            args->trees_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + strlen(pid) + 10) * sizeof(char));
            sprintf(args->trees_file, "%s/.%s.trees.%s", args->data_path, args->base_filestem, pid);
        } else {
            args->trees_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 8) * sizeof(char));
            sprintf(args->trees_file, "%s/%s.trees", args->data_path, args->base_filestem);
        }
    }
    if (args->format == AVATAR_FORMAT) {
        if (force_output || args->predictions_file == NULL) {
            args->predictions_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 7) * sizeof(char));
            sprintf(args->predictions_file, "%s/%s.pred", args->data_path, args->base_filestem);
        }
    } else if (args->format == EXODUS_FORMAT) {
        if (force_output || args->predictions_file == NULL)
            args->predictions_file = av_strdup("predictions.ex2");
    }
    if (force_output || args->oob_file == NULL) {
        args->oob_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 11) * sizeof(char));
        sprintf(args->oob_file, "%s/%s.oob-data", args->data_path, args->base_filestem);
    }
    if (args->caller == PROXIMITY_CALLER) {
        if (force_output || args->prox_sorted_file == NULL) {
            if (args->format == AVATAR_FORMAT) {
                args->prox_sorted_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 12) * sizeof(char));
                sprintf(args->prox_sorted_file, "%s/%s.proximity", args->data_path, args->base_filestem);
            } else if (args->format == EXODUS_FORMAT) {
                args->prox_sorted_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 16) * sizeof(char));
                sprintf(args->prox_sorted_file, "%s/%s_proximity.ex2", args->data_path, args->base_filestem);
            }
        }
        if (force_output || args->prox_matrix_file == NULL) {
            args->prox_matrix_file = (char *)malloc((strlen(args->data_path) + strlen(args->base_filestem) + 19) * sizeof(char));
            sprintf(args->prox_matrix_file, "%s/%s.proximity_matrix", args->data_path, args->base_filestem);
        }
    }
    /*
    printf("Filenames:\n");
    printf("  datafile    = %s\n", args->datafile);
    printf("  testfile    = %s\n", args->test_file);
    printf("  names       = %s\n", args->names_file);
    printf("  trees       = %s\n", args->trees_file);
    printf("  predictions = %s\n", args->predictions_file);
    printf("  oob_file    = %s\n", args->oob_file);
    printf("  prox_sorted = %s\n", args->prox_sorted);
    */
}

void free_Args_Opts(Args_Opts args) {
    free(args.datafile);
    free(args.names_file);
    free(args.trees_file);
    free(args.skipped_features);
    free(args.class_var_name);
    free(args.data_path);
    free(args.base_filestem);
}

void free_Args_Opts_Full(Args_Opts args)
{
  int i;
  free(args.train_times);
  free(args.test_times);
  free(args.class_var_name);
  free(args.datafile);
  free(args.base_filestem);
  free(args.data_path);
  free(args.classes_filename);
  free(args.partitions_filename);
  free(args.skipped_features);
  free(args.explicitly_skipped_features);
  free(args.minority_classes);
  if(args.minority_classes_char)
  {
    for(i = 0; i < args.num_minority_classes; i++)
      free(args.minority_classes_char[i]);
    free(args.minority_classes_char);
  }
  free(args.proportions);
  free(args.actual_proportions);
  free(args.actual_class_attendance);
  free(args.train_file);
  free(args.train_string);
  free(args.names_file);
  free(args.names_string);
  free(args.test_file);
  free(args.test_string);
  free(args.trees_file);
  free(args.trees_string);
  free(args.predictions_file);
  free(args.oob_file);
  free(args.prox_sorted_file);
  free(args.prox_matrix_file);
  free(args.tree_stats_file);
}

void read_partition_file(CV_Partition *partition, Args_Opts *args) {
    char strbuf[262144];
    
    FILE *partitions_file = NULL;
    partitions_file = fopen(args->partitions_filename, "r");
    if (partitions_file == NULL) {
        fprintf(stderr, "Error: Unable to read partitions file '%s'\n", args->partitions_filename);
        exit(8);
    }
    partition->num_partitions = 0;
    partition->partition_datafile = (char **)malloc(sizeof(char *));
    partition->partition_base_filestem = (char **)malloc(sizeof(char *));
    partition->partition_data_path = (char **)malloc(sizeof(char *));
    
    while (fscanf(partitions_file, "%s", strbuf) > 0) {
        int num = partition->num_partitions;
        partition->num_partitions++;
        partition->partition_datafile = (char **)realloc(partition->partition_datafile,
                                                          partition->num_partitions * sizeof(char *));
        partition->partition_base_filestem = (char **)realloc(partition->partition_base_filestem,
                                                              partition->num_partitions * sizeof(char *));
        partition->partition_data_path = (char **)realloc(partition->partition_data_path,
                                                          partition->num_partitions * sizeof(char *));
        partition->partition_datafile[num] = av_strdup(strbuf);
        File_Bits bits = explode_filename(strbuf);
        partition->partition_data_path[num] = av_strdup(bits.dirname);
        partition->partition_base_filestem[num] = av_strdup(bits.basename);
    }
    
    // Set args file info to first partition so we can read some metadata
    if (args->datafile == NULL)
        args->datafile = av_strdup(partition->partition_datafile[0]);
    if (args->base_filestem == NULL)
        args->base_filestem = av_strdup(partition->partition_base_filestem[0]);
    if (args->data_path == NULL)
        args->data_path = av_strdup(partition->partition_data_path[0]);
    if (args->datafile == NULL)
        args->datafile = av_strdup(partition->partition_datafile[0]);
    
    fclose(partitions_file);
}

void read_classes_file(CV_Class *class, Args_Opts *args) {
    int i;
    char strbuf[262144];
    char class_name_list[262144];
    strcpy(class_name_list, "X");
    int num_thresholds;
    
    FILE *classes_file = NULL;
    classes_file = fopen(args->classes_filename, "r");
    if (classes_file == NULL) {
        fprintf(stderr, "Error: Unable to read classes file '%s'\n", args->classes_filename);
        exit(8);
    }
    class->num_classes = 0;
    
    while (fscanf(classes_file, "%s", strbuf) > 0) {
        if (! strcmp(strbuf, "number_of_classes") && fscanf(classes_file, "%s", strbuf) > 0)
            class->num_classes = atoi(strbuf);
        else if (! strcmp(strbuf, "thresholds") && fscanf(classes_file, "%s", strbuf) > 0)
            parse_float_range(strbuf, 1, &num_thresholds, &class->thresholds);
        else if (! strcmp(strbuf, "class_var_name") && fscanf(classes_file, "%s", strbuf) > 0)
            class->class_var_name = av_strdup(strbuf);
        else if (! strcmp(strbuf, "class_names"))
            fscanf(classes_file, "%s", class_name_list);
    }
    // Derive number of classes if not specified
    if (class->num_classes == 0)
        class->num_classes = num_thresholds + 1;
    
    if (num_thresholds != class->num_classes - 1)
        fprintf(stderr, "%d classes require %d thresholds but %d were given\n",
                        class->num_classes, class->num_classes - 1, num_thresholds);
    class->class_frequencies = (int *)calloc(class->num_classes, sizeof(int));
    class->class_names = (char **)malloc(class->num_classes * sizeof(char *));
    if (! strcmp(class_name_list, "X")) {
        // Set defaults
        int nd = num_digits(class->num_classes);
        char class_name[6 + nd + 1];
        char format[9 + num_digits(nd) + 1];
        sprintf(format, "Class %%0%dd", nd);
        for (i = 0; i < class->num_classes; i++) {
            sprintf(class_name, format, i);
            class->class_names[i] = av_strdup(class_name);
        }
    }
}

//Modified by DACIESL June-04-08: Laplacean Estimates
//Added case where --output-laplacean and --output-probabilities can't both be true
int sanity_check(Args_Opts *args) {
    int num_errors = 0;
    int i;
    
    // If version was requested, print it and we're out
    if (args->print_version == TRUE) {
        if (args->caller == AVATARDT_CALLER)
            printf("%s ", "avatardt");
        else if (args->caller == AVATARMPI_CALLER)
            printf("%s ", "avatarmpi");
        else if (args->caller == CROSSVALFC_CALLER)
            printf("%s ", "crossvalfc");
        else if (args->caller == RFFEATUREVALUE_CALLER)
            printf("%s ", "rfFeatureValue");
        else if (args->caller == DIVERSITY_CALLER)
            printf("%s ", "diversity");
        else if (args->caller == PROXIMITY_CALLER)
            printf("%s ", "proximity");
        printf(get_version_string());
        printf("\n");
        printf("For information contact\n");
        printf("Philip Kegelmeyer (wpk@sandia.gov) \n");
        return 0;
    }
    
    if (args->format == UNKNOWN_FORMAT) {
        fprintf(stderr, "no or invalid format type given\n");
        num_errors++;
    }
    if (! args->do_training && ! args->do_testing) {
        if (args->caller == AVATARDT_CALLER)
            fprintf(stderr, "Must specify --train and/or --test\n");
        if (args->caller == CROSSVALFC_CALLER && args->format == EXODUS_FORMAT)
            fprintf(stderr, "Must specify --train-times\n");
        num_errors++;
    }
    if (args->caller == CROSSVALFC_CALLER) {
        if (! args->do_5x2_cv && ! args->do_nfold_cv && ! args->do_loov) {
            fprintf(stderr, "Must specify one of --folds or --5x2 or --loov\n");
            num_errors++;
        }
        if (args->do_loov && args->do_5x2_cv) {
            fprintf(stderr, "--loov will run N-fold cv. Do not specify both --loov and --5x2\n");
            num_errors++;
        }
        if (args->do_loov && args->do_nfold_cv) {
            fprintf(stderr, "WARNING: --loov implies --folds (or -N)\n");
        }
    }
    if (args->format == EXODUS_FORMAT) {
        // exodus-specific checks
        if (args->class_var_name == NULL && args->classes_filename == NULL) {
            fprintf(stderr, "no class variable name or classes file given\n");
            num_errors++;
        }
        if ( (args->do_training == TRUE && args->num_train_times == 0) ||
             (args->do_testing  == TRUE && args->num_test_times  == 0) ) {
                 fprintf(stderr, "must specify --train-times and/or --test-times for exodus data\n");
                 num_errors++;
        }
        if ( (args->caller == AVATARDT_CALLER || args->caller == CROSSVALFC_CALLER) &&
              args->datafile == NULL) {
            fprintf(stderr, "no exodus datafile given\n");
            num_errors++;
        }
        if ( args->caller == AVATARMPI_CALLER &&
              args->datafile == NULL && args->partitions_filename == NULL) {
            fprintf(stderr, "no datafile or partitions file given\n");
            num_errors++;
        }
    } else if (args->format == AVATAR_FORMAT) {
        // avatar-specific checks
        if ((args->caller == AVATARDT_CALLER || args->caller == CROSSVALFC_CALLER) &&
             args->datafile == NULL && args->base_filestem == NULL && args->partitions_filename == NULL) {
                 fprintf(stderr, "no datafile or filestem given\n");
                 num_errors++;
        }
        if ( args->caller == AVATARMPI_CALLER &&
              args->base_filestem == NULL && args->partitions_filename == NULL) {
            fprintf(stderr, "no filestem or partitions file given\n");
            num_errors++;
        }
        // Check if truth column was explicitly excluded.
        // If not, check if it was implicitly excluded and remove it from the list.
        //printf("Sanity checking skipped features with truth-column = %d\n", args->truth_column);
        //printf("Explicitly excluded feature list: ");
        //if (args->num_explicitly_skipped_features > 0) {
        //    char *range;
        //    array_to_range(args->explicitly_skipped_features, args->num_explicitly_skipped_features, &range);
        //    printf("%d Skipped Feature Numbers: %s\n", args->num_explicitly_skipped_features, range);
        //    free(range);
        //}
        if (find_int(args->truth_column, args->num_explicitly_skipped_features, args->explicitly_skipped_features)) {
            fprintf(stderr, "the truth column cannot be excluded\n");
            num_errors++;
        } else if (find_int(args->truth_column, args->num_skipped_features, args->skipped_features)) {
            for (i = 0; i < args->num_skipped_features-1; i++)
                if (i+1 >= args->truth_column)
                    args->skipped_features[i] = args->skipped_features[i+1];
            args->num_skipped_features--;
        }
        find_int_release();
        //printf("After sanity check skipped feature list: ");
        //if (args->num_skipped_features > 0) {
        //    char *range;
        //    array_to_range(args->skipped_features, args->num_skipped_features, &range);
        //    printf("%d Skipped Feature Numbers: %s\n", args->num_skipped_features, range);
        //    free(range);
        //}
    }

    if (args->auto_stop && ! (args->do_ivote == TRUE || args->do_bagging == TRUE)) {
        fprintf(stderr, "--use-stopping-algorithm must be used with either bagging or ivoting\n");
        num_errors++;
    }
    
    if (args->output_margins == TRUE)
        args->output_predictions = TRUE;
    
    if (args->do_testing == FALSE && (args->output_accuracies == ON || args->output_confusion_matrix == TRUE)) {
        // Can only output accuracies/conf_matrix if testing, too
        fprintf(stderr, "--test or --test-times must be specified with --output-accuracies or --output-confusion-matrix\n");
        num_errors++;
    }
    if (args->output_confusion_matrix == TRUE) {
        if (args->output_accuracies == LIMBO) {
            // Turn on accuracies if the confusion matrix was requested
            args->output_accuracies = ON;
        } else if (args->output_accuracies == OFF) {
            // This is an invalid combination
            fprintf(stderr, "cannot have --no-output-accuracies and --output-confusion-matrix\n");
            num_errors++;
        }
    }
    if (args->output_accuracies == LIMBO) {
        // If unspecified and not turned ON by other options, default is OFF
        args->output_accuracies = OFF;
    }
    
    // Skew data handling
    if (args->do_smote == TRUE || args->do_balanced_learning == TRUE ||
        args->majority_bagging == TRUE || args->majority_ivoting == TRUE) {
        if (args->num_minority_classes == 0) {
            fprintf(stderr, "--minority-classes and --proportions must be specified with ");
            if (args->do_smote == TRUE)
                fprintf(stderr, "--smote\n");
            if (args->do_balanced_learning == TRUE)
                fprintf(stderr, "--balanced-learning\n");
            if (args->majority_bagging == TRUE)
                fprintf(stderr, "--majority-bagging\n");
            if (args->majority_ivoting == TRUE)
                fprintf(stderr, "--majority-ivoting\n");
            num_errors++;
        }
        if (args->proportions == NULL || args->minority_classes_char == NULL) {
            fprintf(stderr, "skew data handling requires --minority-classes and --proportions\n");
            num_errors++;
        }
    }
    
    // Set defaults for bag/bite size and warn if majority_* was specified with a bag/bite size
    if (args->do_bagging == TRUE) {
        if (args->majority_bagging == TRUE && av_gtf(args->bag_size, 0.0)) {
            fprintf(stderr, "Overriding the bag size set on command line since Majority Bagging sets its own.\n");
        } else if (args->majority_bagging == FALSE && av_eqf(args->bag_size, 0.0)) {
            args->bag_size = 100.0;
        }
    }
    if (args->do_ivote == TRUE) {
        if (args->majority_ivoting == TRUE && args->bite_size > 0) {
            fprintf(stderr, "Overriding the bite size set on command line since Majority Ivoting sets its own.\n");
        } else if (args->majority_ivoting == FALSE && args->bite_size == 0) {
            args->bite_size = 50;
        }
    }
    if (args->caller == RFFEATUREVALUE_CALLER) {
        if (! args->do_ivote && args->bag_size <= 0.0) {
            fprintf(stderr, "must use either ivoting or bagging\n");
            num_errors++;
        }
    }

    // If we are doing N trees (and not the stopping algorithm) but still want --verbose-oob output,
    // the number of trees requested needs to be a multiple of the build_size.
    // Update and warn the user
    if (args->output_verbose_oob == TRUE && args->num_trees > 0 && args->num_trees % args->build_size != 0) {
        fprintf(stderr, "Number of trees must be a multiple of the stopping algorithm build size.\n");
        args->num_trees = ((args->num_trees / args->build_size) + 1) * args->build_size;
        fprintf(stderr, "Increasing the number of trees to %d\n", args->num_trees);
    }
    
    if (num_errors > 0) {
        fprintf(stderr, "\nRun `%s --help` for more information\n\n",
                (args->caller==AVATARDT_CALLER?"avatardt":
                (args->caller==CROSSVALFC_CALLER?"crossvalfc":
                (args->caller==AVATARMPI_CALLER?"avatarmpi":""))));
        return 0;
    }
    return 1;
}

void print_skewed_per_class_stats(FILE *fh, char *comment, Args_Opts args, CV_Metadata meta) {
    int i;
    int max_digits = 0;
    unsigned int max_label = 0;
    for (i = 0; i < meta.num_classes; i++) {
        if (num_digits(args.actual_class_attendance[i]) > max_digits)
            max_digits = num_digits(args.actual_class_attendance[i]);
        if (strlen(meta.class_names[i]) > max_label)
            max_label = strlen(meta.class_names[i]);
    }
    char *num_format, *label_format;
    num_format = (char *)malloc(100*sizeof(char));
    sprintf(num_format, "%%%dd", max_digits);
    label_format = (char *)malloc(100*sizeof(char));
    sprintf(label_format, "%%%ds", max_label);
    char *entry_format;
    entry_format = (char *)malloc(300*sizeof(char));
    
    sprintf(entry_format, "%s,%%4.1f/%%4.1f,%s", label_format, num_format);
    fprintf(fh, "%sClass Label,Desired/Actual Proportion,Population\n", comment);
    fprintf(fh, "%sMinority Classes       : ", comment);
    int count = 0;
    for (i = 0; i < args.num_minority_classes; i++) {
        if (count++ > 0)
            fprintf(fh, "%s                         ", comment);
        fprintf(fh, entry_format, meta.class_names[args.minority_classes[i]], args.proportions[i]*100.0,
                                  args.actual_proportions[args.minority_classes[i]]*100.0,
                                  args.actual_class_attendance[args.minority_classes[i]]);
        fprintf(fh, "\n");
    }
    
    sprintf(entry_format, "%s,%%5s%%4.1f,%s", label_format, num_format);
    fprintf(fh, "%sMajority Classes       : ", comment);
    count = 0;
    for (i = 0; i < meta.num_classes; i++) {
      if (! find_int(i, args.num_minority_classes, args.minority_classes)) {
            if (count++ > 0)
                fprintf(fh, "%s                         ", comment);
            fprintf(fh, entry_format, meta.class_names[i], " ",
                                      args.actual_proportions[i]*100.0, args.actual_class_attendance[i]);
            fprintf(fh, "\n");
        }
    }
    
    find_int_release();
    free(num_format);
    free(label_format);
    free(entry_format);
}

//Modified by DACIESL June-03-08: HDDT CAPABILITY
//Added Hellinger to Split Method output
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added Laplacean to Output Probabilities menu
void display_dtmpi_opts(FILE *fh, char *comment, Args_Opts args) {
    fprintf(fh, "%s============================================================\n", comment);
    
    fprintf(fh, "%sData Format            : %s\n", comment, args.format==EXODUS_FORMAT?"exodus":
                                                           (args.format==AVATAR_FORMAT?"avatar":"unknown"));
    if (args.format == EXODUS_FORMAT) {
        fprintf(fh, "%sData Filename          : %s\n", comment, args.datafile);
        fprintf(fh, "%sClasses Filename       : %s\n", comment, args.classes_filename);
        fprintf(fh, "%sClass Variable Name    : %s\n", comment, args.class_var_name);
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sFilestem               : %s\n", comment, args.base_filestem);
        fprintf(fh, "%sTruth Column           : %d\n", comment, args.truth_column);
        if (args.num_skipped_features > 0) {
            char *range;
            array_to_range(args.skipped_features, args.num_skipped_features, &range);
            fprintf(fh, "%sSkipped Feature Numbers: %s\n", comment, range);
            free(range);
        }
    }
    if (! args.do_ivote) {
        if (args.num_trees > 0)
            fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
        else if (args.auto_stop == TRUE)
            fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
    }
    if (args.random_seed != 0)
        fprintf(fh, "%sSeed for *rand48       : %ld\n", comment, args.random_seed);

    if (args.do_training) {
        if (args.format == EXODUS_FORMAT) {
            char *range;
            array_to_range(args.train_times, args.num_train_times, &range);
            fprintf(fh, "%sTraining Times         : %s\n", comment, range);
            free(range);
        } else if (args.format == AVATAR_FORMAT) {
            fprintf(fh, "%sTraining Data          : %s/%s.data\n", comment, args.data_path, args.base_filestem);
        }
    }
    if (args.do_testing) {
        if (args.format == EXODUS_FORMAT) {
            char *range;
            array_to_range(args.test_times, args.num_test_times, &range);
            fprintf(fh, "%sTesting Times          : %s\n", comment, range);
            free(range);
        } else if (args.format == AVATAR_FORMAT) {
            fprintf(fh, "%sTesting Data           : %s/%s.test\n", comment, args.data_path, args.base_filestem);
	    fprintf(fh, "%sTesting Data is string?  : %s\n", comment, args.test_file_is_a_string == TRUE ? "TRUE" : "FALSE");
	    fprintf(fh, "%sTrees File is string?  : %s\n", comment, args.trees_file_is_a_string == TRUE ? "TRUE" : "FALSE");
        }
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sNames File             : %s\n", comment, args.names_file);
        fprintf(fh, "%sNames File is string?  : %s\n", comment, args.names_file_is_a_string == TRUE ? "TRUE" : "FALSE");
    }
    if (args.save_trees)
        fprintf(fh, "%sEnsemble File          : %s\n", comment, args.trees_file);
    if (args.output_predictions)
        fprintf(fh, "%sPredictions File       : %s\n", comment, args.predictions_file);
    if (args.output_verbose_oob)
        fprintf(fh, "%sVerbose OOB File       : %s\n", comment, args.oob_file);
    //fprintf(fh, "%sWrite Fold Files       : %s\n", comment, args.write_folds?"TRUE":"FALSE");
    fprintf(fh, "%sVerbosity              : %d\n", comment, args.verbosity);
    if (args.do_training) {
        fprintf(fh, "%sSplit On Zero Gain     : %s\n", comment, args.split_on_zero_gain?"TRUE":"FALSE");
        fprintf(fh, "%sSubsample For Splits   : %d\n", comment, args.subsample);
        fprintf(fh, "%sCollapse Subtrees      : %s\n", comment, args.collapse_subtree?"TRUE":"FALSE");
        fprintf(fh, "%sDynamic Bounds         : %s\n", comment, args.dynamic_bounds?"TRUE":"FALSE");
        fprintf(fh, "%sMinimum Examples       : %d\n", comment, args.minimum_examples);
        fprintf(fh, "%sSplit Method           : %s\n", comment, args.split_method==C45STYLE?"C45 Style":
                                                               (args.split_method==INFOGAIN?"Information Gain":
                                                               (args.split_method==GAINRATIO?"Gain Ratio":
                                                               (args.split_method==HELLINGER?"Hellinger Distance":"N/A"))));
        fprintf(fh, "%sSave Trees             : %s\n", comment, args.save_trees?"TRUE":"FALSE");
        if (args.random_forests > 0)
            fprintf(fh, "%sRandom Forests         : %d\n", comment, args.random_forests);
        if (args.extr_random_trees > 0)
            fprintf(fh, "%sExtremely Random Trees : %d\n", comment, args.extr_random_trees);
        if (args.totl_random_trees > 0)
            fprintf(fh, "%sTotally Random Trees   : %d\n", comment, args.totl_random_trees);
        //fprintf(fh, "%sRandom Attributes      : %d\n", comment, args.random_attributes);
        if (args.random_subspaces > 0.0)
            fprintf(fh, "%sRandom Subspaces       : %f\n", comment, args.random_subspaces);
        if (args.bag_size > 0.0)
            fprintf(fh, "%sBagging                : %.2f%%\n", comment, args.bag_size);
        if (args.do_ivote) {
            fprintf(fh, "%sIVoting                : %s\n", comment, args.do_ivote?"TRUE":"FALSE");
            fprintf(fh, "%sBite Size              : %d\n", comment, args.bite_size);
            if (args.num_trees > 0)
                fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
            else if (args.auto_stop == TRUE)
                fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
        }
    }
    if (args.do_testing) {
        fprintf(fh, "%sOutput Accuracies      : %s\n", comment, args.output_accuracies==ON?"TRUE":"FALSE");
        fprintf(fh, "%sOutput Predictions     : %s\n", comment, args.output_predictions?"TRUE":"FALSE");
        fprintf(fh, "%sOutput Probabilities   : %s\n", comment, args.output_laplacean?"Laplacean":(args.output_probabilities?"TRUE":"FALSE"));
        fprintf(fh, "%sOutput Confusion Matrix: %s\n", comment, args.output_confusion_matrix?"TRUE":"FALSE");
    }
    fprintf(fh, "%s============================================================\n", comment);
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added Hellinger to Split Method output
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added Laplacean to Output Probabilities menu
void display_dt_opts(FILE *fh, char *comment, Args_Opts args, CV_Metadata meta) {
    fprintf(fh, "%s============================================================\n", comment);
    
    fprintf(fh, "%sData Format            : %s\n", comment, args.format==EXODUS_FORMAT?"exodus":
                                            (args.format==AVATAR_FORMAT?"avatar":"unknown"));
    if (args.format == EXODUS_FORMAT) {
        fprintf(fh, "%sData Filename          : %s\n", comment, args.datafile);
        fprintf(fh, "%sClasses Filename       : %s\n", comment, args.classes_filename);
        fprintf(fh, "%sClass Variable Name    : %s\n", comment, args.class_var_name);
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sFilestem               : %s\n", comment, args.base_filestem);
        fprintf(fh, "%sTruth Column           : %d\n", comment, args.truth_column);
        if (args.num_skipped_features > 0) {
            char *range;
            array_to_range(args.skipped_features, args.num_skipped_features, &range);
            fprintf(fh, "%sSkipped Feature Numbers: %s\n", comment, range);
            free(range);
        }
    }
    if (! args.do_ivote && args.do_training) {
        if (args.num_trees > 0)
            fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
        else if (args.auto_stop == TRUE)
            fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
    }
    if (args.random_seed != 0)
        fprintf(fh, "%sSeed for *rand48       : %ld\n", comment, args.random_seed);

    if (args.do_training) {
        if (args.format == EXODUS_FORMAT) {
            char *range;
            array_to_range(args.train_times, args.num_train_times, &range);
            fprintf(fh, "%sTraining Times         : %s\n", comment, range);
            free(range);
        } else if (args.format == AVATAR_FORMAT) {
            fprintf(fh, "%sTraining Data          : %s\n", comment, args.datafile);
        }
    }
    if (args.do_testing) {
        if (args.format == EXODUS_FORMAT) {
            char *range;
            array_to_range(args.test_times, args.num_test_times, &range);
            fprintf(fh, "%sTesting Times          : %s\n", comment, range);
            free(range);
        } else if (args.format == AVATAR_FORMAT) {
            fprintf(fh, "%sTesting Data           : %s\n", comment, args.test_file);
	    fprintf(fh, "%sTesting Data is string?  : %s\n", comment, args.test_file_is_a_string == TRUE ? "TRUE" : "FALSE");
	    fprintf(fh, "%sTrees File is string?  : %s\n", comment, args.trees_file_is_a_string == TRUE ? "TRUE" : "FALSE");
        }
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sNames File             : %s\n", comment, args.names_file);
        fprintf(fh, "%sNames File is string?  : %s\n", comment, args.names_file_is_a_string == TRUE ? "TRUE" : "FALSE");
    }
    if (args.save_trees)
        fprintf(fh, "%sEnsemble File          : %s\n", comment, args.trees_file);
    if (args.output_predictions)
        fprintf(fh, "%sPredictions File       : %s\n", comment, args.predictions_file);
    if (args.output_verbose_oob)
        fprintf(fh, "%sVerbose OOB File       : %s\n", comment, args.oob_file);
    //fprintf(fh, "%sWrite Fold Files       : %s\n", comment, args.write_folds?"TRUE":"FALSE");
    fprintf(fh, "%sVerbosity              : %d\n", comment, args.verbosity);
    if (args.do_training) {
        fprintf(fh, "%sSplit On Zero Gain     : %s\n", comment, args.split_on_zero_gain?"TRUE":"FALSE");
        fprintf(fh, "%sSubsample For Splits   : %d\n", comment, args.subsample);
        fprintf(fh, "%sCollapse Subtrees      : %s\n", comment, args.collapse_subtree?"TRUE":"FALSE");
        fprintf(fh, "%sDynamic Bounds         : %s\n", comment, args.dynamic_bounds?"TRUE":"FALSE");
        fprintf(fh, "%sMinimum Examples       : %d\n", comment, args.minimum_examples);
        fprintf(fh, "%sSplit Method           : %s\n", comment, args.split_method==C45STYLE?"C45 Style":
                                               (args.split_method==INFOGAIN?"Information Gain":
					       (args.split_method==GAINRATIO?"Gain Ratio":
                                               (args.split_method==HELLINGER?"Hellinger Distance":"N/A"))));
        fprintf(fh, "%sSave Trees             : %s\n", comment, args.save_trees?"TRUE":"FALSE");
        if (args.random_forests > 0)
            fprintf(fh, "%sRandom Forests         : %d\n", comment, args.random_forests);
        if (args.extr_random_trees > 0)
            fprintf(fh, "%sExtremely Random Trees : %d\n", comment, args.extr_random_trees);
        if (args.totl_random_trees > 0)
            fprintf(fh, "%sTotally Random Trees   : %d\n", comment, args.totl_random_trees);
        //fprintf(fh, "Random Attributes      : %d\n", args.random_attributes);
        if (args.random_subspaces > 0.0)
            fprintf(fh, "%sRandom Subspaces       : %f\n", comment, args.random_subspaces);
        int num_clumps, num_ex_per_clump;
        if (args.majority_bagging == TRUE || args.majority_ivoting == TRUE ||
            args.do_balanced_learning == TRUE || args.do_smote == TRUE) {
            fprintf(fh, "%sSkew Correction        : ", comment);
            compute_number_of_clumps(meta, &args, &num_clumps, &num_ex_per_clump);
            if (args.majority_bagging == TRUE) {
                fprintf(fh, "Majority Bagging\n");
                fprintf(fh, "%sExamples per Bag       : %d\n", comment, num_ex_per_clump);
            } else if (args.majority_ivoting == TRUE) {
                fprintf(fh, "Majority Ivoting\n");
            } else if (args.do_balanced_learning == TRUE) {
                fprintf(fh, "Balanced Learning\n");
                fprintf(fh, "%sNumber of Clumps       : %d\n", comment, num_clumps);
                fprintf(fh, "%sExamples per Clump     : %d\n", comment, num_ex_per_clump);
            } else if (args.do_smote == TRUE) {
                fprintf(fh, "%s\n", args.smote_type==OPEN_SMOTE?"Open SMOTE":"Closed SMOTE");
                fprintf(fh, "%sNearest Neighbors      : %d\n", comment, args.smote_knn);
                fprintf(fh, "%sDistance Measure       : %d\n", comment, args.smote_Ln);
            }
        }
        if (args.bag_size > 0.0)
            fprintf(fh, "%sBagging                : %.2f%%\n", comment, args.bag_size);
        if (args.do_ivote) {
            if (args.majority_ivoting == FALSE)
                fprintf(fh, "%sIVoting                : %s\n", comment, args.do_ivote?"TRUE":"FALSE");
            fprintf(fh, "%sBite Size              : %d\n", comment, args.bite_size);
            if (args.num_trees > 0)
                fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
            else if (args.auto_stop == TRUE)
                fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
        }
        if (args.majority_bagging == TRUE || args.majority_ivoting == TRUE ||
            args.do_balanced_learning == TRUE || args.do_smote == TRUE) {
            print_skewed_per_class_stats(fh, comment, args, meta);
        }
    }
    if (args.do_testing) {
        fprintf(fh, "%sOutput Accuracies      : %s\n", comment, args.output_accuracies==ON?"TRUE":"FALSE");
        fprintf(fh, "%sOutput Predictions     : %s\n", comment, args.output_predictions?"TRUE":"FALSE");
        fprintf(fh, "%sOutput Probabilities   : %s\n", comment, args.output_laplacean?"Weighted":(args.output_probabilities?"Unweighted":"None"));
        if (args.output_probabilities_warning) {
            fprintf(fh, "%s                         Warning: Old style input tree\n", comment);
            fprintf(fh, "%s                         Unweighted calculation used\n", comment);
	}
        fprintf(fh, "%sOutput Confusion Matrix: %s\n", comment, args.output_confusion_matrix?"TRUE":"FALSE");
    }
    fprintf(fh, "%s============================================================\n", comment);
}

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added Hellinger to Split Method output
//Modified by DACIESL June-04-08: Laplacean Estimates
//Added Laplacean to Output Probabilities menu
void display_cv_opts(FILE *fh, char *comment, Args_Opts args) {
    fprintf(fh, "%s============================================================\n", comment);
    
    fprintf(fh, "%sData Format            : %s\n", comment, args.format==EXODUS_FORMAT?"exodus":
                                            (args.format==AVATAR_FORMAT?"avatar":"unknown"));
    if (args.format == EXODUS_FORMAT) {
        fprintf(fh, "%sData Filename          : %s\n", comment, args.datafile);
        fprintf(fh, "%sClasses Filename       : %s\n", comment, args.classes_filename);
        fprintf(fh, "%sClass Variable Name    : %s\n", comment, args.class_var_name);
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sData Filename          : %s\n", comment, args.datafile);
        fprintf(fh, "%sNames File             : %s\n", comment, args.names_file);
	fprintf(fh, "%sNames File is string?  : %s\n", comment, args.names_file_is_a_string == TRUE ? "TRUE" : "FALSE");
        fprintf(fh, "%sTruth Column           : %d\n", comment, args.truth_column);
        if (args.num_skipped_features > 0) {
            char *range;
            array_to_range(args.skipped_features, args.num_skipped_features, &range);
            fprintf(fh, "%sSkipped Feature Numbers: %s\n", comment, range);
            free(range);
        }
    }
    if (args.save_trees)
        fprintf(fh, "%sEnsemble File          : %s\n", comment, args.trees_file);
    if (args.output_predictions)
        fprintf(fh, "%sPredictions File       : %s\n", comment, args.predictions_file);
    if (args.output_verbose_oob)
        fprintf(fh, "%sVerbose OOB File       : %s\n", comment, args.oob_file);
    if (args.do_nfold_cv)
        fprintf(fh, "%sNumber of Folds        : %d\n", comment, args.num_folds);
    if (! args.do_ivote) {
        if (args.num_trees > 0)
            fprintf(fh, "%sNumber of Trees/Fold   : %d\n", comment, args.num_trees);
        else if (args.auto_stop == TRUE)
            fprintf(fh, "%sNumber of Trees/Fold   : using stopping algorithm\n", comment);
    }
    fprintf(fh, "%sCrossval Type          : %s\n", comment, args.do_5x2_cv?"5x2":
                                                           (args.do_nfold_cv?"N-fold":"N/A"));
    if (args.random_seed != 0)
        fprintf(fh, "%sSeed for *rand48       : %ld\n", comment, args.random_seed);
    if (args.format == EXODUS_FORMAT) {
        char *range;
        array_to_range(args.train_times, args.num_train_times, &range);
        fprintf(fh, "%sTimeteps               : %s\n", comment, range);
        free(range);
    } else {
        fprintf(fh, "%sTraining Data          : %s/%s.data\n", comment, args.data_path, args.base_filestem);
    }
    //fprintf(fh, "%sWrite Fold Files       : %s\n", comment, args.write_folds?"TRUE":"FALSE");
    fprintf(fh, "%sVerbosity              : %d\n", comment, args.verbosity);
    fprintf(fh, "%sSplit On Zero Gain     : %s\n", comment, args.split_on_zero_gain?"TRUE":"FALSE");
    fprintf(fh, "%sSubsample For Splits   : %d\n", comment, args.subsample);
    fprintf(fh, "%sCollapse Subtrees      : %s\n", comment, args.collapse_subtree?"TRUE":"FALSE");
    fprintf(fh, "%sDynamic Bounds         : %s\n", comment, args.dynamic_bounds?"TRUE":"FALSE");
    fprintf(fh, "%sMinimum Examples       : %d\n", comment, args.minimum_examples);
    fprintf(fh, "%sSplit Method           : %s\n", comment, args.split_method==C45STYLE?"C45 Style":
                                                           (args.split_method==INFOGAIN?"Information Gain":
                                                           (args.split_method==GAINRATIO?"Gain Ratio":
                                                           (args.split_method==HELLINGER?"Hellinger Distance":"N/A"))));
    fprintf(fh, "%sSave Trees             : %s\n", comment, args.save_trees?"TRUE":"FALSE");
    if (args.random_forests > 0)
        fprintf(fh, "%sRandom Forests         : %d\n", comment, args.random_forests);
    if (args.extr_random_trees > 0)
        fprintf(fh, "%sExtremely Random Trees : %d\n", comment, args.extr_random_trees);
    if (args.totl_random_trees > 0)
        fprintf(fh, "%sTotally Random Trees   : %d\n", comment, args.totl_random_trees);
    //fprintf(fh, "%sRandom Attributes      : %d\n", comment, args.random_attributes);
    if (args.random_subspaces > 0.0)
        fprintf(fh, "%sRandom Subspaces       : %f\n", comment, args.random_subspaces);
    if (args.majority_bagging == TRUE || args.majority_ivoting == TRUE ||
        args.do_balanced_learning == TRUE || args.do_smote == TRUE) {
        fprintf(fh, "%sSkew Correction        : ", comment);
        if (args.majority_bagging == TRUE) {
            fprintf(fh, "Majority Bagging\n");
        } else if (args.majority_ivoting == TRUE) {
            fprintf(fh, "Majority Ivoting\n");
        } else if (args.do_balanced_learning == TRUE) {
            fprintf(fh, "Balanced Learning\n");
        } else if (args.do_smote == TRUE) {
            fprintf(fh, "%s\n", args.smote_type==OPEN_SMOTE?"Open SMOTE":"Closed SMOTE");
            fprintf(fh, "%sNearest Neighbors      : %d\n", comment, args.smote_knn);
            fprintf(fh, "%sDistance Measure       : %d\n", comment, args.smote_Ln);
        }
    } else {
        if (args.bag_size > 0.0)
            fprintf(fh, "%sBagging                : %.2f%%\n", comment, args.bag_size);
        if (args.do_ivote) {
            fprintf(fh, "%sIVoting                : %s\n", comment, args.do_ivote?"TRUE":"FALSE");
            fprintf(fh, "%sBite Size              : %d\n", comment, args.bite_size);
            if (args.num_trees > 0)
                fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
            else if (args.auto_stop == TRUE)
                fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
        }
    }
    fprintf(fh, "%sOutput Accuracies      : %s\n", comment, args.output_accuracies==ON?"TRUE":"FALSE");
    fprintf(fh, "%sOutput Predictions     : %s\n", comment, args.output_predictions?"TRUE":"FALSE");
    fprintf(fh, "%sOutput Probabilities   : %s\n", comment, args.output_laplacean?"Weighted":(args.output_probabilities?"Unweighted":"None"));
    if (args.output_probabilities_warning) {
        fprintf(fh, "%s                         Warning: Old style input tree\n", comment);
        fprintf(fh, "%s                         Unweighted calculation used\n", comment);
    }
    fprintf(fh, "%sOutput Confusion Matrix: %s\n", comment, args.output_confusion_matrix?"TRUE":"FALSE");
    
    fprintf(fh, "%s============================================================\n", comment);
}

void display_fv_opts(FILE *fh, char *comment, Args_Opts args) {
    fprintf(fh, "%s============================================================\n", comment);
    
    fprintf(fh, "%sData Format            : %s\n", comment, args.format==EXODUS_FORMAT?"exodus":
                                                           (args.format==AVATAR_FORMAT?"avatar":"unknown"));
    if (args.format == EXODUS_FORMAT) {
        fprintf(fh, "%sData Filename          : %s\n", comment, args.datafile);
        fprintf(fh, "%sClasses Filename       : %s\n", comment, args.classes_filename);
        fprintf(fh, "%sClass Variable Name    : %s\n", comment, args.class_var_name);
    }
    if (args.format == AVATAR_FORMAT) {
        fprintf(fh, "%sFilestem               : %s\n", comment, args.base_filestem);
        fprintf(fh, "%sTruth Column           : %d\n", comment, args.truth_column);
        if (args.num_skipped_features > 0) {
            char *range;
            array_to_range(args.skipped_features, args.num_skipped_features, &range);
            fprintf(fh, "%sSkipped Feature Numbers: %s\n", comment, range);
            free(range);
        }
    }
    if (args.do_nfold_cv)
        fprintf(fh, "%sNumber of Folds        : %d\n", comment, args.num_folds);
    if (! args.do_ivote) {
        if (args.num_trees > 0)
            fprintf(fh, "%sNumber of Trees/Fold   : %d\n", comment, args.num_trees);
        else if (args.auto_stop == TRUE)
            fprintf(fh, "%sNumber of Trees/Fold   : using stopping algorithm\n", comment);
    }
    fprintf(fh, "%sCrossval Type          : %s\n", comment, args.do_5x2_cv?"5x2":
                                                           (args.do_nfold_cv?"N-fold":"N/A"));
    if (args.random_seed != 0)
        fprintf(fh, "%sSeed for *rand48       : %ld\n", comment, args.random_seed);
    if (args.format == EXODUS_FORMAT) {
        char *range;
        array_to_range(args.train_times, args.num_train_times, &range);
        fprintf(fh, "%sTimeteps               : %s\n", comment, range);
        free(range);
    } else {
        fprintf(fh, "%sTraining Data          : %s/%s.data\n", comment, args.data_path, args.base_filestem);
    }
    //fprintf(fh, "%sWrite Fold Files       : %s\n", comment, args.write_folds?"TRUE":"FALSE");
    fprintf(fh, "%sVerbosity              : %d\n", comment, args.verbosity);
    fprintf(fh, "%sSplit On Zero Gain     : %s\n", comment, args.split_on_zero_gain?"TRUE":"FALSE");
    fprintf(fh, "%sSubsample For Splits   : %d\n", comment, args.subsample);
    fprintf(fh, "%sCollapse Subtrees      : %s\n", comment, args.collapse_subtree?"TRUE":"FALSE");
    fprintf(fh, "%sDynamic Bounds         : %s\n", comment, args.dynamic_bounds?"TRUE":"FALSE");
    fprintf(fh, "%sMinimum Examples       : %d\n", comment, args.minimum_examples);
    fprintf(fh, "%sSplit Method           : %s\n", comment, args.split_method==C45STYLE?"C45 Style":
                                                           (args.split_method==INFOGAIN?"Information Gain":
                                                           (args.split_method==GAINRATIO?"Gain Ratio":"N/A")));
    fprintf(fh, "%sSave Trees             : %s\n", comment, args.save_trees?"TRUE":"FALSE");
    if (args.random_forests > 0)
        fprintf(fh, "%sRandom Forests         : %d\n", comment, args.random_forests);
    if (args.extr_random_trees > 0)
        fprintf(fh, "%sExtremely Random Trees : %d\n", comment, args.extr_random_trees);
    if (args.totl_random_trees > 0)
        fprintf(fh, "%sTotally Random Trees   : %d\n", comment, args.totl_random_trees);
    //fprintf(fh, "%sRandom Attributes      : %d\n", comment, args.random_attributes);
    if (args.random_subspaces > 0.0)
        fprintf(fh, "%sRandom Subspaces       : %f\n", comment, args.random_subspaces);
    if (args.bag_size > 0.0)
        fprintf(fh, "%sBagging                : %.2f%%\n", comment, args.bag_size);
    if (args.do_ivote) {
        fprintf(fh, "%sIVoting                : %s\n", comment, args.do_ivote?"TRUE":"FALSE");
        fprintf(fh, "%sBite Size              : %d\n", comment, args.bite_size);
        if (args.num_trees > 0)
            fprintf(fh, "%sNumber of Trees        : %d\n", comment, args.num_trees);
        else if (args.auto_stop == TRUE)
            fprintf(fh, "%sNumber of Trees        : using stopping algorithm\n", comment);
    }
    fprintf(fh, "%sOutput Accuracies      : %s\n", comment, args.output_accuracies==ON?"TRUE":"FALSE");
    fprintf(fh, "%sOutput Predictions     : %s\n", comment, args.output_predictions?"TRUE":"FALSE");
    fprintf(fh, "%sOutput Probabilities   : %s\n", comment, args.output_probabilities?"TRUE":"FALSE");
    fprintf(fh, "%sOutput Confusion Matrix: %s\n", comment, args.output_confusion_matrix?"TRUE":"FALSE");
    
    fprintf(fh, "%s============================================================\n", comment);
}

