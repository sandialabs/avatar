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
#ifndef __CROSSVAL__
#define __CROSSVAL__

#include "av_utils.h"
#include "datatypes.h"
#include "schema.h"

typedef enum {
    OPEN_SMOTE = 0,
    CLOSED_SMOTE = 1
} Smote_Type;

typedef enum {
    ALL_MINORITY_CLASSES = 0,
    FROM_BOOSTED_SET_ONLY = 1
} SmoteBoost_Type;

typedef enum {
    LIMBO = -1,
    OFF = 0,
    ON = 1,
    VERBOSE = 2
} Tristate;

typedef enum {
    UNKNOWN_CALLER,
    CROSSVALFC_CALLER,
    AVATARDT_CALLER,
    AVATARMPI_CALLER,
    RFFEATUREVALUE_CALLER,
    DIVERSITY_CALLER,
    PROXIMITY_CALLER
} Caller;

typedef enum {
    TEST_MODE,
    TRAIN_MODE
} CV_Mode;


typedef enum {
    CV_NFOLD,            // N-fold cross validation
    CV_5x2,              // 5x2 cross validation
    TS_BASED             // Timestep-based train/test
} Fold_Type;


typedef enum {
    BRANCH,
    LEAF
} Branch_Type;

//Modified by DACIESL June-02-08: HDDT CAPABILITY
//Added HELLINGER
typedef enum {
    INFOGAIN,
    GAINRATIO,
    C45STYLE,
    HELLINGER
} Split_Method;

typedef enum {
    ALPHA,
    OMEGA
} Alpha_Omega;

// For proximity
typedef enum {
    STANDARD_DEVIATION,
    ABSOLUTE_DEVIATION
} Deviation_Type;

typedef struct tree_bookkeeping_struct {
    int num_malloced_nodes;
    int next_unused_node;
    int current_node;
} Tree_Bookkeeping;

typedef struct bst_node_struct {
    Tree_Bookkeeping Books;
    int left;
    int right;
    float value;
} BST_Node;

typedef struct class_metadata_struct {
    char *class_var_name;
    int num_classes;
    float *thresholds;
    int *class_frequencies;
    char **class_names;
} CV_Class;

typedef struct partition_metadata_struct {
    int num_partitions;
    char **partition_datafile;
    char **partition_data_path;
    char **partition_base_filestem;
} CV_Partition;

typedef struct data_example_struct {
    int global_id_num;
    int random_gid;
    int fclib_seq_num;
    int fclib_id_num;
    int containing_class_num;
    int predicted_class_num;
    int containing_fold_num;
    int bl_clump_num;
    Boolean is_missing;
    Boolean in_bag;
    int *distinct_attribute_values;
} CV_Example;

union data_type_union {
    float Real;
    int Integer;
};

typedef struct exodus_data_struct {
    int num_seq_meshes;
    FC_Mesh *seq_meshes;
    FC_AssociationType assoc_type;
    FC_Variable **variables; // The FC_Variable associated with each sequence and attribute:
                             // seq_variables[fclib_seq][att_num]
} Exo_Data;

typedef struct crossval_metadata_struct {
    int num_classes;
    int num_attributes;
    int num_fclib_seq;      // Number of fclib sequences (i.e. number of meshes over all timesteps)
    int num_examples;
    char **attribute_names; // Array holding the names of the attributes
    int *num_discrete_values;
    char ***discrete_attribute_map; // dam[i][j] gives the attribute label for the jth value of the ith attribute
    char **class_names; // Array holding the names of the classes
    int *num_examples_per_class; // Array holding how many examples are in each class.
    Attribute_Type *attribute_types;
    int *global_offset;     // Global offset for start of each fclib sequence
                            // Total number of examples is then given by global_offset[num_fclib_seq]
    Exo_Data exo_data;
    union data_point_union *Missing;
} CV_Metadata;

/* CV_Dataset and CV_Subset should hold CV_Metadata instead of the fields themselves */
typedef struct crossval_dataset_struct {
    CV_Metadata meta;
    CV_Example *examples;
} CV_Dataset;

typedef struct crossval_sub_dataset_struct {
    CV_Metadata meta;
    int malloc_examples;
    Boolean *discrete_used;
    CV_Example *examples;
    float **float_data;     // Translates indexed ints back to floats for data values
    int *high;              // Index of largest value in this subset in discrete_data array
    int *low;               // Index of smallest value in this subset in discrete_data array
    float **smote_float_data;
    int *smote_high;
    int *smote_low;
    double *weights;
} CV_Subset;


//Modified by DACIESL June-03-08: Laplacean Estimates
//Added class_count and class_probs
typedef struct dt_node_struct {
    Branch_Type branch_type;
    int attribute;
    int num_branches;
    int num_errors;
    float branch_threshold;
    Attribute_Type attribute_type;
    int *class_count;
    float *class_probs;
    union node_value_union {
        int class_label;
        int *branch;
    } Node_Value;
} DT_Node;

typedef struct dt_ensemble_struct {
    DT_Node **Trees;
    double *boosting_betas;
    Tree_Bookkeeping *Books;
    int num_trees;
    int num_classes;
    int num_attributes;
    int num_training_examples;
    int *num_training_examples_per_class;
    Attribute_Type *attribute_types;
    float *weights;
    union data_point_union *Missing;
} DT_Ensemble;

typedef struct crossval_matrix_struct {
    union data_type_union **data; // data[num_examples][num_classifiers + 1] = class
                                  // Boosting uses float and all others use int
    int *classes;   // THIS IS ONLY USED BY THE BOOSTING MATRIX
    int num_examples;
    int num_classifiers;
    int additional_cols; // For boosting = 0; otherwise = 1 to hold the truth
    int num_classes;
} CV_Matrix;

//typedef struct crossval_boost_matrix_struct {
//    float **data;     // data[num_examples][num_classifiers + 1] = class
//    int *classes;
//    int num_examples;
//    int num_classifiers;
//    int num_classes;
//} CV_Boost_Matrix;

typedef struct crossval_prob_matrix_struct {
    float **data;     // data[num_examples][num_classes] = class probability
    int num_examples;
    int num_classes;
} CV_Prob_Matrix;

typedef struct crossval_voting_struct {
    float **data;       // data[num_examples][num_classes] = some measure of voting for this class
    int *class;         // Holds true class for the example
    int *best_class;    // Holds the best class for the example
    int num_examples;
    int num_classes;
} CV_Voting;

//Modified by DACIESL June-11-08: Laplacean Estimates
//added float ** class_weighted_votes_test
//weighted voting (laplace estimates)
typedef struct voting_cache_struct {
    int num_classifiers;
    int current_classifier_count;    // Holds which iteration we're on
    int num_classes;
    int bite_size;
    int num_train_examples;
    int num_test_examples;
    float average_train_accuracy;    // a running average accuracy. updated with each iteration.
    float average_test_accuracy;
    double oob_error;                // = e(k) for ivoting
    double test_error;
    int **oob_class_votes;           // oob_class_votes[num_train_samples][num_classes]
                                     // holds number of votes for this class across all OOB trees
    int **class_votes_test;        // holds number of votes for this class across all trees for test data
    float ** oob_class_weighted_votes;
    float ** class_weighted_votes_test; //class_weighted_votes_test[num_train_samples][num_classes]
    int *best_train_class;           // majority_vote_winner[num_samples]
    int *best_test_class;            // Store results for test data since we're tossing the trees
} Vote_Cache;

//Modified by DACIESL June-03-08: Laplacean Estimates
//Added output_laplacean
//Modified by MAMUNSO Aug-04-10: added option for tree stats
//Modified by MEGOLDS August, 2012: subsampling
//Added integer subsample arg
typedef struct args_and_opts_struct {
    Boolean go4it;
    
    //Boolean do_ts_based;
    Boolean do_5x2_cv;
    Boolean do_nfold_cv;
    Boolean do_loov;
    Boolean do_rigorous_strat;
    int num_folds;
    int num_train_times;
    int *train_times;
    int num_test_times;
    int *test_times;
    Boolean write_folds;
    long random_seed;
    char *class_var_name;
    Data_Format format;
    char *datafile;
    char *base_filestem;
    char *data_path;        // Derived argument
    char *classes_filename;
    char *partitions_filename;
    int verbosity;
    Boolean print_version;
    
    Caller caller;
    int do_training;
    int do_testing;
    
    // User customization options
    int truth_column;
    int num_skipped_features;
    int *skipped_features;
    int exclude_all_features_above;
    int num_explicitly_skipped_features;
    int *explicitly_skipped_features;
    
    // Decision Tree Generation options
    int num_trees;
    int auto_stop;
    int slide_size;
    int build_size;
    Boolean split_on_zero_gain;
    int subsample;
    Boolean dynamic_bounds;
    int minimum_examples;
    Split_Method split_method;
    Boolean save_trees;
    int random_forests;
    int extr_random_trees;
    int totl_random_trees;
    int random_attributes;
    float random_subspaces;
    Boolean collapse_subtree;
    Boolean do_bagging;
    float bag_size;
    Boolean majority_bagging;
    
    // ivote options
    Boolean do_ivote;
    int bite_size;
    float ivote_p_factor;
    Boolean majority_ivoting;
    
    // SMOTE options
    Boolean do_smote;
    //int *majority_class;
    int smote_knn;
    int smote_Ln;
    Smote_Type smote_type;
    
    // Boosting and SMOTEBoost options
    Boolean do_boosting;
    Boolean do_smoteboost;
    SmoteBoost_Type smoteboost_type;
    
    // balanced learning options
    Boolean do_balanced_learning;
    
    // skew data handling options
    int num_minority_classes;
    int *minority_classes;
    char **minority_classes_char;
    float *proportions;
    float *actual_proportions;
    int *actual_class_attendance;
    
    // rfFeatureValue options
    Boolean do_noising;
    
    // diversity options
    Boolean kappa_plot_data;
    
    // deviation type for standardizing outlier metric in proximity
    Deviation_Type deviation_type;
    int sort_line_num;
    Boolean print_prox_prog;
    Boolean save_prox_matrix;
    Boolean load_prox_matrix;
    
    // Alternate filenames]
    char *train_file;
    Boolean train_file_is_a_string;
    char *train_string;
    char *names_file;
    char *names_string;
    Boolean names_file_is_a_string; // if nonzero, then names_file is a string from which to read the names, not a filename to open
    char *test_file;
    char *test_string;
    Boolean test_file_is_a_string;
    char *trees_file;
    Boolean trees_file_is_a_string;
    char *trees_string;
    char *predictions_file;
    char *oob_file;
    char *prox_sorted_file;
    char *prox_matrix_file;
    
    // Output option
    Tristate output_accuracies;
    Boolean output_predictions;
    Boolean output_probabilities;
    Boolean output_laplacean;
    Boolean output_probabilities_warning;
    Boolean output_confusion_matrix;
    char *tree_stats_file;
    Boolean output_verbose_oob;
    Boolean output_margins;
    
    // Ensemble handling options
    Boolean do_mass_majority_vote;
    Boolean do_ensemble_majority_vote;
    Boolean do_margin_ensemble_majority_vote;
    Boolean do_probabilistic_majority_vote;
    Boolean do_scaled_probabilistic_majority_vote;

    // Unpublished options
    Boolean debug;
    Boolean read_folds;
    Boolean use_opendt_shuffle;
    Boolean run_regression_test;
    Boolean break_ties_randomly;
    Boolean stopping_algorithm_regtest;
    Boolean show_per_process_stats;
    Boolean common_mpi_rand48_seed;
    
    // Derived element for MPI
    int mpi_rank;
    
} Args_Opts;

#define NO_SPLIT strtod("NAN",(char**)NULL)

//int num_log_comps;
//int num_pot_log_comps;
//int max_log_lookup;

void assign_class_based_folds(int num_folds, AV_SortedBlobArray examples, int **population);
Boolean att_label_has_leading_number(char **names, int num, Args_Opts args);
void cv_class_print(CV_Class class);
void cv_dataset_print(CV_Dataset dataset, char *title);
void cv_example_print(CV_Example example, int num_atts);
//void cv_subset_print(CV_Subset subset, char *title);
void display_usage(void);
//void example_to_string(int num, CV_Subset data, int truth_col, char **string);
int get_class_index(double value, CV_Class *class);
void make_train_test_subsets(CV_Subset full, int fold_pop, int fold_num, CV_Subset *train_subset, CV_Subset *test_subset, AV_SortedBlobArray *blob);
//void write_as_data_file(char *filename, CV_Subset data);
void __print_datafile(CV_Subset data, char *prefix);

int cv_example_compare_by_seq_id(const void *n, const void *m);
int qsort_example_compare_by_class(const void *n, const void *m);
int cv_example_compare_by_class(const void *n, const void *m);
int fclib2global(int fclib_seq, int fclib_id, int num_seq, int *global_offset);

// REVIEW-2012-03-26-ArtM: Might be able to remove these; only used in unit tests at moment.
int check_class_attendance(int num_folds, CV_Class class, CV_Subset data, int verbose);
int check_fold_attendance(int num_folds, CV_Subset data, int verbose);
int global2fclib(int global_id, int num_seq, int *global_offset, int *fclib_seq, int *fclib_id);
int qsort_example_compare_by_seq_id(const void *n, const void *m);
int qsort_example_compare_by_fold(const void *n, const void *m);
int cv_example_compare_by_fold(const void *n, const void *m);
//int qsort_example_compare_by_gid(const void *n, const void *m);
//int cv_example_compare_by_gid(const void *n, const void *m);
//int qsort_example_compare_by_rgid(const void *n, const void *m);
//int cv_example_compare_by_rgid(const void *n, const void *m);


#endif // __CROSSVAL__
