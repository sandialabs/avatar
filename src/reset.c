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
#include "version_info.h"
#include "crossval.h"
#include "reset.h"

void reset_CV_Class(CV_Class *cvc) {

    cvc->class_var_name    = NULL;
    cvc->num_classes       = 0;
    cvc->thresholds        = NULL;
    cvc->class_frequencies = NULL;
    cvc->class_names       = NULL;
    return;

}

void reset_CV_Dataset(CV_Dataset *cvd) {

    reset_CV_Metadata(&(cvd->meta));
    cvd->examples = NULL;
    return;

}

void reset_Exo_Data(Exo_Data *ed) {

    ed->num_seq_meshes = 0;
    ed->seq_meshes     = NULL;
    ed->assoc_type     = 0;
    ed->variables      = NULL; 
    return;

} 

void reset_CV_Example(CV_Example *cve) {

    cve->global_id_num        = 0;
    cve->random_gid           = 0;
    cve->fclib_seq_num        = 0;
    cve->fclib_id_num         = 0;
    cve->containing_class_num = 0;
    cve->predicted_class_num  = 0;
    cve->containing_fold_num  = 0;
    cve->bl_clump_num         = 0;
    cve->distinct_attribute_values = NULL;
    return;

}

void reset_CV_Metadata(CV_Metadata *cvm) {

    reset_Exo_Data(&(cvm->exo_data));
    cvm->num_classes    = 0;
    cvm->num_attributes = 0;
    cvm->num_fclib_seq  = 0; 
    cvm->num_examples   = 0;
    cvm->attribute_names        = NULL; 
    cvm->num_discrete_values    = NULL;
    cvm->discrete_attribute_map = NULL; 
    cvm->class_names            = NULL; 
    cvm->num_examples_per_class = NULL; 
    cvm->attribute_types        = NULL;
    cvm->global_offset          = NULL;     
    cvm->Missing                = NULL;
    return;

} 

void reset_CV_Subset(CV_Subset *cvs) {

    reset_CV_Metadata(&(cvs->meta));
    cvs->malloc_examples  = 0;
    cvs->discrete_used    = NULL;
    cvs->float_data       = NULL;
    cvs->high             = NULL;
    cvs->low              = NULL; 
    cvs->smote_float_data = NULL;
    cvs->smote_high       = NULL;
    cvs->smote_low        = NULL;
    cvs->weights          = NULL;
    cvs->examples         = NULL;
    return;

}

void reset_DT_Ensemble(DT_Ensemble *dte) {
    if(!dte) return;
    dte->Trees          = NULL;
    dte->boosting_betas = NULL;
    dte->Books          = NULL;
    dte->num_trees             = 0;
    dte->num_classes           = 0;
    dte->num_attributes        = 0;
    dte->num_training_examples = 0;
    dte->num_training_examples_per_class = NULL;
    dte->attribute_types                 = NULL;
    dte->weights                         = NULL;
    dte->Missing                         = NULL;
    return;
}

void reset_DT_Node(DT_Node * n) {
  if(!n) return;
  n->branch_type = 0;
  n->attribute = 0;
  n->num_branches = 0;
  n->num_errors = 0;
  n->branch_threshold = 0;
  n->attribute_type = 0;
  n->class_count = 0;
  n->class_probs = 0;
  n->Node_Value.class_label =0;
}

void reset_Args_Opts(Args_Opts *d) {
    if(!d) return;
    d->go4it = 0;
    
    //Boolean do_ts_based
    d->do_5x2_cv=0;
    d->do_nfold_cv=0;
    d->do_loov=0;
    d->do_rigorous_strat=0;
    d->num_folds=0;
    d->num_train_times=0;
    d->train_times=0;
    d->num_test_times=0;
    d->test_times=0;
    d->write_folds=0;
    d->random_seed=0;
    d->class_var_name=0;
    d->format=0;
    d->datafile=0;
    d->base_filestem=0;
    d->data_path=0;        // Derived argument
    d->classes_filename=0;
    d->partitions_filename=0;
    d->verbosity=0;
    d->print_version=0;
    
    d->caller=0;
    d->do_training=0;
    d->do_testing=0;
    
    // User customization options
    d->truth_column=0;
    d->num_skipped_features=0;
    d->skipped_features=0;
    d->exclude_all_features_above=0;
    d->num_explicitly_skipped_features=0;
    d->explicitly_skipped_features=0;
    
    // Decision Tree Generation options
    d->num_trees=0;
    d->auto_stop=0;
    d->slide_size=0;
    d->build_size=0;
    d->split_on_zero_gain=0;
    d->subsample=0;
    d->dynamic_bounds=0;
    d->minimum_examples=0;
    d->split_method=0;
    d->save_trees=0;
    d->random_forests=0;
    d->extr_random_trees=0;
    d->totl_random_trees=0;
    d->random_attributes=0;
    d->random_subspaces=0;
    d->collapse_subtree=0;
    d->do_bagging=0;
    d->bag_size=0;
    d->majority_bagging=0;
    
    // ivote options
    d->do_ivote=0;
    d->bite_size=0;
    d->ivote_p_factor=0;
    d->majority_ivoting=0;
    
    // SMOTE options
    d->do_smote=0;
    //d->majority_class=0;
    d->smote_knn=0;
    d->smote_Ln=0;
    d->smote_type=0;
    
    // Boosting and SMOTEBoost options
    d->do_boosting=0;
    d->do_smoteboost=0;
    d->smoteboost_type=0;
    
    // balanced learning options
    d->do_balanced_learning=0;
    
    // skew data handling options
    d->num_minority_classes=0;
    d->minority_classes=0;
    d->minority_classes_char=0;
    d->proportions=0;
    d->actual_proportions=0;
    d->actual_class_attendance=0;
    
    // rfFeatureValue options
    d->do_noising=0;
    
    // diversity options
    d->kappa_plot_data=0;
    
    // deviation type for standardizing outlier metric in proximity
    d->deviation_type=0;
    d->sort_line_num=0;
    d->print_prox_prog=0;
    d->save_prox_matrix=0;
    d->load_prox_matrix=0;
    
    // Alternate filenames]
    d->train_file=0;
    d->train_file_is_a_string=0;
    d->train_string=0;
    d->names_file=0;
    d->names_string=0;
    d->names_file_is_a_string=0; // if nonzero, then names_file is a string from which to read the names, not a filename to open
    d->test_file=0;
    d->test_string=0;
    d->test_file_is_a_string=0;
    d->trees_file=0;
    d->trees_file_is_a_string=0;
    d->trees_string=0;
    d->predictions_file=0;
    d->oob_file=0;
    d->prox_sorted_file=0;
    d->prox_matrix_file=0;
    
    // Output option
    d->output_accuracies=0;
    d->output_predictions=0;
    d->output_probabilities=0;
    d->output_laplacean=0;
    d->output_probabilities_warning=0;
    d->output_confusion_matrix=0;
    d->tree_stats_file=0;
    d->output_verbose_oob=0;
    d->output_margins=0;
    
    // Ensemble handling options
    d->do_mass_majority_vote=0;
    d->do_ensemble_majority_vote=0;
    d->do_margin_ensemble_majority_vote=0;
    d->do_probabilistic_majority_vote=0;
    d->do_scaled_probabilistic_majority_vote=0;

    // Unpublished options
    d->debug=0;
    d->read_folds=0;
    d->use_opendt_shuffle=0;
    d->run_regression_test=0;
    d->break_ties_randomly=0;
    d->stopping_algorithm_regtest=0;
    d->show_per_process_stats=0;
    d->common_mpi_rand48_seed=0;
    
    // Derived element for MPI
    d->mpi_rank=0;
    
}

void reset_CV_Matrix(CV_Matrix * m) {
  if(!m) return;
  m->data=0;
  m->classes=0;
  m->num_examples=0;
  m->num_classifiers=0;
  m->additional_cols=0;
  m->num_classes=0;
}
