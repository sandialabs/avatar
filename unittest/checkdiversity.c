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
#include <string.h>
#include <stdio.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/rw_data.h"
#include "../src/tree.h"
#include "../src/options.h"
#include "../tools/diversity_measures.h"

static CV_Subset Subset;
static DT_Ensemble Ensemble;
static Args_Opts Args;
static int **result_matrix;

void _read_data_and_trees_init_matrix() {
    AV_SortedBlobArray Sorted_Examples;
    FC_Dataset ds = {0};
    CV_Dataset Dataset = {0};
    memset(&Args, 0, sizeof(Args_Opts));
    memset(&Subset, 0, sizeof(CV_Subset));
    memset(&Ensemble, 0, sizeof(DT_Ensemble));
    Args.caller = DIVERSITY_CALLER;
    Args.format = AVATAR_FORMAT;
    Args.base_filestem = av_strdup("diversity_test");
    Args.data_path = av_strdup("./data");
    Args.datafile = av_strdup("./data/diversity_test.data");
    Args.truth_column = 5;
    Args.kappa_plot_data = TRUE;
    Args.save_trees = TRUE; // This prevents the dotted temp filename from being used for the trees file
    
    set_output_filenames(&Args, TRUE, TRUE);
    av_exitIfError(av_initSortedBlobArray(&Sorted_Examples));
    read_testing_data(&ds, Dataset.meta, &Dataset, &Subset, &Sorted_Examples, &Args);
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    read_ensemble(&Ensemble, -1, 0, &Args);
    init_matrix(Ensemble, Subset, &result_matrix);
}

START_TEST(check_kappa)
{
    _read_data_and_trees_init_matrix();
    double kappa = compute_dietterich_kappa(Ensemble.num_trees, Subset.meta.num_classes, Subset.meta.num_examples,
                                            result_matrix, Args);
    fail_unless(av_eqf(kappa, 0.8372805670872627), "average kappa incorrect");
    // Check plot data
    FILE *fh_test, *fh_truth;
    fail_unless((fh_test = fopen("./data/diversity_test_kappa.plot", "r")) != NULL, "Could not read kappa plot file");
    fh_truth = fopen("./data/diversity_test_kappa.truth", "r");
    char test_str[128];
    char truth_str[128];
    while (fgets(truth_str, 120, fh_truth) != NULL) {
        fail_unless(fgets(test_str, 120, fh_test) != NULL && ! strncmp(truth_str, test_str, strlen(truth_str)),
                    "Plot data incorrect");
    }
    fail_unless(fgets(test_str, 120, fh_test) == NULL, "plot file is too long");
    fclose(fh_test);
    fclose(fh_truth);
}
END_TEST

START_TEST(check_Q)
{
    _read_data_and_trees_init_matrix();
    double Q = compute_Q_statistic(Ensemble.num_trees, Subset.meta.num_examples, result_matrix);
    fail_unless(av_eqf(Q, 0.8118253057747596), "Q value incorrect");
}
END_TEST

START_TEST(check_interrater)
{
    _read_data_and_trees_init_matrix();
    double kappa = compute_interrater_kappa(Ensemble.num_trees, Subset.meta.num_examples, result_matrix);
    fail_unless(av_eqf(kappa, 0.4321642597504667), "kappa value incorrect");
}
END_TEST

START_TEST(check_PCDM)
{
    _read_data_and_trees_init_matrix();
    double PCDM = compute_pcdm(Ensemble.num_trees, Subset.meta.num_examples, result_matrix);
    fail_unless(av_eqf(PCDM, 0.36), "PCDM value incorrect");
}
END_TEST

Suite *diversity_suite(void)
{
    Suite *suite = suite_create("DiversityMeasures");
    
    TCase *tc_diversity_measures = tcase_create(" Check DiversityMeasures ");
    
    suite_add_tcase(suite, tc_diversity_measures);
    tcase_add_test(tc_diversity_measures, check_kappa);
    tcase_add_test(tc_diversity_measures, check_Q);
    tcase_add_test(tc_diversity_measures, check_interrater);
    tcase_add_test(tc_diversity_measures, check_PCDM);
        
    return suite;
}
