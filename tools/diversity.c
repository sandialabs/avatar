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
#include "../src/crossval.h"
#include "../src/options.h"
#include "../src/version_info.h"
#include "../src/rw_data.h"
#include "../src/tree.h"
#include "diversity_measures.h"

int main(int argc, char **argv) {
    DT_Ensemble Ensemble;
    Args_Opts Args;
    AV_SortedBlobArray Sorted_Examples;
    FC_Dataset ds;
    CV_Dataset Dataset;
    CV_Subset Subset;
    int **result_matrix;
    
    Args = process_opts(argc, argv);
    Args.do_testing = TRUE;
    Args.caller = DIVERSITY_CALLER;
    if (! sanity_check(&Args))
        exit(-1);
    set_output_filenames(&Args, FALSE, FALSE);
    
    if (Args.format == EXODUS_FORMAT && ! Args.do_training) {
        init_fc(Args);
        open_exo_datafile(&ds, Args.datafile);
    }
    av_exitIfError(av_initSortedBlobArray(&Sorted_Examples));
    read_testing_data(&ds, Subset.meta, &Dataset, &Subset, &Sorted_Examples, &Args);
    late_process_opts(Dataset.meta.num_attributes, Dataset.meta.num_examples, &Args);
    
    read_ensemble(&Ensemble, -1, 0, &Args);
    
    // Initialize matrix holding per-tree results
    init_matrix(Ensemble, Subset, &result_matrix);
    printf("Average Kappa     = %f\n", compute_dietterich_kappa(Ensemble.num_trees, Subset.meta.num_classes,
                                                                Subset.meta.num_examples, result_matrix, Args));
    printf("Q Statistic       = %f\n", compute_Q_statistic(Ensemble.num_trees, Subset.meta.num_examples,
                                                           result_matrix));
    printf("Inter-rater Kappa = %f\n", compute_interrater_kappa(Ensemble.num_trees, Subset.meta.num_examples,
                                                                result_matrix));
    printf("PCDM              = %f\n", compute_pcdm(Ensemble.num_trees, Subset.meta.num_examples, result_matrix));
    return 0;
}

void display_usage( void ) {
    printf("\ndiversity ");
    printf(get_version_string());
    printf("\n");
    printf("\nUsage: diversity options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME     : Data format. FMTNAME is either 'exodus' or 'avatar'\n");
    printf("                               Default = 'exodus'\n");
    printf("    --output-kappa-plot-data : Write a plot file containing the following four columns:\n");
    printf("                               1) classifier 1 number\n");
    printf("                               2) classifier 2 number\n");
    printf("                               3) Kappa value for the two classifiers\n");
    printf("                               4) Average error for the two classifiers\n");
    printf("\n");
    printf("required exodus-specific arguments:\n");
    printf("    -d, --datafile=FILE     : Use FILE as the exodus datafile\n");
    printf("    --test-times=R          : Use data from the range of times R to test (e.g. 5,7-10)\n");
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
    printf("avatar-specific options:\n");
    printf("    --include=R      : Include the features listed in R (e.g. 1-4,6)\n");
    printf("    --exclude=R      : Exclude the features listed in R (e.g. 1-4,6)\n");
    printf("                       --include and --exclude may be specified multiple times.\n");
    printf("                       They are applied left to right.\n");
    printf("    --truth-column=S : Location of the truth column. S = first or last\n");
    printf("                       Default = last\n");
    printf("\n");
    printf("alternate filenames:\n");
    printf("    --names-file=FILE       : For avatar format data, use FILE for the names file\n");
    printf("    --test-file=FILE        : For avatar format data, use FILE for testing data\n");
    printf("    --trees-file=FILE       : For avatar format data, use FILE for ensemble file\n");
    printf("\n");
    exit(-1);
}

