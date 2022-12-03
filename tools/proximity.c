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
#include "../src/reset.h"
#include "../src/options.h"
#include "../src/version_info.h"
#include "../src/rw_data.h"
#include "../src/tree.h"
#include "../src/evaluate.h"
#include "../src/array.h"
#include "../src/util.h"
#include "proximity_utils.h"

int main(int argc, char **argv) {
    int i;
    DT_Ensemble Ensemble;
    Args_Opts Args;
    AV_SortedBlobArray Sorted_Examples;
    FC_Dataset ds;
    CV_Dataset Dataset;
    CV_Subset Subset;

    reset_DT_Ensemble(&Ensemble) ;
    reset_CV_Dataset(&Dataset) ;
    reset_CV_Subset(&Subset) ;

    int **node_matrix;
    Prox_Matrix *prox_matrix = NULL;
    int *voted_class;
    float *outlier_metric;
    
    Args = process_opts(argc, argv);
    Args.do_testing = TRUE;
    Args.caller = PROXIMITY_CALLER;
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
    init_node_matrix(&Ensemble, &Subset, &node_matrix);
    if (Args.load_prox_matrix == TRUE) {
        // Load the proximity matrix from disk
        printf("Loading proximity matrix from '%s' ... ", Args.prox_matrix_file);
        fflush(NULL);
        read_proximity_matrix(&prox_matrix, Args);
        printf("Done\n");
        printf("Length of matrix = %d\n", LL_length(prox_matrix));
    } else {
        // Generate the linked list holding the proximity matrix
        assemble_prox_matrix(Subset.meta.num_examples, Ensemble.num_trees, node_matrix, &prox_matrix, Args.print_prox_prog);
        printf("Length of matrix = %d\n", LL_length(prox_matrix));
        if (Args.save_prox_matrix == TRUE) {
            printf("Saving proximity matrix to '%s' ... ", Args.prox_matrix_file);
            fflush(NULL);
            int should, did;
            if ( (did = write_proximity_matrix(prox_matrix, Args)) != (should = LL_length(prox_matrix)) )
                fprintf(stderr, "\nERROR: Wrote %d of %d nodes to proximity matrix file\n", did, should);
            else
                printf("Done\n");
        }
    }
    // Generate an array holding the truth class for each sample
    voted_class = (int *)malloc(Subset.meta.num_examples * sizeof(int));
    for (i = 0; i < Subset.meta.num_examples; i++)
        voted_class[i] = Subset.examples[i].containing_class_num;
    
    // Compute outlier metric
    outlier_metric = (float *)malloc(Subset.meta.num_examples * sizeof(float));
    compute_outlier_metric(Subset.meta.num_examples, 
                           Subset.meta.num_classes, 
                           voted_class, 
                           prox_matrix, 
                           &outlier_metric, 
                           Args.deviation_type);
    
    // Store result ...
    
    // Sort, if needed
    int *sample = NULL;
    float *proximity = NULL;
    int num_valid_prox_values = 0; //. This is the number of samples with a valid proximity value
    // If Args.sort_line_num == 0 then this is the default so use the first example which is example 0
    // If Args.sort_line_num > 0 then this is the 1-based line number specified by the user
    //     subtract 1 to get the 0-based example number
    if (Args.sort_line_num >= 0) {
        if (Args.sort_line_num > 0)
            Args.sort_line_num--;
        
        sample = (int *)malloc((Subset.meta.num_examples-1) * sizeof(int));
        proximity = (float *)malloc((Subset.meta.num_examples-1) * sizeof(float));
        // Initialize sample and proximity to -1
        for (i = 0; i < Subset.meta.num_examples-1; i++) {
            proximity[i] = -1.0;
            sample[i] = -1.0;
        }
        
        // Compile two arrays: an int array of sample number and a float array of proximity to sort_line_num
        Prox_Matrix *current;
        current = prox_matrix;
        while (current != NULL) {
            // if row xor col is sort_line_num then store this proximity
            if (current->row != current->col) {
                if (current->row == Args.sort_line_num) {
                    //printf("Proximity between %5d and %5d = %f\n", current->row, current->col, current->data);
                    sample[num_valid_prox_values] = current->col;
                    proximity[num_valid_prox_values] = current->data;
                    num_valid_prox_values++;
                } else if (current->col == Args.sort_line_num) {
                    //printf("Proximity between %5d and %5d = %f\n", current->col, current->row, current->data);
                    sample[num_valid_prox_values] = current->row;
                    proximity[num_valid_prox_values] = current->data;
                    num_valid_prox_values++;
                }
            }
            current = current->next;
        }
        float_int_array_sort(Subset.meta.num_examples-1, proximity-1, sample-1);
    }
    //printf("Should print %d samples\n", num_valid_prox_values);
    //int counts = 0;
    //for (i = 0; i < Subset.meta.num_examples-1; i++) {
    //    if (sample[i] > -1) {
    //        counts++;
    //        printf("Sample %5d has proximity = %f\n", sample[i], proximity[i]);
    //    }
    //}
    //printf("Did print %d samples\n", counts);
    
    // Print to output file
    
    if (Args.format == AVATAR_FORMAT) {
        // Load the datafile as strings
        char **data_strings, **comment_strings;
        int num_data_strings, num_comment_strings;
        FILE *fh;
        datafile_to_string_array(Args.test_file, &num_data_strings, &data_strings, &num_comment_strings, &comment_strings);
        if ((fh = fopen(Args.prox_sorted_file, "w")) == NULL) {
            fprintf(stderr, "ERROR: Could not write to '%s'\n", Args.prox_sorted_file);
            exit(-1);
        }
        
        int num_tokens;
        char **label_tokens;
        for (i = 0; i < num_comment_strings; i++) {
            // Look for a #labels comment
            Boolean found_label = FALSE;
            parse_space_sep_string(comment_strings[i], &num_tokens, &label_tokens);
            if (num_tokens > 0 && strlen(label_tokens[0]) > 6) {
                char *junk;
                junk = label_tokens[0] + strlen(label_tokens[0]) - 6;
                if (! strcasecmp(junk, "labels"))
                    found_label = TRUE;
            } else if (num_tokens > 1 && ! strcasecmp(label_tokens[1], "labels")) {
                found_label = TRUE;
            }
            if (found_label == TRUE) {
                fprintf(fh, "%s,Outlier Measure", comment_strings[i]);
                if (Args.sort_line_num >= 0)
                    fprintf(fh, ",Proximity");
                fprintf(fh, "\n");
            } else {
                fprintf(fh, "%s\n", comment_strings[i]);
            }
        }
        // Print datalines
        if (Args.sort_line_num >= 0) {
            // Start with the probe line
            //printf("print probe line\n");
            fprintf(fh, "%s,%f,%f\n", data_strings[Args.sort_line_num], outlier_metric[Args.sort_line_num], 1.0);
            // Then all lines with a defined proximity
            // The list is sorted small to big so write the list backwards
            for (i = Subset.meta.num_examples-2; i >= 0; i--) {
                if (sample[i] > -1) {
                    //printf("%5d,%s,%f,%f\n", sample[i], data_strings[sample[i]], outlier_metric[sample[i]], proximity[i]);
                    fprintf(fh, "%s,%f,%f\n", data_strings[sample[i]], outlier_metric[sample[i]], proximity[i]);
                }
            }
            // Then all the ones we skipped
            for (i = 0; i < num_data_strings; i++) {
                // Only sort the sample array the first time through. It needs to be sorted because it was
                // manipulated based on proximity previously.
              if (i != Args.sort_line_num && ! find_int(i, Subset.meta.num_examples-1, sample)) {
                    //printf("%5d,%s,%f,%f\n", i, data_strings[i], outlier_metric[i], -1.0);
                    fprintf(fh, "%s,%f,%f\n", data_strings[i], outlier_metric[i], -1.0);
                }
            }
        } else {
            for (i = 0; i < num_data_strings; i++)
                fprintf(fh, "%s,%f\n", data_strings[i], outlier_metric[i]);
        }
        
    } else if (Args.format == EXODUS_FORMAT) {
        FC_Dataset ods;
        init_outlier_file(Subset.meta, &ods, Args);
        store_outlier_values(Subset, outlier_metric, proximity, sample, ods, Args);
    }
    
    return 0;
}

void display_usage( void ) {
    printf("\ndiversity ");
    printf("%s", get_version_string());
    printf("\n");
    printf("\nUsage: diversity options\n");
    printf("\nbasic arguments:\n");
    printf("    -o, --format=FMTNAME : Data format. FMTNAME is either 'exodus' or 'avatar'\n");
    printf("                           Default = 'exodus'\n");
    printf("    --standard-deviation : Use the standard deviation for standardizing the outlier\n");
    printf("                           measure [DEFAULT]\n");
    printf("    --absolute-deviation : Use the absolute deviation for standardizing the outlier\n");
    printf("                           measure\n");
    printf("    --sort=LINE          : Generate a sorted data file. The first sample in the new\n");
    printf("                           datafile is the sample at line LINE (or the first sample\n");
    printf("                           if LINE is not specified) and the rest of the samples\n");
    printf("                           are sorted by proximity to the first.\n");
    printf("    --load-matrix        : Load the pre-computed proximity matrix from the file\n");
    printf("                           specified by the --prox-matrix-file option or from the\n");
    printf("                           default which is the datafile basename with an extension\n");
    printf("                           of 'proximity_matrix'\n");
    printf("    --no-save-matrix     : When computed (i.e. NOT read from disk), the proximity\n");
    printf("                           matrix is saved to disk by default. This option overrides\n");
    printf("                           this and the matrix is NOT saved to disk\n");
    printf("    --print-proximity-progress : Print progress of proximity matrix computation\n");
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
    printf("    --trees-file=FILE       : For avatar format data, use FILE for the ensemble file\n");
    printf("    --prox-sorted-file=FILE : Use FILE as the output file\n");
    printf("    --prox-matrix-file=FILE : Use FILE as the file for storing/reading the proximity\n");
    printf("                              matrix\n");
    printf("\n");
    exit(-1);
}

