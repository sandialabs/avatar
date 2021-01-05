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
#include "crossval.h"
#include "skew.h"
#include "distinct_values.h"
#include "array.h"
#include "smote.h"

/*
 * Populates Args_Opts:minority_classes with class id for specified minority classes
 * Also populates Args_Opts:actual_class_attendance and Args_Opts:actual_proportions
 *
 * Input values:
 * CV_Metadata meta
 * Args_Opts args
 *
 * Output values:
 * Args_Opts args (updated)
 */
void decode_minority_class_names(CV_Metadata meta, Args_Opts *args) {
    int i;
    args->minority_classes = (int *)malloc(args->num_minority_classes * sizeof(int));
    for (i = 0; i < args->num_minority_classes; i++)
        args->minority_classes[i] = translate_discrete(meta.class_names, meta.num_classes, args->minority_classes_char[i]);
    update_actual_att_props(1, meta, args);
}

void update_actual_att_props(int init, CV_Metadata meta, Args_Opts *args) {
    int i;
    if (init) {
        args->actual_proportions = (float *)malloc(meta.num_classes * sizeof(float));
        args->actual_class_attendance = (int *)malloc(meta.num_classes * sizeof(int));
    }
    if (args->do_smote == TRUE) {
        int ns;
        int *deficits;
        compute_deficits(meta, *args, &ns, &deficits);
        for (i = 0; i < meta.num_classes; i++) {
            args->actual_class_attendance[i] = meta.num_examples_per_class[i] + deficits[i];
            args->actual_proportions[i] = (float)args->actual_class_attendance[i]/(float)ns;
        }
    } else {
        int nc, nspc;
        compute_number_of_clumps(meta, args, &nc, &nspc);
        for (i = 0; i < meta.num_classes; i++) {
            args->actual_class_attendance[i] = _num_per_class_per_clump(i, nc, meta, *args);
            args->actual_proportions[i] = (float)args->actual_class_attendance[i]/(float)nspc;
        }
    }
}

/*
 * Compute the number of clumps required to achieve the target class proportions
 * 
 * Input values:
 * CV_Metadata meta
 * Args_Opts args
 *
 * Return values:
 * int *num_c = the number of clumps required
 * int *num_s = the number of samples in each clump
 */
void compute_number_of_clumps(CV_Metadata meta, Args_Opts *args, int *num_c, int *num_s) {
    int i;
    *num_c = 0;
    float min_total_prop = 0.0; // Sum of minority class proportions
    static int first_time_through = 1; // Only print warnings the first time through
    
    // Compute average of total numbers to get each minority class to correct percentage
    float N_temp = 0;
    for (i = 0; i < args->num_minority_classes; i++) {
        float this_N = ((float)meta.num_examples_per_class[args->minority_classes[i]])/args->proportions[i];
        // Warn if a population decrease would be needed to achieve proportions
        if (first_time_through && (int)(this_N + 0.5) > meta.num_examples) {
            first_time_through = 0;
            fprintf(stderr, "WARNING: Specified proportions for class %d ('%s') are too small and unattainable\n",
                             args->minority_classes[i], meta.class_names[args->minority_classes[i]]);
        }
        N_temp += this_N;
        min_total_prop += args->proportions[i];
    }
    N_temp /= args->num_minority_classes;
    *num_s = (int)(N_temp + 0.5);
    
    // Compute proportions of majority classes in original data
    int sum_maj = 0;
    for (i = 0; i < meta.num_classes; i++)
      if (! find_int(i, args->num_minority_classes, args->minority_classes))
            sum_maj += meta.num_examples_per_class[i];
    for (i = 0; i < meta.num_classes; i++) {
      if (! find_int(i, args->num_minority_classes, args->minority_classes)) {
            // Percentage of majority classes for this majority class
            float p_of_m = ((float)meta.num_examples_per_class[i])/(float)sum_maj;
            // Target for each partition
            float new_p = p_of_m * (1.0 - min_total_prop);
            // Number of samples of this class in each partition
            int N_new = ceilf(*num_s * new_p);
            // Number of clumps this class needs
            int np = (int)ceilf(((float)meta.num_examples_per_class[i])/(float)N_new);
            // Use the largest number of clumps
            if (np > *num_c)
                *num_c = np;
        }
    }
    find_int_release();
    
    // Given max number of clumps, recalculate the number of samples per clump
    *num_s = 0;
    for (i = 0; i < meta.num_classes; i++) {
        *num_s += _num_per_class_per_clump(i, *num_c, meta, *args);
    }
    
    // Update bag/bite size if necessary
    if (args->majority_bagging == TRUE)
        args->bag_size = ((float)*num_s * 100.0)/(float)meta.num_examples;
    else if (args->majority_ivoting == TRUE)
        args->bite_size = *num_s;
    
    // If we're doing balanced learning, make sure num_s > bite_size or bag_size
    if (args->do_balanced_learning == TRUE) {
        if (args->do_ivote == TRUE && args->bite_size > *num_s) {
            fprintf(stderr, "ERROR: Bite size is too large. Maximum value is %d\n", *num_s);
            exit(-1);
        }
        if (args->do_bagging == TRUE && (float)(args->bag_size * meta.num_examples)/100.0 > *num_s) {
            fprintf(stderr, "ERROR: Bag size is too large. Maximum value is %.2f%%\n",
                            ((float)*num_s * 100.0)/(float)meta.num_examples);
            exit(-1);
        }
    }
}
/*
 * Returns the number of samples in each clump for the specified class
 *
 * Input values:
 * int class = the specified class
 * int num_clumps = the number of clumps [gotten from compute_number_of_clumps()]
 * CV_Metadata meta
 *
 * Output values: NONE
 *
 */
int _num_per_class_per_clump(int class, int num_clumps, CV_Metadata meta, Args_Opts args) {
    int num = -1;
    // Minority classes get all their examples
    // Majority classes get divied up into num_clumps clumps
    if (find_int(class, args.num_minority_classes, args.minority_classes))
        num = meta.num_examples_per_class[class];
    else
        num = (int)ceilf(((float)meta.num_examples_per_class[class])/(float)num_clumps);
    find_int_release();
    return num;
}
