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
#include <stdio.h>
#include <stdlib.h>
#include "crossval.h"
#include "array.h"
#include "util.h"

/*
 *  assign_class_based_folds
 *
 *  Assigns a random gid to each example, sorts on class then random gid,
 *  and then assigns fold numbers by stepping through the sorted list of
 *  examples. This assures that each class is present in each fold in
 *  (almost) the same proportion as in the entire dataset -- the differences
 *  being +/- 1 depending on how the number of examples divides the number
 *  of folds.
 */
 
void assign_class_based_folds(int num_folds, AV_SortedBlobArray examples, int **population) {
    int i;
    int *ran_id;
    ran_id = (int *)malloc(examples.numBlob * sizeof(int));
    for (i = 0; i < examples.numBlob; i++)
        ran_id[i] = i;
    _knuth_shuffle(examples.numBlob, ran_id);
    for (i = 0; i < examples.numBlob; i++) {
        CV_Example *blob = (CV_Example *)examples.blobs[i];
        blob->random_gid = ran_id[i];
    }
    free(ran_id);
    qsort(examples.blobs, examples.numBlob, sizeof(CV_Example *), qsort_example_compare_by_class);
    (*population) = (int *)calloc(num_folds, sizeof(int));
    for (i  = 0; i < examples.numBlob; i++) {
        CV_Example *blob = (CV_Example *)examples.blobs[i];
        blob->containing_fold_num = i % num_folds;
        (*population)[i % num_folds]++;
    }
}

Boolean att_label_has_leading_number(char **names, int num, Args_Opts args) {
    int i;
    char *stripped_label;
    int stripped_num;
    int skip_offset = 0;
    Boolean prefixed = TRUE;
    for (i = 0; i < num + args.num_skipped_features; i++) {
        int col = i+1;
        if (args.truth_column > 0 && col >= args.truth_column)
            col++;
        if (find_int(col, args.num_skipped_features, args.skipped_features)) {
            skip_offset++;
        } else {
            stripped_num = strtol(names[i-skip_offset], &stripped_label, 10);
            if (stripped_num != col || stripped_label[0] != ' ')
                prefixed = FALSE;
        }
        find_int_release();
    }
    return prefixed;
}


/*
 * check_fold_attendence
 *
 * Checks that the number of examples in each fold is correct.
 * Checks that each example is in exactly one fold.
 */
int check_fold_attendance(int num_folds, CV_Subset data, int verbose) {
    int i;
    int *ex_per_fold, *folds_per_ex;
    int nominal_target, num_deviants;
    int num_one_over;
    
    int res = 0;
    
    // This holds the number of folds each example belongs to
    folds_per_ex = (int *)calloc(data.meta.num_examples, sizeof(int));
    ex_per_fold = (int *)calloc(num_folds, sizeof(int));
    
    // Set to nominal value. num_deviants folds will have one more than the nominal value
    nominal_target = data.meta.num_examples / num_folds;
    num_deviants = data.meta.num_examples % num_folds;
    if (verbose == 2)
        printf("Expecting %d examples per fold with %d folds having %d examples\n", nominal_target, num_deviants,
                                                                                    nominal_target + 1);
    
    for (i = 0; i < data.meta.num_examples; i++) {
        int fold = data.examples[i].containing_fold_num;
        ex_per_fold[fold]++;
        folds_per_ex[i]++;
    }
    
    num_one_over = 0;
    for (i = 0; i < num_folds; i++) {
        if (ex_per_fold[i] > nominal_target) {
            if (ex_per_fold[i] > nominal_target + 1) {
                res += ex_per_fold[i] - nominal_target + 1;
                if (verbose > 0)
                    printf("Fold %d has too many examples: %d instead of %d(+1)\n", i, ex_per_fold[i], nominal_target);
            } else {
                num_one_over++;
            }
        } else if (ex_per_fold[i] < nominal_target) {
            res += nominal_target - ex_per_fold[i];
            if (verbose > 0)
                printf("Fold %d has too few examples: %d instead of %d(+1)\n", i, ex_per_fold[i], nominal_target);
        }
    }
    if (num_one_over != num_deviants) {
        res += (int)fabs((float)(num_deviants - num_one_over));
        if (verbose > 0)
            printf("%d instead of %d folds had %d examples\n", num_one_over, num_deviants, nominal_target + 1);
    }

    free(folds_per_ex);
    free(ex_per_fold);
    
    return res;
}


/*
 *  check_class_attendence
 *
 *  Check that the distribution of classes in each fold is the same as in the whole dastaset.
 *  But allow +/- 1 of each class in a fold as long as the differences add to 0
 */
int check_class_attendance(int num_folds, CV_Class class, CV_Subset data, int verbose) {
    int i, j;
    int **class_attend;
    float *target;
    int *fold_count;
    
    int res = 0;
    
    target = (float *)malloc(class.num_classes * sizeof(float));
    if (verbose > 1)
        printf("Target class populations:");
    for (j = 0; j < class.num_classes; j++) {
        target[j] = (float)class.class_frequencies[j] / (float)data.meta.num_examples;
        if (verbose > 1)
            printf(" %.4f,", target[j]);
    }
    if (verbose)
        printf("\b\n");
        
    fold_count = (int *)calloc(num_folds, sizeof(int));
    class_attend = (int **)malloc(num_folds * sizeof(int *));
    for (i = 0; i < num_folds; i++)
        class_attend[i] = (int *)calloc(class.num_classes, sizeof(int));
    
    for (i = 0; i < data.meta.num_examples; i++) {
        CV_Example e = data.examples[i];
        class_attend[e.containing_fold_num][e.containing_class_num]++;
        fold_count[e.containing_fold_num]++;
    }
    
    int target_total = 0;
    for (i = 0; i < num_folds; i++) {
        int sum_diffs = 0;
        for (j = 0; j < class.num_classes; j++) {
            target_total += (int)(fold_count[i] * target[j] + 0.5);
            int diff = (int)(fold_count[i] * target[j] + 0.5) - class_attend[i][j];
            // If |difference| is 2 or more then there is a problem.
            if (diff > 1 || diff < -1) {
                res += diff;
                if (verbose)
                    printf("Fold %d has %d in class %d instead of %d\n",
                            i, class_attend[i][j], j, (int)(fold_count[i] * target[j] + 0.5));
            // If |difference| = 1 then add them up and see if they all cancel out
            } else if (diff == 1 || diff == -1) {
                if (verbose)
                    printf("Fold/Class %d/%d is off by %d\n", i, j, diff);
                sum_diffs += diff;
            }
        }
        res += sum_diffs;
    }
    
    if (target_total != data.meta.num_examples) {
        res = -1 * res;
        if (verbose)
            printf("Class attendance check inconclusive as target total = %d instead of %d\n",
                   target_total, data.meta.num_examples);
    }
    
    free(target);
    free(fold_count);
    for (i = 0; i < num_folds; i++)
        free(class_attend[i]);
    free(class_attend);
    
    return res;
}



void cv_class_print(CV_Class class) {
    int i;
    printf("CV_Class:\n  num_classes = %d\n  thresholds =", class.num_classes);
    for (i = 0; i < class.num_classes - 1; i++)
        printf(" %f", class.thresholds[i]);
    printf("\n  class_frequencies =");
    for (i = 0; i < class.num_classes; i++)
        printf(" %d", class.class_frequencies[i]);
    printf("\n");
}

void cv_dataset_print(CV_Dataset dataset, char *title) {
    int i, j;
    printf("CV_Dataset: %s\n", title);
    printf("  num_classes = %d\n", dataset.meta.num_classes);
    printf("  num_attributes = %d\n", dataset.meta.num_attributes);
    printf("  num_fclib_seq = %d\n", dataset.meta.num_fclib_seq);
    printf("  num_examples = %d\n", dataset.meta.num_examples);
    printf("  attribute_names(types)(values) =\n");
    for (i = 0; i < dataset.meta.num_attributes; i++) {
        printf("     '%s'(%s)(", dataset.meta.attribute_names[i], dataset.meta.attribute_types[i]==DISCRETE?"D":"C");
        for (j = 0; j < dataset.meta.num_discrete_values[i]; j++)
            printf("'%s',", dataset.meta.discrete_attribute_map[i][j]);
        printf("\b)\n");
    }
    printf("  class_names =");
    for (i = 0; i < dataset.meta.num_classes; i++)
        printf(" '%s'", dataset.meta.class_names[i]);
    printf("\n\n");
}


void cv_example_print(CV_Example example, int num_atts) {
    int i;
    printf("CV_Example:\n");
    printf("  global_id_num = %d\n", example.global_id_num);
    printf("  random_gid = %d\n", example.random_gid);
    printf("  fclib_seq_num = %d\n", example.fclib_seq_num);
    printf("  fclib_id_num  = %d\n", example.fclib_id_num);
    printf("  containing_class_num = %d\n", example.containing_class_num);
    printf("  containing_fold_num  = %d\n", example.containing_fold_num);
    printf("  distinct_attribute_values:\n");
    for (i = 0; i < num_atts; i++)
        printf("    att %d => %d\n", i, example.distinct_attribute_values[i]);
    printf("\n");
}

// REVIEW-2012-03-26-ArtM: This looks like dead code.
/*
void cv_subset_print(CV_Subset subset, char *title) {
    int i;
    printf("CV_Dataset: %s\n", title);
    printf("  num_fclib_seq = %d\n", subset.meta.num_fclib_seq);
    printf("  num_examples = %d\n", subset.meta.num_examples);
    printf("  malloc_examples = %d\n", subset.malloc_examples);
    printf("  num_classes = %d\n", subset.meta.num_classes);
    printf("  class_names(number per) =");
    for (i = 0; i < subset.meta.num_classes; i++)
        printf(" '%s'(%d)", subset.meta.class_names[i], subset.meta.num_examples_per_class[i]);
    printf("\n");
    printf("  num_attributes = %d\n", subset.meta.num_attributes);
    printf("  attribute_names(types) =");
    for (i = 0; i < subset.meta.num_attributes; i++)
        printf(" '%s'(%s)", subset.meta.attribute_names[i], subset.meta.attribute_types[i]==DISCRETE?"DISCRETE":"CONTINUOUS");
    printf("\n");
    printf("  attribute high/low =");
    for (i = 0; i < subset.meta.num_attributes; i++)
        printf(" %d/%d", subset.high[i], subset.low[i]);
    printf("\n\n");
}
*/

// REVIEW-2012-03-26-ArtM: this looks like dead code.
/*
   Converts a data sample to a string
   NOTE: The string representation is NOT identical to what was in the original datafile.
         This function is probably a useful debugging tool rather than something to use in production code
 */
/*
void example_to_string(int num, CV_Subset data, int truth_col, char **string) {
    int i;
    unsigned int size = data.meta.num_attributes+2; // initial size accounts for all commas + 1 which gets removed
                                                    // plus the trailing NUL
    // Add lengths for each attribute
    for (i = 0; i < data.meta.num_attributes; i++) {
        if (data.meta.attribute_types[i] == CONTINUOUS) {
            char temp[128];
            sprintf(temp, "%f", data.float_data[i][data.examples[num].distinct_attribute_values[i]]);
            size += strlen(temp);
        } else if (data.meta.attribute_types[i] == DISCRETE) {
            size += strlen(data.meta.discrete_attribute_map[i][data.examples[num].distinct_attribute_values[i]]);
        }
    }
    // Add length for class
    size += strlen(data.meta.class_names[data.examples[num].containing_class_num]);
    
    *string = (char *)malloc((size+1) * sizeof(char));
    (*string)[0] = '\0';
    
    for (i = 0; i < data.meta.num_attributes; i++) {
        if (i+1 == truth_col) {
            strncat(*string, data.meta.class_names[data.examples[num].containing_class_num], size);
            strncat(*string, ",", size);
        }
        if (data.meta.attribute_types[i] == CONTINUOUS) {
            sprintf(*string, "%s%f,", *string, data.float_data[i][data.examples[num].distinct_attribute_values[i]]);
        } else if (data.meta.attribute_types[i] == DISCRETE) {
            strncat(*string, data.meta.discrete_attribute_map[i][data.examples[num].distinct_attribute_values[i]], size);
            strncat(*string, ",", size);
        }
    }
    
    // Remove the last comma by moving the NUL one spot
    (*string)[size-2] = '\0';
}
*/

int get_class_index(double value, CV_Class *class) {
    int i = 0;
    int idx = -1;
    while (idx < 0 && i < class->num_classes - 1)
        if (av_lteqf(value, class->thresholds[i++]))
            idx = i - 1;
    if (idx < 0)
        idx = class->num_classes - 1;
    return idx;
}


// REVIEW-2012-03-26-ArtM: This looks like dead code.
/*
void write_as_data_file(char *filename, CV_Subset data) {
    int i, j;
    FILE *fh;
    if ((fh = fopen(filename, "w")) == NULL)
        return;
    for (i = 0; i < data.meta.num_examples; i++) {
        for (j = 0; j < data.meta.num_attributes; j++) {
            int dav = data.examples[i].distinct_attribute_values[j];
            if (data.meta.attribute_types[j] == DISCRETE)
                fprintf(fh, "%s,", data.meta.discrete_attribute_map[j][dav]);
            else if (data.meta.attribute_types[j] == CONTINUOUS)
                fprintf(fh, "%g,", data.float_data[j][dav]);
        }
        fprintf(fh, "%s\n", data.meta.class_names[data.examples[i].containing_class_num]);
    }
}
*/


void __print_datafile(CV_Subset data, char *prefix) {
    int j, k;
    
    //printf("Data to print has %d samples\n", data.meta.num_examples);
    printf("\n");
    for (j = 0; j < data.meta.num_examples; j++) {
        //printf("Printing data for sample %d\n", j);
        printf("%s:SAMPLE:%d:", prefix, data.examples[j].fclib_id_num+1);
        for (k = 0; k < data.meta.num_attributes; k++) {
            if (data.meta.attribute_types[k] == CONTINUOUS) {
                printf("%f,", data.float_data[k][data.examples[j].distinct_attribute_values[k]]);
            } else if (data.meta.attribute_types[k] == DISCRETE) {
                printf("%s,", data.meta.discrete_attribute_map[k][data.examples[j].distinct_attribute_values[k]]);
            }
        }
        printf("%s\n", data.meta.class_names[data.examples[j].containing_class_num]);
    }
}


// Sort on fclib_seq_num with secondary sort on fclib_id_num

int qsort_example_compare_by_seq_id(const void *n, const void *m) {
    const CV_Example **x = (const CV_Example **)n;
    const CV_Example **y = (const CV_Example **)m;
    return cv_example_compare_by_seq_id(*x, *y);
}

int cv_example_compare_by_seq_id(const void *n, const void *m) {
    int as = ((const CV_Example*)n)->fclib_seq_num;
    int ai = ((const CV_Example*)n)->fclib_id_num;
    int bs = ((const CV_Example*)m)->fclib_seq_num;
    int bi = ((const CV_Example*)m)->fclib_id_num;
    if (as > bs)
        return 1;
    else if (as < bs)
        return -1;
    else
        if (ai > bi)
            return 1;
        else if (ai < bi)
            return -1;
        else
            return 0;
}

// REVIEW-2012-03-26-ArtM: This looks like dead code.
/*
int qsort_example_compare_by_gid(const void *n, const void *m) {
    const CV_Example **x = (const CV_Example **)n;
    const CV_Example **y = (const CV_Example **)m;
    return cv_example_compare_by_gid(*x, *y);
}

int cv_example_compare_by_gid(const void *n, const void *m) {
    int a = ((const CV_Example*)n)->global_id_num;
    int b = ((const CV_Example*)m)->global_id_num;
    if (a > b)
        return 1;
    else if (a < b)
        return -1;
    else
        return 0;
}
*/

// REVIEW-2012-03-26-ArtM: This looks like dead code.
/*
int qsort_example_compare_by_rgid(const void *n, const void *m) {
    const CV_Example **x = (const CV_Example **)n;
    const CV_Example **y = (const CV_Example **)m;
    return cv_example_compare_by_rgid(*x, *y);
}

int cv_example_compare_by_rgid(const void *n, const void *m) {
    int a = ((const CV_Example*)n)->random_gid;
    int b = ((const CV_Example*)m)->random_gid;
    if (a > b)
        return 1;
    else if (a < b)
        return -1;
    else
        return 0;
}
*/

/*
 * Sorts on class and then random_gid
 */
int qsort_example_compare_by_class(const void *n, const void *m) {
    const CV_Example **x = (const CV_Example **)n;
    const CV_Example **y = (const CV_Example **)m;
    return cv_example_compare_by_class(*x, *y);
}

int cv_example_compare_by_class(const void *n, const void *m) {
    int ac = ((const CV_Example*)n)->containing_class_num;
    int ag = ((const CV_Example*)n)->random_gid;
    int bc = ((const CV_Example*)m)->containing_class_num;
    int bg = ((const CV_Example*)m)->random_gid;
    if (ac > bc)
        return 1;
    else if (ac < bc)
        return -1;
    else
        if (ag > bg)
            return 1;
        else if (ag < bg)
            return -1;
        else
            return 0;
}

/*
 *  sorts on fold and then random_gid
 */
int qsort_example_compare_by_fold(const void *n, const void *m) {
    const CV_Example **x = (const CV_Example **)n;
    const CV_Example **y = (const CV_Example **)m;
    return cv_example_compare_by_fold(*x, *y);
}

int cv_example_compare_by_fold(const void *n, const void *m) {
    int af = ((const CV_Example*)n)->containing_fold_num;
    int ag = ((const CV_Example*)n)->random_gid;
    int bf = ((const CV_Example*)m)->containing_fold_num;
    int bg = ((const CV_Example*)m)->random_gid;
    if (af > bf)
        return 1;
    else if (af < bf)
        return -1;
    else
        if (ag > bg)
            return 1;
        else if (ag < bg)
            return -1;
        else
            return 0;
}


/*
 *  fclib2global
 *
 *  Converts a (fclib sequence, on-sequence id) pair to the global_id
 */
 
int fclib2global(int fclib_seq, int fclib_id, int num_seq, int *global_offset) {
    if (fclib_seq > num_seq - 1 || fclib_seq < 0 || fclib_id < 0)
        return -1;
    if (global_offset[fclib_seq] + fclib_id >= global_offset[fclib_seq + 1])
        return -1;
    int global_id = global_offset[fclib_seq] + fclib_id;
    if (global_id >= global_offset[num_seq])
        return -1;
    return global_id;
}

/*
 *  global2fclib
 *
 *  Converts a global_id to the fclib sequence and on-sequence id
 */
int global2fclib(int global_id, int num_seq, int *global_offset, int *fclib_seq, int *fclib_id) {
    if (global_id >= global_offset[num_seq]) {
        *fclib_seq = -1;
        *fclib_id = -1;
        return -1;
    }
    
    *fclib_seq = num_seq - 1;
    while (*fclib_seq >= 0 && global_id < global_offset[*fclib_seq]) {
        (*fclib_seq)--; 
    }
    if (*fclib_seq >= num_seq || *fclib_seq < 0) {
        *fclib_seq = -1;
        *fclib_id = -1;
    } else {
        *fclib_id = global_id - global_offset[*fclib_seq];
        if (*fclib_id < 0 || *fclib_id > global_offset[*fclib_seq + 1] - global_offset[*fclib_seq]) {
            *fclib_seq = -1;
            *fclib_id = -1;
        }

    }
    return *fclib_seq;
}
