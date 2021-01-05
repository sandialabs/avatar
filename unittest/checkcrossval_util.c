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
#include "../src/av_utils.h"
#include "../src/array.h"
#include "../src/crossval.h"


START_TEST(fclib_global_conv)
{
    int i;
    int new_fclib_seq, new_fclib_id;

    int num_seq = 3;
    int global_offset[] = { 0, 10, 15, 25 };

    int num_valid_tests = 6;
    int valid_fclib_seq[] = { 0, 0, 1, 1, 2, 2 };
    int valid_fclib_id[]  = { 0, 3, 0, 1, 0, 9 };
    for (i = 0; i < num_valid_tests; i++) {
        int global_id = fclib2global(valid_fclib_seq[i], valid_fclib_id[i], num_seq, global_offset);
        global2fclib(global_id, num_seq, global_offset, &new_fclib_seq, &new_fclib_id);
        fail_unless(new_fclib_seq == valid_fclib_seq[i] && new_fclib_id == valid_fclib_id[i], "fclib<->global failed");
    }

    int num_invalid_tests = 5;
    int invalid_fclib_seq[] = { -1, -1, 1, 1, 3 };
    int invalid_fclib_id[] = { -1, 2, -1, 6, 0 };
    for (i = 0; i < num_invalid_tests; i++) {
        int global_id = fclib2global(invalid_fclib_seq[i], invalid_fclib_id[i], num_seq, global_offset);
        fail_unless(global_id == -1, "global_id should be -1");
        global2fclib(global_id, num_seq, global_offset, &new_fclib_seq, &new_fclib_id);
        fail_unless(new_fclib_seq == -1 && new_fclib_id == -1, "fclib_seq/id should be -1");
    }
    
    num_invalid_tests = 4;
    int invalid_global_id[] = { -2, -1, 25, 100 };
    for (i = 0; i < num_invalid_tests; i++) {
        global2fclib(invalid_global_id[i], num_seq, global_offset, &new_fclib_seq, &new_fclib_id);
        fail_unless(new_fclib_seq == -1 && new_fclib_id == -1, "fclib_seq/id should be -1");
    }

}
END_TEST

START_TEST(class_index)
{
    int i;
    double value;
    CV_Class Class = {0};
    Class.num_classes = 6;
    Class.thresholds = (float *)malloc((Class.num_classes - 1) * sizeof(float));
    Class.class_frequencies = (int *)calloc(Class.num_classes, sizeof(int));
    for (i = 0; i < Class.num_classes - 1; i++)
        Class.thresholds[i] = 0.2*i + 0.1;
    for (value = -0.1; value <= 1.1; value+=0.001)
        Class.class_frequencies[get_class_index(value, &Class)]++;
    
    int baseline[] = { 201, 200, 200, 200, 200, 200 };
    fail_unless(! memcmp(Class.class_frequencies, baseline, Class.num_classes*sizeof(int)),
                "class_frequencies are incorrect");
    
    free(Class.thresholds);
    free(Class.class_frequencies);
}
END_TEST

START_TEST(comp_functions)
{
    CV_Example *Examples, **Examples_ptr;
    int num_seq = 3;
    int num_classes = 3;
    int num_folds = 4;
    int global_offset[] = { 0, 10, 15, 25 };
    int i, j, count;
    AV_SortedBlobArray Sorted;
    
    // Generate blobs and sort by seq_id
    Examples = (CV_Example *)calloc(global_offset[num_seq], sizeof(CV_Example));
    Examples_ptr = (CV_Example **)calloc(global_offset[num_seq], sizeof(CV_Example *));
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (1)");
    for (i = 0; i < global_offset[num_seq]; i++) {
        Examples[i].global_id_num = i;
        Examples[i].random_gid = i;
        global2fclib(Examples[i].global_id_num, num_seq, global_offset, &(Examples[i].fclib_seq_num),
                                                                     &(Examples[i].fclib_id_num));
        Examples[i].containing_class_num = i % num_classes;
        Examples[i].containing_fold_num = i % num_folds;
        // This is the baseline
        Examples_ptr[i] = &Examples[i];
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Examples[i], cv_example_compare_by_seq_id) == 1,
                    "failed to add blob to SBA (1)");
    }

    fail_unless(Sorted.numBlob == global_offset[num_seq], "Don't have correct number of blobs (1)");
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to add blobs sorted by seq_id");
    qsort(Sorted.blobs, Sorted.numBlob, sizeof(CV_Example *), qsort_example_compare_by_seq_id);
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to qsort by seq_id");
    av_freeSortedBlobArray(&Sorted);
    
    // Sort by class
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (2)");
    count = 0;
    for (i = 0; i < global_offset[num_seq]; i++)
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Examples[i], cv_example_compare_by_class) == 1,
                    "failed to add blob to SBA (2)");
    // Generate baseline
    for (i = 0; i < num_classes; i++)
        for (j = 0; j < global_offset[num_seq]; j++)
            if (j % num_classes == i)
                Examples_ptr[count++] = &Examples[j];
    fail_unless(Sorted.numBlob == global_offset[num_seq], "Don't have correct number of blobs (1)");
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to add blobs sorted by class");
    qsort(Sorted.blobs, Sorted.numBlob, sizeof(CV_Example *), qsort_example_compare_by_class);
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to qsort by class");
    av_freeSortedBlobArray(&Sorted);
    
    // Sort by fold
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (2)");
    count = 0;
    for (i = 0; i < global_offset[num_seq]; i++)
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Examples[i], cv_example_compare_by_fold) == 1,
                    "failed to add blob to SBA (2)");
    // Generate baseline
    for (i = 0; i < num_folds; i++)
        for (j = 0; j < global_offset[num_seq]; j++)
            if (j % num_folds == i)
                Examples_ptr[count++] = &Examples[j];
    fail_unless(Sorted.numBlob == global_offset[num_seq], "Don't have correct number of blobs (1)");
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to add blobs sorted by fold");
    qsort(Sorted.blobs, Sorted.numBlob, sizeof(CV_Example *), qsort_example_compare_by_fold);
    fail_unless(! memcmp(Sorted.blobs, Examples_ptr, global_offset[num_seq] * sizeof(CV_Example *)),
                "failed to qsort by fold");
    av_freeSortedBlobArray(&Sorted);
    
    free(Examples);
    free(Examples_ptr);
}
END_TEST

START_TEST(assign_folds)
{
    CV_Class Class = {0};
    CV_Subset Data = {0};
    int num_seq = 3;
    Class.num_classes = 3;
    int num_folds = 3;
    int global_offset[] = { 0, 100, 150, 270 };
    int i;
    int *fold_pops;
    AV_SortedBlobArray Sorted;
    
    Class.class_frequencies = (int *)calloc(Class.num_classes, sizeof(int));
    Data.meta.num_examples = global_offset[num_seq];
    Data.examples = (CV_Example *)calloc(Data.meta.num_examples, sizeof(CV_Example));
    
    // Generate blobs and sort by seq_id
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (1)");
    for (i = 0; i < global_offset[num_seq]; i++) {
        Data.examples[i].global_id_num = i;
        Data.examples[i].random_gid = i;
        global2fclib(Data.examples[i].global_id_num, num_seq, global_offset, &(Data.examples[i].fclib_seq_num),
                                                                             &(Data.examples[i].fclib_id_num));
        Data.examples[i].containing_class_num = i % Class.num_classes;
        Class.class_frequencies[Data.examples[i].containing_class_num]++;
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Data.examples[i], cv_example_compare_by_seq_id) == 1,
                    "failed to add blob to SBA (1)");
    }
    assign_class_based_folds(num_folds, Sorted, &fold_pops);
    fail_unless(check_class_attendance(num_folds, Class, Data, 0) == 0, "class attendance check failed");
    fail_unless(check_fold_attendance(num_folds, Data, 0) == 0, "fold attendance check failed");
    
    free(fold_pops);
    av_freeSortedBlobArray(&Sorted);
    free(Class.class_frequencies);
    free(Data.examples);
    
    /*
        Rerun with uneven folds
    */
    
    Class.num_classes = 3;
    num_folds = 4;
    
    Class.class_frequencies = (int *)calloc(Class.num_classes, sizeof(int));
    Data.meta.num_examples = global_offset[num_seq];
    Data.examples = (CV_Example *)malloc(Data.meta.num_examples * sizeof(CV_Example));
    
    // Generate blobs and sort by seq_id
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (1)");
    for (i = 0; i < global_offset[num_seq]; i++) {
        Data.examples[i].global_id_num = i;
        Data.examples[i].random_gid = i;
        global2fclib(Data.examples[i].global_id_num, num_seq, global_offset, &(Data.examples[i].fclib_seq_num),
                                                                             &(Data.examples[i].fclib_id_num));
        Data.examples[i].containing_class_num = i % Class.num_classes;
        Class.class_frequencies[Data.examples[i].containing_class_num]++;
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Data.examples[i], cv_example_compare_by_seq_id) == 1,
                    "failed to add blob to SBA (1)");
    }
    assign_class_based_folds(num_folds, Sorted, &fold_pops);
    fail_unless(check_class_attendance(num_folds, Class, Data, 0) == 0, "class attendance check failed");
    fail_unless(check_fold_attendance(num_folds, Data, 0) == 0, "fold attendance check failed");
    
    free(fold_pops);
    av_freeSortedBlobArray(&Sorted);
    free(Class.class_frequencies);
    free(Data.examples);
}
END_TEST

START_TEST(assign_folds_extreme)
{
    CV_Class Class = {0};
    CV_Subset Data = {0};
    Class.num_classes = 3;
    int num_folds = 4;
    int i;
    int *fold_pops;
    AV_SortedBlobArray Sorted;
    
    Data.meta.num_fclib_seq = 3;
    Class.class_frequencies = (int *)calloc(Class.num_classes, sizeof(int));
    Data.meta.global_offset = (int *)malloc((Data.meta.num_fclib_seq+1) * sizeof(int));
    Data.meta.global_offset[0] = 0;
    Data.meta.global_offset[1] = 1000;
    Data.meta.global_offset[2] = 1500;
    Data.meta.global_offset[3] = 2700;
    Data.meta.num_examples = Data.meta.global_offset[Data.meta.num_fclib_seq];
    Data.examples = (CV_Example *)malloc((Data.meta.num_examples+3) * sizeof(CV_Example));
    
    // Generate blobs and sort by seq_id
    fail_unless(av_initSortedBlobArray(&Sorted) == AV_SUCCESS, "failed to init SortedBlobArray (1)");
    for (i = 0; i < Data.meta.global_offset[Data.meta.num_fclib_seq]; i++) {
        Data.examples[i].global_id_num = i;
        Data.examples[i].random_gid = i;
        global2fclib(Data.examples[i].global_id_num, Data.meta.num_fclib_seq, Data.meta.global_offset,
                     &(Data.examples[i].fclib_seq_num), &(Data.examples[i].fclib_id_num));
        Data.examples[i].containing_class_num = i % Class.num_classes;
        Class.class_frequencies[Data.examples[i].containing_class_num]++;
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Data.examples[i], cv_example_compare_by_seq_id) == 1,
                    "failed to add blob to SBA (1)");
    }

    assign_class_based_folds(num_folds, Sorted, &fold_pops);
    fail_unless(check_class_attendance(num_folds, Class, Data, 0) == 0, "class attendance check failed");
    fail_unless(check_fold_attendance(num_folds, Data, 0) == 0, "fold attendance check failed");

    // Assign a fourth class with only 3 examples
    Data.meta.num_fclib_seq = 4;
    Class.num_classes = 4;
    Data.meta.global_offset = (int *)realloc(Data.meta.global_offset, (Data.meta.num_fclib_seq+1) * sizeof(int));
    Data.meta.global_offset[Data.meta.num_fclib_seq] = 2703;
    Class.class_frequencies = (int *)realloc(Class.class_frequencies, Class.num_classes * sizeof(int));
    // Zero out the new class frequency
    Class.class_frequencies[3] = 0;
    Data.meta.num_examples = Data.meta.global_offset[Data.meta.num_fclib_seq];
    //Data.examples = (CV_Example *)realloc(Data.examples, Data.meta.num_examples * sizeof(CV_Example));
    for (i = Data.meta.global_offset[Data.meta.num_fclib_seq-1]; i < Data.meta.global_offset[Data.meta.num_fclib_seq]; i++) {
        Data.examples[i].global_id_num = i;
        Data.examples[i].random_gid = i;
        global2fclib(Data.examples[i].global_id_num, Data.meta.num_fclib_seq, Data.meta.global_offset,
                     &(Data.examples[i].fclib_seq_num), &(Data.examples[i].fclib_id_num));
        Data.examples[i].containing_class_num = 3;
        Class.class_frequencies[Data.examples[i].containing_class_num]++;
        fail_unless(av_addBlobToSortedBlobArray(&Sorted, &Data.examples[i], cv_example_compare_by_seq_id) == 1,
                    "failed to add blob to SBA (1)");
    }
    
    int true_class_freq[] = { 900, 900, 900, 3 };
    fail_unless(!memcmp(Class.class_frequencies, true_class_freq, Class.num_classes * sizeof(int)),
                "class frequencies are incorrect");
    assign_class_based_folds(num_folds, Sorted, &fold_pops);
    fail_unless(check_class_attendance(num_folds, Class, Data, 0) == -1, "class attendance check failed");
    fail_unless(check_fold_attendance(num_folds, Data, 0) == 0, "fold attendance check failed");
    
    free(fold_pops);
    av_freeSortedBlobArray(&Sorted);
    free(Class.class_frequencies);
    free(Data.examples);
    free(Data.meta.global_offset);
}
END_TEST

START_TEST(test_parse_range)
{
    char *int_string = " 14, 4 , 15,  6-9,20 ,1,5, 18 - 19,10 ";
    int int_unsorted_truth[] = {14,4,15,6,7,8,9,20,1,5,18,19,10};
    int int_sorted_truth[] = {1,4,5,6,7,8,9,10,14,15,18,19,20};
    int int_num;
    int *int_range;

    char *float_string = " 4.444, 0.1 , 5.5,  6.06,15,  18,9,10.0 ,14,7.17, 8,   19,20.123 ";
    float float_unsorted_truth[] = {4.444,0.1,5.5,6.06,15,18,9,10,14,7.17,8,19,20.123};
    float float_sorted_truth[] = {0.1,4.444,5.5,6.06,7.17,8,9,10,14,15,18,19,20.123};
    int float_num;
    float *float_range;
    
    parse_int_range(int_string, 0, &int_num, &int_range);
    fail_unless(int_num == 13, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_unsorted_truth, int_num*sizeof(int)), "failed to parse unsorted int range");
    free(int_range);
    parse_int_range(int_string, 1, &int_num, &int_range);
    fail_unless(int_num == 13, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_sorted_truth, int_num*sizeof(int)), "failed to parse sorted int range");
    free(int_range);

    parse_float_range(float_string, 0, &float_num, &float_range);
    fail_unless(float_num == 13, "failed to parse float range");
    fail_unless(! memcmp(float_range, float_unsorted_truth, float_num*sizeof(int)), "failed to parse unsorted float range");
    free(float_range);
    parse_float_range(float_string, 1, &float_num, &float_range);
    fail_unless(float_num == 13, "failed to parse float range");
    fail_unless(! memcmp(float_range, float_sorted_truth, float_num*sizeof(int)), "failed to parse sorted float range");
    free(float_range);
}
END_TEST

START_TEST(test_parse_range_small)
{
    char *int_string;
    int *int_unsorted_truth;
    int *int_sorted_truth;
    int int_num;
    int *int_range;
    
    int_unsorted_truth = (int *)malloc(3 * sizeof(int));
    int_sorted_truth = (int *)malloc(3 * sizeof(int));
    
    int_string = "1";
    int_sorted_truth[0] = int_unsorted_truth[0] = 1;
    parse_int_range(int_string, 0, &int_num, &int_range);
    fail_unless(int_num == 1, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_unsorted_truth, int_num*sizeof(int)), "failed to parse unsorted int range (1)");
    free(int_range);
    parse_int_range(int_string, 1, &int_num, &int_range);
    fail_unless(int_num == 1, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_sorted_truth, int_num*sizeof(int)), "failed to parse sorted int range (1)");
    free(int_range);
    
    int_string = "1,2";
    int_sorted_truth[0] = int_unsorted_truth[0] = 1;
    int_sorted_truth[1] = int_unsorted_truth[1] = 2;
    parse_int_range(int_string, 0, &int_num, &int_range);
    fail_unless(int_num == 2, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_unsorted_truth, int_num*sizeof(int)), "failed to parse unsorted int range (2)");
    free(int_range);
    parse_int_range(int_string, 1, &int_num, &int_range);
    fail_unless(int_num == 2, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_sorted_truth, int_num*sizeof(int)), "failed to parse sorted int range (2)");
    free(int_range);
    
    int_string = "1-3";
    int_sorted_truth[0] = int_unsorted_truth[0] = 1;
    int_sorted_truth[1] = int_unsorted_truth[1] = 2;
    int_sorted_truth[2] = int_unsorted_truth[2] = 3;
    parse_int_range(int_string, 0, &int_num, &int_range);
    fail_unless(int_num == 3, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_unsorted_truth, int_num*sizeof(int)), "failed to parse unsorted int range (3)");
    free(int_range);
    parse_int_range(int_string, 1, &int_num, &int_range);
    fail_unless(int_num == 3, "failed to parse int range");
    fail_unless(! memcmp(int_range, int_sorted_truth, int_num*sizeof(int)), "failed to parse sorted int range (3)");
    free(int_range);
    
    free(int_unsorted_truth);
    free(int_sorted_truth);

}
END_TEST


START_TEST(test_prefix_checking)
{
    char **names;
    int num;
    Args_Opts args = {0};
    Boolean res;
    
    // Set up true test
    args.truth_column = 3;
    args.num_skipped_features = 2;
    args.skipped_features = (int *)malloc(2 * sizeof(int));
    args.skipped_features[0] = 2;
    args.skipped_features[1] = 5;
    num = 3;
    names = (char **)malloc(num * sizeof(char *));
    names[0] = strdup("1 Length");
    names[1] = strdup("4 Width");
    names[2] = strdup("6 Height");
    res = att_label_has_leading_number(names, num, args);
    fail_unless(res == TRUE, "Names do have leading column numbers but they were not found");
    
    free(names[0]);
    free(names[1]);
    free(names[2]);
    
    names[0] = strdup("1 Length");
    names[1] = strdup("4 Width");
    names[2] = strdup("6Height");
    res = att_label_has_leading_number(names, num, args);
    fail_unless(res == FALSE, "Names do not have leading column numbers but they were found (1)");
    
    free(names[0]);
    free(names[1]);
    free(names[2]);
    
    names[0] = strdup("1 Length");
    names[1] = strdup("4 Width");
    names[2] = strdup("5 Height");
    res = att_label_has_leading_number(names, num, args);
    fail_unless(res == FALSE, "Names do not have leading column numbers but they were found (2)");

    free(names[0]);
    free(names[1]);
    free(names[2]);
    free(names);
}
END_TEST


Suite *crossval_util_suite(void)
{
    Suite *suite = suite_create("CV_Utilities");
    
    TCase *tc_conversions = tcase_create(" Conversions ");
    TCase *tc_class_manip = tcase_create(" Class Manipulations ");
    TCase *tc_blob_sorting = tcase_create(" Blob Sorting ");
    TCase *tc_assign_folds = tcase_create(" Fold Assignment ");
    TCase *tc_parse_range = tcase_create(" Parse Range ");
    TCase *tc_prefix = tcase_create(" Prefix Checking ");
    
    suite_add_tcase(suite, tc_conversions);
    tcase_add_test(tc_conversions, fclib_global_conv);

    suite_add_tcase(suite, tc_class_manip);
    tcase_add_test(tc_class_manip, class_index);
    
    suite_add_tcase(suite, tc_blob_sorting);    
    tcase_add_test(tc_blob_sorting, comp_functions);
    
    suite_add_tcase(suite, tc_assign_folds);
    tcase_add_test(tc_assign_folds, assign_folds);
    tcase_add_test(tc_assign_folds, assign_folds_extreme);
    
    suite_add_tcase(suite, tc_parse_range);
    tcase_add_test(tc_parse_range, test_parse_range);
    tcase_add_test(tc_parse_range, test_parse_range_small);
    
    suite_add_tcase(suite, tc_prefix);
    tcase_add_test(tc_prefix, test_prefix_checking);
    
    return suite;
}
