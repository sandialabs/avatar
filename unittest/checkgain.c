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
#include <math.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/gain.h"

void _gen_data(CV_Subset *data);
void _free_data(CV_Subset data);
void _compute_truth_min1(float *true_info_gain, float *true_split_info);
void _compute_truth_min5(float *true_info_gain, float *true_split_info);

START_TEST(check_dlog_2)
{
    fail_unless(! av_ltf(dlog_2(2.0), 1.0) && ! av_gtf(dlog_2(2.0), 1.0), "log_2(2.0) should be 1.0");
    fail_unless(av_eqf(dlog_2(2.0), 1.0), "log_2(2.0) should be 1.0");
}
END_TEST

START_TEST(check_dlog_2_int)
{
    int i;
    for (i = 0; i < 10000; i++)
        fail_unless(av_eqf(dlog_2((float)i), dlog_2_int(i)), "failed");
}
END_TEST

START_TEST(info)
{
    CV_Subset data = {0};
    int *class_count;
    int i;
    
    data.meta.num_classes = 4;
    data.meta.num_examples = 100;
    class_count = (int *)malloc(data.meta.num_classes * sizeof(int));
    class_count[0] = 25;
    class_count[1] = 25;
    class_count[2] = 25;
    class_count[3] = 25;
    data.examples = (CV_Example *)malloc(data.meta.num_examples * sizeof(CV_Example));
    for (i = 0; i < data.meta.num_examples; i++)
        data.examples[i].containing_class_num = i % data.meta.num_classes;
    //fail_unless(av_eqf(_compute_info(&data), 2.0), "info should be 2.0");
    fail_unless(av_eqf(compute_info_from_array(class_count, data.meta.num_classes), 2.0), "info should be 2.0");
    free(class_count);
    free(data.examples);
}
END_TEST

START_TEST(split_info)
{
    int *array;
    int num_splits = 5;
    array = (int *)malloc(num_splits * sizeof(int));
    array[0] = 200;
    array[1] = 100;
    array[2] = 50;
    array[3] = 25;
    array[4] = 25;
    fail_unless(av_eqf(compute_split_info(array, num_splits), 1.875), "split info should be 1.875");
    free(array);
}
END_TEST

START_TEST(gain)
{
    int i, j;
    int **array;
    int num_splits = 4;
    int num_classes = 4;
    array = (int **)malloc(num_splits * sizeof(int *));
    for (i = 0; i < num_splits; i++) {
        array[i] = (int *)malloc(num_classes * sizeof(int));
        for (j = 0; j < num_classes; j++) {
            array[i][j] = 100;
        }
    }
    fail_unless(av_eqf(compute_gain(array, num_splits, num_classes), 0.0), "gain should be 0.0");
    for (i = 0; i < num_splits; i++)
        free(array[i]);
    free(array);
}
END_TEST

void _gen_data(CV_Subset *data) {
    int i;
    
    data->high = (int *)malloc(sizeof(int));
    data->low = (int *)malloc(sizeof(int));
    data->high[0] = 9;
    data->low[0] = 0;
    data->meta.num_examples = 10;
    data->meta.num_classes = 2;
    data->examples = (CV_Example *)malloc(data->meta.num_examples * sizeof(CV_Example));
    data->meta.attribute_types = (Attribute_Type *)malloc(sizeof(Attribute_Type));
    data->meta.attribute_types[0] = CONTINUOUS;
    
    for (i = 0; i < data->meta.num_examples; i++) {
        data->examples[i].distinct_attribute_values = (int *)malloc(sizeof(int));
        // Each example has unique att value
        data->examples[i].distinct_attribute_values[0] = i;
        // Even split on classes
        data->examples[i].containing_class_num = i/5;
    }
    // Rearrange to classes to get:
    // att val: 0 1 2 3 4 5 6 7 8 9
    // class  : 0 0 0 0 1 0 1 1 1 1
    data->examples[4].containing_class_num = 1;
    data->examples[5].containing_class_num = 0;
}

void _free_data(CV_Subset data) {
    int i;
    for (i = 0; i < data.meta.num_examples; i++) {
        free(data.examples[i].distinct_attribute_values);
    }
    free(data.examples);
    free(data.meta.attribute_types);
    free(data.high);
    free(data.low);
}

void _compute_truth_min1(float *info_gain, float *split) {
    // Best split is between examples 3 and 4 yielding the following attribute values for each split:
    //   D1 = 0 1 2 3
    //   D2 = 4 5 6 7 8 9
    
    int size_d1 = 4;
    int size_d2 = 6;
    int size_d = 10;
    float info_d = 1.0;
    float info_d1 = 0.0;
    float info_d2 = -( (1.0/6.0)*dlog_2(1.0/6.0) + (5.0/6.0)*dlog_2(5.0/6.0) );
    *info_gain = info_d - (size_d1*info_d1 + size_d2*info_d2)/(float)size_d;
    *split = -( size_d1*dlog_2((double)size_d1/(double)size_d) +
                size_d2*dlog_2((double)size_d2/(double)size_d) ) / (float)size_d;
}

void _compute_truth_min5(float *info_gain, float *split) {
    
    // Since get_min_examples_per_split() will return 5,
    // Best split is between examples 4 and 5 yielding the following attribute values for each split:
    //   D1 = 0 1 2 3 4
    //   D2 = 5 6 7 8 9
    
    int size_d1 = 5;
    int size_d2 = 5;
    int size_d = 10;
    float info_d = 1.0;
    float info_d1 = -( (4.0/5.0)*dlog_2(4.0/5.0) + (1.0/5.0)*dlog_2(1.0/5.0) );
    float info_d2 = -( (1.0/5.0)*dlog_2(1.0/5.0) + (4.0/5.0)*dlog_2(4.0/5.0) );
    *info_gain = info_d - (size_d1*info_d1 + size_d2*info_d2)/(float)size_d;
    *split = -( size_d1*dlog_2((double)size_d1/(double)size_d) +
                size_d2*dlog_2((double)size_d2/(double)size_d) ) / (float)size_d;
}

START_TEST(c45_split)
{
    CV_Subset data = {0};
    Args_Opts args = {0};
    float true_info_gain, true_split_info;
    int best_split_high, best_split_low;
    float check_info_gain;
    float c45_truth;
    
    _gen_data(&data);

    args.split_on_zero_gain = TRUE;
    args.dynamic_bounds = FALSE;
    args.debug = FALSE;
    
    args.minimum_examples = 1;
    _compute_truth_min1(&true_info_gain, &true_split_info);
    c45_truth = (true_info_gain - dlog_2(9.0)/data.meta.num_examples) / true_split_info;
    check_info_gain = best_c45_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, c45_truth), "best_c45_split incorrect for min1");
    fail_unless(best_split_high == 4 && best_split_low == 3, "best_c45_split in wrong position for min1");

    args.minimum_examples = 5;
    _compute_truth_min5(&true_info_gain, &true_split_info);
    c45_truth = (true_info_gain - dlog_2(1.0)/data.meta.num_examples) / true_split_info;
    check_info_gain = best_c45_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, c45_truth), "best_c45_split incorrect for min5");
    fail_unless(best_split_high == 5 && best_split_low == 4, "best_c45_split in wrong position for min5");

    _free_data(data);
}
END_TEST

START_TEST(gain_split)
{
    CV_Subset data = {0};
    Args_Opts args = {0};
    float true_info_gain, true_split_info;
    int best_split_high, best_split_low;
    float check_info_gain;
    float gain_truth;

    _gen_data(&data);
    
    args.dynamic_bounds = FALSE;
        
    args.minimum_examples = 1;
    _compute_truth_min1(&true_info_gain, &true_split_info);
    gain_truth = true_info_gain;
    check_info_gain = best_gain_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, gain_truth), "best_gain_split incorrect for min1");
    fail_unless(best_split_high == 4 && best_split_low == 3, "best_gain_split in wrong position for min1");

    args.minimum_examples = 5;
    _compute_truth_min5(&true_info_gain, &true_split_info);
    gain_truth = true_info_gain;
    check_info_gain = best_gain_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, gain_truth), "best_gain_split incorrect for min5");
    fail_unless(best_split_high == 5 && best_split_low == 4, "best_gain_split in wrong position for min5");

    _free_data(data);
}
END_TEST

START_TEST(gain_ratio_split)
{
    CV_Subset data = {0};
    Args_Opts args = {0};
    float true_info_gain, true_split_info;
    int best_split_high, best_split_low;
    float check_info_gain;
    float gain_ratio_truth;

    _gen_data(&data);
    
    args.dynamic_bounds = FALSE;
    
    args.minimum_examples = 2;    
    _compute_truth_min1(&true_info_gain, &true_split_info);
    gain_ratio_truth = true_info_gain / true_split_info;
    check_info_gain = best_gain_ratio_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, gain_ratio_truth), "best_gain_ratio_split incorrect for min1");
    fail_unless(best_split_high == 4 && best_split_low == 3, "best_gain_ratio_split in wrong position for min1");
    
    args.minimum_examples = 5;    
    _compute_truth_min5(&true_info_gain, &true_split_info);
    gain_ratio_truth = true_info_gain / true_split_info;
    check_info_gain = best_gain_ratio_split(&data, 0, &best_split_high, &best_split_low, args);
    fail_unless(av_eqf(check_info_gain, gain_ratio_truth), "best_gain_ratio_split incorrect for min5");
    fail_unless(best_split_high == 5 && best_split_low == 4, "best_gain_ratio_split in wrong position for min5");
    
    _free_data(data);
}
END_TEST

Suite *gain_suite(void)
{
    Suite *suite = suite_create("Gain");
    
    TCase *tc_gain_utils = tcase_create(" Check GainUtils ");
    suite_add_tcase(suite, tc_gain_utils);
    tcase_add_test(tc_gain_utils, check_dlog_2);
    tcase_add_test(tc_gain_utils, check_dlog_2_int);
    tcase_add_test(tc_gain_utils, info);
    tcase_add_test(tc_gain_utils, split_info);
    tcase_add_test(tc_gain_utils, gain);
    
    TCase *tc_best_splits = tcase_create(" Best Splits ");
    suite_add_tcase(suite, tc_best_splits);
    tcase_add_test(tc_best_splits, c45_split);
    tcase_add_test(tc_best_splits, gain_split);
    tcase_add_test(tc_best_splits, gain_ratio_split);
    
    dlog_2_int(-1);
    
    return suite;
}
