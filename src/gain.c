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
#include "crossval.h"
#include "gain.h"

double dlog_2_int(int x) {
    return dlog_2((double) x);
}

double dlog_2(double x) {
    //extern int num_log_comps;
    //num_log_comps++;
    //double b10 = log(x);
    double return_val;
    //if (isinf(b10))
    if (x < 0.000001)
        return_val = 0.0;
    else
        return_val = log(x)/log(2.0);
        //return_val = b10/log(2.0);
        //return b10/LOG2;
    //printf("Log(%.0f) returning %.14g\n", x, return_val);
    return return_val;
}

/*
 * Unused at the current time
 *
float _log_2(double x) {
    float b10 = log(x);
    float return_val;
    //if (isinf(b10))
    if (x < 0.000001)
        return_val = 0.0;
    else
        return_val = b10/log(2.0);
//        return_val = b10/LOG2;
    //printf("Log(%.0f) returning %.14g\n", x, return_val);
    return return_val;
}
 *
 */

float compute_info_from_array(int *array, int num) {

    int i;
    int total = 0;
    double baseline_info = 0.0;
    //float baseline_info = 0.0;

    for (i = 0; i < num; i++) {
        total += array[i];
        //baseline_info -= array[i] * _log_2((double)array[i]);
        //baseline_info -= array[i] * dlog_2(array[i]);
        baseline_info -= array[i] * dlog_2_int(array[i]);
    }

    //return baseline_info/(float)total + _log_2((double)total);
    //return baseline_info/(double)total + dlog_2((double)total);

    // Avoid a divide-by-zero problem

    if (total == 0)
        return 0.0;

    return baseline_info/(double)total + dlog_2_int(total);
}

//Added by DACIESL June-02-08: HDDT CAPABILITY
//Function returns Hellinger distance of a split
float compute_hellinger_from_array(int **array, int num, int a, int b) {

    int i;
    int total[2];
    double hellinger = 0.0;

    total[0]=0,total[1]=0;

    for (i = 0; i < num; i++) {
        total[0] += array[i][a];
        total[1] += array[i][b];
    }

    for (i = 0; i < num; i++) {
        if(total[0] != 0 && total[1] != 0) {
	    hellinger += pow(sqrt((1.0*array[i][a])/(1.0*total[0])) - sqrt((1.0*array[i][b])/(1.0*total[1])), 2.0);
        }
    }

    return sqrt(hellinger);
}


/*
 * Unused at the current time
 *
float _compute_info(CV_Subset *data) {
    int i;
    float p_j;
    float baseline_info = 0.0;

    // Consider using class_frequencies but these need to be updated for new subsets
    int *class_count;
    class_count = (int *)calloc(data->meta.num_classes, sizeof(int));
    for (i = 0; i < data->meta.num_examples; i++)
        class_count[data->examples[i].containing_class_num]++;
    for (i = 0; i < data->meta.num_classes; i++) {
        p_j = (float)class_count[i]/(float)data->meta.num_examples;
        baseline_info -= p_j * dlog_2(p_j);
    }

    free(class_count);

    return baseline_info;
}
 *
 */

float compute_split_info(int *array, int num) {
    int i;
    int total = 0;
    double temp = 0.0;
    for (i = 0; i < num; i++) {
        temp -= array[i] * dlog_2(array[i]);
        total += array[i];
    }
    return temp/(float)total + dlog_2(total);
}

//Added by DACIESL June-02-08: HDDT CAPABILITY
//Function returns average Hellinger distance between each pair of classes
float compute_hellinger(int **array, int splits, int classes) {
    int i, j, k;
    float result = 0.0, compares = 0.0, a = 0.0, b = 0.0;

    for (j = 0; j < classes - 1; j++) {
  	for (k = j + 1; k < classes; k++) {
            a = 0.0;
            b = 0.0;
            for (i = 0; i < splits; i++) {
  	        a += array[i][j];
  	        b += array[i][k];
            }
            if (a != 0.0 && b != 0.0) {
                compares += 1.0;
        	result += compute_hellinger_from_array(array, splits, j, k);
            }
        }
    }

    if (compares > 0.0) {
        //printf("compute_hellinger: return %.14g\n", result/compares);
        return (result/compares);
    }
    else {
        return 0;
    }
}

float compute_gain(int **array, int splits, int classes) {
    int i, j;
    int per_split_total;
    int total = 0;
    int *across_splits;
    float gain = 0.0;

    across_splits = (int *)calloc(classes, sizeof(int));
    for (i = 0; i < splits; i++) {
        per_split_total = 0;
        for (j = 0; j < classes; j++) {
            per_split_total += array[i][j];
            across_splits[j] += array[i][j];
        }
        total += per_split_total;
        gain += (float)per_split_total * compute_info_from_array(array[i], classes);
    }

    //printf("compute_gain: return %.14g\n", gain/(float)total);
    //printf("compute_gain: baseline_info = %.14g\n", compute_info_from_array(across_splits, classes));
    float full_gainer = compute_info_from_array(across_splits, classes) - (gain / (float)total);
    free(across_splits);
    return full_gainer;
}

int get_min_examples_per_split(CV_Subset *data, Args_Opts args) {
    int min;
    if (args.dynamic_bounds) {
        min = (int)(0.1 * (float)data->meta.num_examples / (float)data->meta.num_classes);
        if (min <= args.minimum_examples)
            min = args.minimum_examples;
        else if (min > 25)
            min = 25;
    } else {
        min = args.minimum_examples;
    }
    return min;
}

float best_gain_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args) {
    int i, j, k;
    int min_split;
    float return_value = INTMIN;

    if (data->meta.attribute_types[att_num] == CONTINUOUS) {
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        if (num_distinct_values > 1) {

            float information_gain;

            // Initialize
            int split_info[] = { 0, data->meta.num_examples };
            int *total_per_distinct;
            total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
            int *gain_array[2];
            for (i = 0; i < 2; i++)
                gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
            int *avc[data->meta.num_classes];
            for (i = 0; i < data->meta.num_classes; i++)
                avc[i] = (int *)calloc(num_distinct_values, sizeof(int));

            // Populate arrays
            for (i = 0; i < data->meta.num_examples; i++) {
                CV_Example e = data->examples[i];
                if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
                    e.distinct_attribute_values[att_num] >= data->low[att_num]) {
                        avc[e.containing_class_num][e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        total_per_distinct[e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        gain_array[1][e.containing_class_num]++;
                }
            }

            i = 0;
            while (total_per_distinct[i] == 0 && i < num_distinct_values - 2) i++;
            while (i < num_distinct_values - 1) {
                j = i + 1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                if (total_per_distinct[j] != 0) {
                    for (k = 0; k < data->meta.num_classes; k++) {
                        split_info[0] += avc[k][i];
                        split_info[1] -= avc[k][i];
                        gain_array[0][k] += avc[k][i];
                        gain_array[1][k] -= avc[k][i];
                    }

                    min_split = get_min_examples_per_split(data, args);

                    if (split_info[0] >= min_split && split_info[1] >= min_split) {
                        information_gain = compute_gain(gain_array, 2, data->meta.num_classes);
                        if (information_gain > return_value) {
                            return_value = information_gain;
                            *returned_low = i + data->low[att_num];
                            *returned_high = j + data->low[att_num];
                        }
                    }
                }
                i = j;
            }

            // Clean up
            free(total_per_distinct);
            for (i = 0; i < 2; i++)
                free(gain_array[i]);
            for (i = 0; i < data->meta.num_classes; i++)
                free(avc[i]);

        }
    } else if (data->meta.attribute_types[att_num] == DISCRETE) {
        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        int **gain_array, *split_info;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_info = (int *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int));

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++)
            gain_array[data->examples[i].distinct_attribute_values[att_num]][data->examples[i].containing_class_num]++;

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_info[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_info[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        min_split = get_min_examples_per_split(data, args);
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_info[i] > max_val)
                max_val = split_info[i];
        if (max_val <= data->meta.num_examples - min_split)
            return_value = compute_gain(gain_array, data->meta.num_discrete_values[att_num], data->meta.num_classes);

        // Clean up
        free(split_info);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);
    }

    if (return_value > INTMIN)
        return return_value;
     else
        return NO_SPLIT;
}

float best_gain_ratio_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args) {
    int i, j, k;
    int min_split;
    float return_value = INTMIN;
    float information_gain;

    if (data->meta.attribute_types[att_num] == CONTINUOUS) {
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        if (num_distinct_values > 1) {

            // Initialize
            int split_info[] = { 0, data->meta.num_examples };
            int *total_per_distinct;
            total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
            int *gain_array[2];
            for (i = 0; i < 2; i++)
                gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
            int *avc[data->meta.num_classes];
            for (i = 0; i < data->meta.num_classes; i++)
                avc[i] = (int *)calloc(num_distinct_values, sizeof(int));

            // Populate arrays
            for (i = 0; i < data->meta.num_examples; i++) {
                CV_Example e = data->examples[i];
                if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
                    e.distinct_attribute_values[att_num] >= data->low[att_num]) {
                        avc[e.containing_class_num][e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        total_per_distinct[e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        gain_array[1][e.containing_class_num]++;
                }
            }

            i = 0;
            while (total_per_distinct[i] == 0 && i < num_distinct_values - 2) i++;
            while (i < num_distinct_values - 1) {
                j = i + 1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                if (total_per_distinct[j] != 0) {
                    for (k = 0; k < data->meta.num_classes; k++) {
                        split_info[0] += avc[k][i];
                        split_info[1] -= avc[k][i];
                        gain_array[0][k] += avc[k][i];
                        gain_array[1][k] -= avc[k][i];
                    }

                    min_split = get_min_examples_per_split(data, args);

                    if (split_info[0] >= min_split && split_info[1] >= min_split) {
                        information_gain = compute_gain(gain_array, 2, data->meta.num_classes) /
                                           compute_split_info(split_info, 2);
                        if (information_gain > return_value) {
                            return_value = information_gain;
                            *returned_low = i + data->low[att_num];
                            *returned_high = j + data->low[att_num];
                        }
                    }
                }
                i = j;
            }

            // Clean up
            free(total_per_distinct);
            for (i = 0; i < 2; i++)
                free(gain_array[i]);
            for (i = 0; i < data->meta.num_classes; i++)
                free(avc[i]);

        }
    } else if (data->meta.attribute_types[att_num] == DISCRETE) {
        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        int **gain_array, *split_info;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_info = (int *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int));

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++)
            gain_array[data->examples[i].distinct_attribute_values[att_num]][data->examples[i].containing_class_num]++;
        information_gain = compute_gain(gain_array, data->meta.num_discrete_values[att_num], data->meta.num_classes);

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_info[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_info[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        min_split = get_min_examples_per_split(data, args);
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_info[i] > max_val)
                max_val = split_info[i];
        if (max_val <= data->meta.num_examples - min_split) {
            return_value = information_gain / compute_split_info(split_info, data->meta.num_discrete_values[att_num]);
        }

        // Clean up
        free(split_info);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);
    }

    if (return_value > INTMIN)
        return return_value;
     else
        return NO_SPLIT;
}

float best_c45_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args) {
    int i, j, k;
    int min_split;
    int best_split_low = 0;
    int best_split_high = 0;
    float best_split_info = 0.0;
    float best_information_gain = INTMIN;
    float return_value = INTMIN;

    if (data->meta.attribute_types[att_num] == CONTINUOUS) {
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        if (0 || args.debug)
            printf("There are %d distinct values\n", num_distinct_values);
        if (num_distinct_values > 1) {

            int num_potential_splits = 0;
            float information_gain;

            // Initialize
            int split_info[] = { 0, data->meta.num_examples };
            int *total_per_distinct;
            total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
            int *gain_array[2];
            for (i = 0; i < 2; i++)
                gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
            int *avc[data->meta.num_classes];
            for (i = 0; i < data->meta.num_classes; i++)
                avc[i] = (int *)calloc(num_distinct_values, sizeof(int));

            // Populate arrays
            for (i = 0; i < data->meta.num_examples; i++) {
                CV_Example e = data->examples[i];
                if (0 || args.debug)
                    printf("att %d: ex %d: %d <= %d <= %d\n",
                           att_num, i, data->low[att_num], e.distinct_attribute_values[att_num], data->high[att_num]);
                if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
                    e.distinct_attribute_values[att_num] >= data->low[att_num]) {
                        if (0 || args.debug)
                            printf("Att:%d Val:%d Class:%d\n", att_num,
                                   e.distinct_attribute_values[att_num], e.containing_class_num);
                        avc[e.containing_class_num][e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        total_per_distinct[e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        gain_array[1][e.containing_class_num]++;
                }
            }

            i = 0;
            while (total_per_distinct[i] == 0 && i < num_distinct_values - 2) i++;
            while (i < num_distinct_values - 1) {
                j = i + 1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                if (total_per_distinct[j] != 0) {
                    for (k = 0; k < data->meta.num_classes; k++) {
                        split_info[0] += avc[k][i];
                        split_info[1] -= avc[k][i];
                        gain_array[0][k] += avc[k][i];
                        gain_array[1][k] -= avc[k][i];
                    }
                    //printf("%d: si = %d,%d ga0 = %d,%d ga1 = %d,%d\n", i, split_info[0], split_info[1],
                    //        gain_array[0][0], gain_array[0][1], gain_array[1][0], gain_array[1][1]);

                    min_split = get_min_examples_per_split(data, args);

                    if (split_info[0] >= min_split && split_info[1] >= min_split) {
                        if (0 && args.debug)
                            printf("GA: %d %d %d %d\n", gain_array[0][0], gain_array[0][1],
                                                        gain_array[1][0], gain_array[1][1]);
                        information_gain = compute_gain(gain_array, 2, data->meta.num_classes);
                        if (information_gain > best_information_gain) {
                            best_information_gain = information_gain;
                            best_split_info = compute_split_info(split_info, 2);
                            best_split_low = i + data->low[att_num];
                            best_split_high = j + data->low[att_num];
                            if (0 && args.debug)
                                printf(":::best info gain for att %d between %d/%d = %.14g\n", att_num,
                                       best_split_low, best_split_high, information_gain);
                        }
                        num_potential_splits++;
                    }
                }
                i = j;
            }

            if (num_potential_splits > 0) {
                best_information_gain -= dlog_2((double)num_potential_splits)/(float)data->meta.num_examples;
                if (best_information_gain > 0.0 || args.split_on_zero_gain) {
                    return_value = best_information_gain / best_split_info;
                    *returned_high = best_split_high;
                    *returned_low = best_split_low;
                }
            }

            // Clean up
            free(total_per_distinct);
            for (i = 0; i < 2; i++)
                free(gain_array[i]);
            for (i = 0; i < data->meta.num_classes; i++)
                free(avc[i]);
        }
    } else if (data->meta.attribute_types[att_num] == DISCRETE) {
        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        int **gain_array, *split_info;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_info = (int *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int));

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++)
            gain_array[data->examples[i].distinct_attribute_values[att_num]][data->examples[i].containing_class_num]++;
        best_information_gain = compute_gain(gain_array, data->meta.num_discrete_values[att_num], data->meta.num_classes);

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_info[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_info[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        min_split = get_min_examples_per_split(data, args);
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_info[i] > max_val)
                max_val = split_info[i];
        if (max_val <= data->meta.num_examples - min_split)
            return_value = best_information_gain / compute_split_info(split_info, data->meta.num_discrete_values[att_num]);

        // Clean up
        free(split_info);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);
    }

    if (return_value > INTMIN)
        return return_value;
    else
        return NO_SPLIT;
}

void get_total_per_distinct(CV_Subset *data, int att_num, int *total_per_distinct) {

    int i;

    // Populate array with number of samples per distinct attribute value
    // WARNING: it assumes total_per_distinct was initialized to all zeros before this call
    for (i = 0; i < data->meta.num_examples; i++) {
        CV_Example e = data->examples[i];
        if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
            e.distinct_attribute_values[att_num] >= data->low[att_num]) {
            int idExample = e.distinct_attribute_values[att_num] - data->low[att_num];
            total_per_distinct[idExample]++;
        }
    }
    return ;

}

void get_avc_gain_arrays(CV_Subset *data, int att_num, int **gain_array, int **avc) {

    int i;
    // Populate arrays
    for (i = 0; i < data->meta.num_examples; i++) {
        CV_Example e = data->examples[i];
        if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
            e.distinct_attribute_values[att_num] >= data->low[att_num]) {
            int idClass   = e.containing_class_num;
            int idExample = e.distinct_attribute_values[att_num] - data->low[att_num];
            avc[idClass][idExample]++;
            gain_array[1][idClass]++;
        }
    }

    return ;

}

//#define DEBUG_COSMIN
float best_ert_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low,
                     float *cut_threshold, Args_Opts args) {
    int i, j, k;
    int min_split;
    float return_value = INTMIN;
    float best_information_gain=INTMIN, best_split_info=INTMIN;

    min_split = get_min_examples_per_split(data, args);

    if (data->meta.attribute_types[att_num] == CONTINUOUS) {

#ifdef DEBUG_COSMIN
        printf("ERT: low/high:%d %d\n",att_num,data->high[att_num],data->low[att_num]);
#endif

        // exit if the no. of distinct values is less than the minimum no. of samples
        // allowed on each side of the split
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        // if (num_distinct_values < 2 || num_distinct_values < 2*min_split) return NO_SPLIT;
        if (num_distinct_values < 2) return NO_SPLIT;

        // compute total_per_distinct to use later and decide we have enough samples
        int *total_per_distinct;
        total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
        get_total_per_distinct(data, att_num, total_per_distinct);

        // initialize split information
        int split_info[] = { 0, data->meta.num_examples };
        int *gain_array[2], *avc[data->meta.num_classes];

        for (i = 0; i < 2; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_classes; i++)
            avc[i] = (int *)calloc(num_distinct_values, sizeof(int));
        get_avc_gain_arrays(data, att_num, gain_array, avc);

        // Flag potential candidates
        int *lowBounds, *highBounds, countBounds=0;
        lowBounds  = (int *)calloc(num_distinct_values, sizeof(int));
        highBounds = (int *)calloc(num_distinct_values, sizeof(int));
        i=0;
        while (i < num_distinct_values - 1) {
            if (total_per_distinct[i] > 0) {
                j=i+1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                int num_samples_below = 0;
                for (k = 0; k <= i; k++)
                    num_samples_below += total_per_distinct[k];
                int num_samples_above = 0;
                for (k = j; k < num_distinct_values; k++)
                    num_samples_above += total_per_distinct[k];
                if (num_samples_below >= min_split && num_samples_above >= min_split) {
                    lowBounds[countBounds]  = i;
                    highBounds[countBounds] = j;
                    countBounds++;
                }
            }
            i++;
        }

        //if (countBounds == 0) return  NO_SPLIT;

        if (countBounds>0) {
            // this is sampling the pair then the cut inside the pair
            // int   randPair = rand() % countBounds;             // get a random pair of consecutive bounds
            // float rndcut   = (float)rand() / (float)RAND_MAX ; // random cut inside that pair
            // *returned_low  = data->low[att_num] + lowBounds[randPair];
            // *returned_high = data->low[att_num] + highBounds[randPair];
            // float deltaHmL  = data->float_data[att_num][*returned_high] - data->float_data[att_num][*returned_low];
            // *cut_threshold  = data->float_data[att_num][*returned_low]  + rndcut * deltaHmL;

            float rndcut   = (float)rand() / (float)RAND_MAX ; // random cut inside that pair
            int idxmin = data->low[att_num] + lowBounds[0];
            int idxmax = data->low[att_num] + highBounds[countBounds-1];
            float deltaHmL = data->float_data[att_num][idxmax] - data->float_data[att_num][idxmin];
            *cut_threshold = data->float_data[att_num][idxmin]  + rndcut * deltaHmL;

            int randPair = 0;
            *returned_low  = data->low[att_num] + lowBounds[randPair];
            *returned_high = data->low[att_num] + highBounds[randPair];
            while (!(data->float_data[att_num][*returned_low]<=*cut_threshold && data->float_data[att_num][*returned_high]>=*cut_threshold)) {
                randPair++;
                *returned_low  = data->low[att_num] + lowBounds[randPair];
                *returned_high = data->low[att_num] + highBounds[randPair];
            }

            i = 0;
            while (total_per_distinct[i] == 0 && i < num_distinct_values - 2) i++;
            while (i <= lowBounds[randPair]) {
                j = i + 1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                // printf("%d/%d %d\n",i,j,num_distinct_values);
                if (total_per_distinct[j] != 0) {
                    for (k = 0; k < data->meta.num_classes; k++) {
                        split_info[0] += avc[k][i];
                        split_info[1] -= avc[k][i];
                        gain_array[0][k] += avc[k][i];
                        gain_array[1][k] -= avc[k][i];
                    }
                }
                i = j;
            }
            best_information_gain = compute_gain(gain_array, 2, data->meta.num_classes);
            best_split_info       = compute_split_info(split_info, 2);
            best_information_gain -= dlog_2((double)countBounds)/(float)data->meta.num_examples;
            return_value   = best_information_gain / best_split_info;
        }
#ifdef DEBUG_COSMIN
        printf("  ->ndvs:%d,i:%d,j:%d,rv:%f,rl:%d,rh:%d,ct:%f\n",
                num_distinct_values,i,j,return_value,*returned_low,*returned_high,*cut_threshold);
        //printf("  ->ms:%d,nsb:%d,nsa:%d\n",min_split,num_samples_below,num_samples_above);
        fflush(stdout);
#endif
        // Clean up
        free(lowBounds);
        free(highBounds);
        free(total_per_distinct);
        for (i = 0; i < 2; i++)
            free(gain_array[i]);
        for (i = 0; i < data->meta.num_classes; i++)
            free(avc[i]);


    } else if (data->meta.attribute_types[att_num] == DISCRETE) {

        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        // Check if there are enough samples to actually split
        int **gain_array, *split_info;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_info = (int  *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int) );

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++) {
            CV_Example e = data->examples[i];
            gain_array[e.distinct_attribute_values[att_num]][e.containing_class_num]++;
        }

        best_information_gain = compute_gain(gain_array, data->meta.num_discrete_values[att_num], data->meta.num_classes);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_info[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_info[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_info[i] > max_val)
                max_val = split_info[i];
        if (max_val <= data->meta.num_examples - min_split)
            return_value = best_information_gain / compute_split_info(split_info, data->meta.num_discrete_values[att_num]);

        // Clean up
        free(split_info);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);

    }

    if (return_value > INTMIN)
        return return_value;
    else
        return NO_SPLIT;
}
//#undef DEBUG_COSMIN

//#define DEBUG_COSMIN
float best_trt_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low,
                     float *cut_threshold, Args_Opts args) {
    int i, j, k;
    int min_split;
    float return_value = INTMIN;

    min_split = get_min_examples_per_split(data, args);

    // printf("Here!\n");
    if (data->meta.attribute_types[att_num] == CONTINUOUS) {

#ifdef DEBUG_COSMIN
        printf("TRT: low/high:%d %d\n",att_num,data->high[att_num],data->low[att_num]);
#endif

        // exit if the no. of distinct values is less than the minimum no. of samples
        // allowed on each side of the split
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        if (num_distinct_values < 2) return NO_SPLIT;

        // compute total_per_distinct to use later and decide we have enough samples
        int *total_per_distinct;
        total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
        // Populate array with number of samples per distinct attribute value
        for (i = 0; i < data->meta.num_examples; i++) {
            CV_Example e = data->examples[i];
            // printf("%d,%d,%d\n",data->low[att_num],data->high[att_num],e.distinct_attribute_values[att_num]);
            if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
                e.distinct_attribute_values[att_num] >= data->low[att_num]) {
                total_per_distinct[e.distinct_attribute_values[att_num] - data->low[att_num]]++;
            }
        }
        // printf("------------------------\n");
        // for (i = 0; i < num_distinct_values; i++) {
        //     printf("%d,%d\n",i,total_per_distinct[i]);
        // }

        // Flag potential candidates
        int *lowBounds, *highBounds, countBounds=0;
        lowBounds  = (int *)calloc(num_distinct_values, sizeof(int));
        highBounds = (int *)calloc(num_distinct_values, sizeof(int));
        i=0;
        // printf("Before: %d\n",countBounds);
        // fflush(stdout);
        while (i < num_distinct_values - 1) {
            if (total_per_distinct[i] > 0) {
                j=i+1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                int num_samples_below = 0;
                for (k = 0; k <= i; k++)
                    num_samples_below += total_per_distinct[k];
                int num_samples_above = 0;
                for (k = j; k < num_distinct_values; k++)
                    num_samples_above += total_per_distinct[k];
                if (num_samples_below >= min_split && num_samples_above >= min_split) {
                    lowBounds[countBounds]  = i;
                    highBounds[countBounds] = j;
                    countBounds++;
                }
            }
            i++;
        }
        // printf("------------------------\n");
        // printf("After: %d,%d\n",countBounds,min_split);
        // for (i = 0; i < countBounds; i++) {
        //     printf("%d,%d -> %d\n",i,lowBounds[i],highBounds[i]);
        // }
        // exit(0);

        // if (countBounds == 0) return  NO_SPLIT;

        if (countBounds>0) {

            // int   randPair = rand() % countBounds;             // get a random pair of consecutive bounds
            // float rndcut   = (float)rand() / (float)RAND_MAX ; // random cut inside that pair
            // return_value   = (float)rand() / (float)RAND_MAX ; // random gain value
            //
            //
            // // draw a random split between min and max
            // *returned_low  = data->low[att_num] + lowBounds[randPair];
            // *returned_high = data->low[att_num] + highBounds[randPair];
            // float deltaHmL  = data->float_data[att_num][*returned_high] - data->float_data[att_num][*returned_low];
            // *cut_threshold  = data->float_data[att_num][*returned_low]  + rndcut * deltaHmL;

            return_value   = (float)rand() / (float)RAND_MAX ; // random gain value
            float rndcut   = (float)rand() / (float)RAND_MAX ; // random cut inside that pair
            int idxmin = data->low[att_num] + lowBounds[0];
            int idxmax = data->low[att_num] + highBounds[countBounds-1];
            float deltaHmL = data->float_data[att_num][idxmax] - data->float_data[att_num][idxmin];
            *cut_threshold = data->float_data[att_num][idxmin]  + rndcut * deltaHmL;

            int randPair = 0;
            *returned_low  = data->low[att_num] + lowBounds[randPair];
            *returned_high = data->low[att_num] + highBounds[randPair];
            while (!(data->float_data[att_num][*returned_low]<=*cut_threshold && data->float_data[att_num][*returned_high]>=*cut_threshold)) {
              randPair++;
              *returned_low  = data->low[att_num] + lowBounds[randPair];
              *returned_high = data->low[att_num] + highBounds[randPair];
            }
        }

        // Clean up
        free(total_per_distinct);
        free(lowBounds);
        free(highBounds);

// #define DEBUG_COSMIN
#ifdef DEBUG_COSMIN
        printf("  ->rv:%f,rl:%d,rh:%d,ct:%f,%f\n",
              return_value,*returned_low,*returned_high,*cut_threshold,deltaHmL);
        printf("  ->ms:%d\n",min_split);
        fflush(stdout);
#endif
// #undef DEBUG_COSMIN

    } else if (data->meta.attribute_types[att_num] == DISCRETE) {

        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        // Check if there are enough samples to actually split
        int **gain_array, *split_info;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_info = (int  *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int) );

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++) {
            CV_Example e = data->examples[i];
            gain_array[e.distinct_attribute_values[att_num]][e.containing_class_num]++;
        }
        //Cosmin: I do not need the gain_array, can compute directly split_info
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_info[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_info[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_info[i] > max_val)
                max_val = split_info[i];
        if (max_val <= data->meta.num_examples - min_split)
            return_value =  (float)rand() / (float)RAND_MAX;

        // Clean up
        free(split_info);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);

    }

    if (return_value > INTMIN)
        return return_value;
    else
        return NO_SPLIT;

}

//Added by DACIESL June-02-08: HDDT CAPABILITY
//Function returns best split determined by Hellinger distance
float best_hellinger_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args) {
    int i, j, k;
    int min_split;
    int best_split_low = 0;
    int best_split_high = 0;
    float best_split_hellinger = 0.0;
    float best_hellinger = INTMIN;
    float return_value = INTMIN;

    if (data->meta.attribute_types[att_num] == CONTINUOUS) {
        int num_distinct_values = data->high[att_num] - data->low[att_num] + 1;
        if (0 || args.debug)
            printf("There are %d distinct values\n", num_distinct_values);
        if (num_distinct_values > 1) {

            int num_potential_splits = 0;
            float hellinger;

            // Initialize
            int split_hellinger[] = { 0, data->meta.num_examples };
            int *total_per_distinct;
            total_per_distinct = (int *)calloc(num_distinct_values, sizeof(int));
            int *gain_array[2];
            for (i = 0; i < 2; i++)
                gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
            int *avc[data->meta.num_classes];
            for (i = 0; i < data->meta.num_classes; i++)
                avc[i] = (int *)calloc(num_distinct_values, sizeof(int));

            // Populate arrays
            for (i = 0; i < data->meta.num_examples; i++) {
                CV_Example e = data->examples[i];
                if (0 || args.debug)
                    printf("att %d: ex %d: %d <= %d <= %d\n",
                           att_num, i, data->low[att_num], e.distinct_attribute_values[att_num], data->high[att_num]);
                if (e.distinct_attribute_values[att_num] <= data->high[att_num] &&
                    e.distinct_attribute_values[att_num] >= data->low[att_num]) {
                        if (0 || args.debug)
                            printf("Att:%d Val:%d Class:%d\n", att_num,
                                   e.distinct_attribute_values[att_num], e.containing_class_num);
                        avc[e.containing_class_num][e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        total_per_distinct[e.distinct_attribute_values[att_num] - data->low[att_num]]++;
                        gain_array[1][e.containing_class_num]++;
                }
            }

            i = 0;
            while (total_per_distinct[i] == 0 && i < num_distinct_values - 2) i++;
            while (i < num_distinct_values - 1) {
                j = i + 1;
                while (total_per_distinct[j] == 0 && j < num_distinct_values - 1) j++;
                if (total_per_distinct[j] != 0) {
                    for (k = 0; k < data->meta.num_classes; k++) {
                        split_hellinger[0] += avc[k][i];
                        split_hellinger[1] -= avc[k][i];
                        gain_array[0][k] += avc[k][i];
                        gain_array[1][k] -= avc[k][i];
                    }
                    //printf("%d: si = %d,%d ga0 = %d,%d ga1 = %d,%d\n", i, split_info[0], split_info[1],
                    //        gain_array[0][0], gain_array[0][1], gain_array[1][0], gain_array[1][1]);
                    
                    min_split = get_min_examples_per_split(data, args);

                    if (split_hellinger[0] >= min_split && split_hellinger[1] >= min_split) {
                        if (0 && args.debug)
                            printf("GA: %d %d %d %d\n", gain_array[0][0], gain_array[0][1],
                                                        gain_array[1][0], gain_array[1][1]);

                        //compute_gain --> compute_hellinger
                        hellinger = compute_hellinger(gain_array, 2, data->meta.num_classes);
                        if (hellinger > best_hellinger) {
                            best_hellinger = hellinger;
                            best_split_hellinger = sqrt(2);
                            best_split_low = i + data->low[att_num];
                            best_split_high = j + data->low[att_num];
                            if (0 && args.debug)
                                printf(":::best hellinger distance for att %d between %d/%d = %.14g\n", att_num,
                                       best_split_low, best_split_high, hellinger);
                        }
                        num_potential_splits++;
                    }
                }
                i = j;
            }

            if (num_potential_splits > 0) {
                best_hellinger -= dlog_2((double)num_potential_splits)/(float)data->meta.num_examples;
                if (best_hellinger > 0.0 || args.split_on_zero_gain) {
                    return_value = best_hellinger / best_split_hellinger;
                    *returned_high = best_split_high;
                    *returned_low = best_split_low;
                }
            }

            // Clean up
            free(total_per_distinct);
            for (i = 0; i < 2; i++)
                free(gain_array[i]);
            for (i = 0; i < data->meta.num_classes; i++)
                free(avc[i]);
        }
    } else if (data->meta.attribute_types[att_num] == DISCRETE) {
        if (data->discrete_used[att_num] == TRUE)
            return NO_SPLIT;

        int **gain_array, *split_hellinger;
        gain_array = (int **)malloc(data->meta.num_discrete_values[att_num] * sizeof(int*));
        split_hellinger = (int *)malloc(data->meta.num_discrete_values[att_num] * sizeof(int));

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            gain_array[i] = (int *)calloc(data->meta.num_classes, sizeof(int));
        for (i = 0; i < data->meta.num_examples; i++)
            gain_array[data->examples[i].distinct_attribute_values[att_num]][data->examples[i].containing_class_num]++;
        best_hellinger = compute_hellinger(gain_array, data->meta.num_discrete_values[att_num], data->meta.num_classes);

        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++) {
            split_hellinger[i] = 0;
            for (j = 0; j < data->meta.num_classes; j++)
                split_hellinger[i] += gain_array[i][j];
        }

        // Determine if enough splits are present
        min_split = get_min_examples_per_split(data, args);
        int max_val = 0;
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            if (split_hellinger[i] > max_val)
                max_val = split_hellinger[i];
        if (max_val <= data->meta.num_examples - min_split)
	    return_value = best_hellinger / sqrt(2);

        // Clean up
        free(split_hellinger);
        for (i = 0; i < data->meta.num_discrete_values[att_num]; i++)
            free(gain_array[i]);
        free(gain_array);
    }

    if (return_value > INTMIN)
        return return_value;
    else
        return NO_SPLIT;
}
