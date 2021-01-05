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
#include <math.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/knn.h"
#include "../src/rw_data.h"

void set_up_Args_for_knn_read(int d, Args_Opts *args);
void set_up_Args_for_knn_read(int d, Args_Opts *args) {
    
    // Stuff to read the data
    args->format = AVATAR_FORMAT;
    args->datafile = (char *)malloc(128 * sizeof(char));
    sprintf(args->datafile, "./data/smote_%dd.data", d);
    args->base_filestem = (char *)malloc(128 * sizeof(char));
    sprintf(args->base_filestem, "smote_%dd", d);
    args->data_path = strdup("./data");
    args->names_file = (char *)malloc(128 * sizeof(char));
    sprintf(args->names_file, "./data/smote_%dd.names", d);
    args->do_training = TRUE;
    args->num_skipped_features = 0;
    args->skipped_features = NULL;
    args->truth_column = d+1;
    args->exclude_all_features_above = -1;
    
}

START_TEST(check_stdev_5d)
{
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;
    set_up_Args_for_knn_read(5, &Args);
    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    float median = compute_median_of_stdev(Data);
    fail_unless(av_eqf(median, 10.71862346), "Median not correct");
}
END_TEST

START_TEST(check_stdev_6d)
{
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;
    set_up_Args_for_knn_read(6, &Args);
    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    float median = compute_median_of_stdev(Data);
    fail_unless(av_eqf(median, 8.81136426), "Median not correct");
}
END_TEST

START_TEST(brute_force_5d_open)
{
    int i;
    Nearest_Neighbors **kNN;

    // Truth is for example 5
    int P = 5;
    int N = 7;
    int neighbor_truth_L1[] = { 4,3,6,7,8,2,1 };
    double distance_truth_L1[] = {  6,
                                   14,
                                   18.71862346,
                                   26.71862346,
                                   32.71862316,
                                   36.71862346,
                                   46.71862346
                                 };
    int neighbor_truth_L2[] = { 4,3,6,7,8,2,9 };
    double distance_truth_L2[] = {  3.464101615,
                                    8.246211251,
                                   11.78511302,
                                   14.24390708,
                                   16.69996663,
                                   18.94436298,
                                   23.19003602
                                 };
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;
    set_up_Args_for_knn_read(5, &Args);
    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    compute_knn(Data, N, 1, OPEN_SMOTE, &kNN);

    int *neighbor_L1;
    double *distance_L1;
    neighbor_L1 = (int *)malloc(N * sizeof(int));
    distance_L1 = (double *)malloc(N * sizeof(double));
    for (i = 0; i < N; i++) {
        neighbor_L1[i] = kNN[P][i].neighbor;
        distance_L1[i] = kNN[P][i].distance;
    }


    fail_unless(! memcmp(neighbor_L1, neighbor_truth_L1, N*sizeof(int)), "NN don't match for 5d open L1");
    for (i = 0; i < N; i++) {
        //printf("Comparing L1 %.12f and %.12f\n", distance_L1[i], distance_truth_L1[i]);
        fail_unless(av_eqf(distance_L1[i], distance_truth_L1[i]), "distances don't match for 5d open L1");
    }
    
    for (i = 0; i < Data.meta.num_examples; i++)
        free(kNN[i]);
    free(kNN);
    
    compute_knn(Data, N, 2, OPEN_SMOTE, &kNN);
    
    int *neighbor_L2;
    double *distance_L2;
    neighbor_L2 = (int *)malloc(N * sizeof(int));
    distance_L2 = (double *)malloc(N * sizeof(double));
    for (i = 0; i < N; i++) {
        neighbor_L2[i] = kNN[P][i].neighbor;
        distance_L2[i] = kNN[P][i].distance;
    }

    fail_unless(! memcmp(neighbor_L2, neighbor_truth_L2, N*sizeof(int)), "NN don't match for 5d open L2");
    for (i = 0; i < N; i++) {
        //printf("Comparing L2 %.12f and %.12f\n", distance_L2[i], distance_truth_L2[i]);
        fail_unless(av_eqf(distance_L2[i], distance_truth_L2[i]), "distances don't match for 5d open L2");
    }
    
    for (i = 0; i < Data.meta.num_examples; i++)
        free(kNN[i]);
    free(kNN);

}
END_TEST

START_TEST(brute_force_5d_closed)
{
    int i;
    Nearest_Neighbors **kNN;
    // Truth is for example 5
    int P = 5;
    int N = 7;
    int neighbor_truth_L1[] = { 4,3,6,7,2,1,0 };
    double distance_truth_L1[] = {  6,
                                   14,
                                   18.71862346,
                                   26.71862346,
                                   36.71862346,
                                   46.71862346,
                                   54.71862346
                                 };
    int neighbor_truth_L2[] = { 4,3,6,7,2,1,0 };
    double distance_truth_L2[] = {  3.464101615,
                                    8.246211251,
                                   11.78511302,
                                   14.24390708,
                                   18.94436298,
                                   24.55379581,
                                   29.03254878
                                 };
    
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;
    set_up_Args_for_knn_read(5, &Args);
    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    
    compute_knn(Data, N, 1, CLOSED_SMOTE, &kNN);
    
    int *neighbor_L1;
    double *distance_L1;
    neighbor_L1 = (int *)malloc(N * sizeof(int));
    distance_L1 = (double *)malloc(N * sizeof(double));
    for (i = 0; i < N; i++) {
        neighbor_L1[i] = kNN[P][i].neighbor;
        distance_L1[i] = kNN[P][i].distance;
    }
                       
    fail_unless(! memcmp(neighbor_L1, neighbor_truth_L1, N*sizeof(int)), "NN don't match for 5d closed L1");
    for (i = 0; i < N; i++) {
        //printf("Comparing L1 %.12f and %.12f\n", distance_L1[i], distance_truth_L1[i]);
        fail_unless(av_eqf(distance_L1[i], distance_truth_L1[i]), "distances don't match for 5d closed L1");
    }
    
    for (i = 0; i < Data.meta.num_examples; i++)
        free(kNN[i]);
    free(kNN);
    
    compute_knn(Data, N, 2, CLOSED_SMOTE, &kNN);
    
    int *neighbor_L2;
    double *distance_L2;
    neighbor_L2 = (int *)malloc(N * sizeof(int));
    distance_L2 = (double *)malloc(N * sizeof(double));
    for (i = 0; i < N; i++) {
        neighbor_L2[i] = kNN[P][i].neighbor;
        distance_L2[i] = kNN[P][i].distance;
    }
                       
    fail_unless(! memcmp(neighbor_L2, neighbor_truth_L2, N*sizeof(int)), "NN don't match for 5d closed L2");
    for (i = 0; i < N; i++) {
        //printf("Comparing L2 %.12f and %.12f\n", distance_L2[i], distance_truth_L2[i]);
        fail_unless(av_eqf(distance_L2[i], distance_truth_L2[i]), "distances don't match for 5d closed L2");
    }
    
    for (i = 0; i < Data.meta.num_examples; i++)
        free(kNN[i]);
    free(kNN);
}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *knn_suite(void)
{
    Suite *suite = suite_create("KNN");
    
    TCase *tc_knn = tcase_create(" - KNN ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_knn);
    tcase_add_test(tc_knn, brute_force_5d_open);
#if 0
    tcase_add_test(tc_knn, brute_force_5d_closed);
    tcase_add_test(tc_knn, check_stdev_5d);
    tcase_add_test(tc_knn, check_stdev_6d);
#endif    
    return suite;
}
