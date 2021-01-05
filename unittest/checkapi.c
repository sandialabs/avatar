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
#include <string.h>
#include "check.h"
#include "checkall.h"
#include "../src/crossval.h"
#include "../src/avatar_api.h"
#include "../src/crossval.h"

struct Avatar_struct{
  CV_Dataset Train_Dataset, Test_Dataset;
  CV_Subset Train_Subset, Test_Subset;
  Vote_Cache Cache;
  AV_SortedBlobArray Train_Sorted_Examples, Test_Sorted_Examples;
  FC_Dataset ds, pred_prob;
  CV_Partition Partitions;
  DT_Ensemble* Test_Ensembles;
  Args_Opts Args;
};

START_TEST(check_avatar_train)
{
    Avatar_handle* a;
    a = calloc(1, sizeof(Avatar_handle));
    int argc;
    char * argv[4];
    argv[0] = "./avatar_api.c";
    argv[1] = "-fapi_testing";
    argv[2] = "--format=avatar";
    argv[3] = "--seed=24601";

    argc = 4;

    FILE *datafile;
    char datastr[10000];
    char* filename = "api_testing.data";

    char train_string[10000];
    
    datafile = fopen(filename, "r");
    while (fgets(datastr,10000, datafile) != NULL)
	strcat(train_string,datastr);

    fclose(datafile);

    FILE *namesfile;
    char namesstr[10000];
    filename = "api_testing.names";

    char names_string[10000];
    
    namesfile = fopen(filename, "r");
    while (fgets(namesstr,10000, namesfile) != NULL)
	strcat(names_string,namesstr);

    fclose(namesfile);

    a = avatar_train(argc, argv, names_string, 1, train_string, 1);

    CV_Subset dataset = {0};
    dataset = a->Train_Subset;
    DT_Ensemble ensemble = {0};
    ensemble = a->Test_Ensembles[0];

    //Check train data meta
    fail_unless(dataset.meta.num_examples == 12, "Number of training examples should be 12 but is %d", dataset.meta.num_examples);
    fail_unless(dataset.meta.num_classes == 2, "Number of classes should be 2 but is %d", dataset.meta.num_classes);
    fail_unless(dataset.meta.num_examples_per_class[0] == 8, "Number of examples in first class should be 8 but is %d", dataset.meta.num_examples_per_class[0]);
    fail_unless(dataset.meta.num_examples_per_class[1] == 4, "Number of examples in second class should be 4 but is %d", dataset.meta.num_examples_per_class[1]);

    //Check ensemble stats 
    fail_unless(ensemble.num_trees==1, "Num trees should be 1 but is %d", ensemble.num_trees);
    fail_unless(ensemble.num_classes==2, "Num classes should be 2 but is %d", ensemble.num_classes);
    fail_unless(ensemble.num_attributes==4, "Num attributes should be 4 but is %d", ensemble.num_attributes);
    fail_unless(ensemble.Trees[0][1].Node_Value.class_label==0, "The second node should have class value 0 but has value %d", ensemble.Trees[0][1].Node_Value.class_label);
    fail_unless(ensemble.Trees[0][2].Node_Value.class_label==1, "The third node should have class value 1 but has value %d", ensemble.Trees[0][2].Node_Value.class_label);

    free(a);
}
END_TEST

START_TEST(check_avatar_load)
{
    Avatar_handle* a;
    a = malloc(sizeof(Avatar_handle));

    a = avatar_load("data/api_testing", "data/api_testing.names", 0, "data/api_testing.trees", 0);

    CV_Subset dataset;
    dataset = a->Train_Subset;
    CV_Metadata meta = dataset.meta;
    DT_Ensemble ensemble;
    ensemble = a->Test_Ensembles[0];

    //Check ensemble stats 
    fail_unless(ensemble.num_trees==1, "Num trees should be 1 but is %d\n", ensemble.num_trees);
    fail_unless(ensemble.num_classes==2, "Num classes should be 2 but is %d", ensemble.num_classes);
    fail_unless(ensemble.num_attributes==4, "Num attributes should be 4 but is %d", ensemble.num_attributes);
    fail_unless(ensemble.num_training_examples == 12, "Number of training examples should be 12", ensemble.num_training_examples);

}
END_TEST

START_TEST(check_avatar_test)
{
    Avatar_handle* a;
    a = avatar_load("api_testing", NULL, 0, NULL, 0); 

    char* input = "5,37,1,3,0\n8,31,1,3,0\n9,23,1,3,0\n12,21,1,4,0\n19,19,1,4,0\n17,17,1,4,0\n3,13,2,4,0\n24,11,2,4,0\n50,7,2,4,0\n26,5,2,3,0\n88,3,2,3,0\n";
    
    int* preds = (int*)malloc(11 * sizeof(int));
    float* probs = (float*)malloc(2 * 11 * sizeof(float));


    avatar_test(a, input, 1, preds, probs);

    fail_unless(preds[0]==0, "1st input should be classified as zero but it's classified as %d", preds[0]);
    fail_unless(preds[1]==0, "2nd input should be classified as zero but it's classified as %d", preds[1]);
    fail_unless(preds[2]==0, "3rd input should be classified as zero but it's classified as %d", preds[2]);
    fail_unless(preds[3]==0, "4th input should be classified as zero but it's classified as %d", preds[3]);
    fail_unless(preds[4]==0, "5th input should be classified as zero but it's classified as %d", preds[4]);
    fail_unless(preds[5]==0, "6th input should be classified as zero but it's classified as %d", preds[5]);
    fail_unless(preds[6]==0, "7th input should be classified as zero but it's classified as %d", preds[6]);
    fail_unless(preds[7]==1, "8th input should be classified as one but it's classified as %d", preds[7]);
    fail_unless(preds[8]==1, "9th input should be classified as one but it's classified as %d", preds[8]);
    fail_unless(preds[9]==1, "10th input should be classified as one but it's classified as %d", preds[9]);
    fail_unless(preds[10]==1, "11th input should be classified as one but it's classified as %d", preds[10]);

    avatar_cleanup(a);

    fail_unless(1==1, "Avatar cleanup should not have any issues");

}
END_TEST


Suite *api_suite(void)
{
    Suite *suite = suite_create("API");
    
    TCase *tc_api = tcase_create(" API ");
    
    suite_add_tcase(suite, tc_api);
    //tcase_add_test(tc_api, check_avatar_train);
    tcase_add_test(tc_api, check_avatar_load);
    //tcase_add_test(tc_api, check_avatar_test);
    
    return suite;
}
