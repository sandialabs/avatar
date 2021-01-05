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
#include "../src/knn.h"
#include "../src/smote.h"
#include "../src/util.h"
#include "../src/rw_data.h"

void set_up_Args_for_read(Args_Opts *args);
void set_up_Args_for_read(Args_Opts *args) {
    
    // Stuff to read the data
    args->format = AVATAR_FORMAT;
    args->datafile = strdup("./data/smote_4d.data");
    args->base_filestem = strdup("smote_4d");
    args->data_path = strdup("./data");
    args->names_file = strdup("./data/smote_4d.names");
    args->do_training = TRUE;
    args->num_skipped_features = 0;
    args->skipped_features = NULL;
    args->truth_column = 5;
    args->exclude_all_features_above = -1;
    
}

START_TEST(check_new_examples_closed)
{
    int i, j;
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;
    
    set_up_Args_for_read(&Args);
    
    // Stuff for this test
    Args.do_smote = TRUE;
    Args.num_minority_classes = 1;
    Args.minority_classes = (int *)malloc(sizeof(int));
    Args.minority_classes[0] = 1;
    Args.proportions = (float *)malloc(sizeof(float));
    Args.proportions[0] = 0.5;
    Args.smote_knn = 3;
    Args.smote_Ln = 2;
    Args.smote_type = CLOSED_SMOTE;
    Args.random_seed = 3;

    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    
    float true_catt_values[16][2] = {
                                      {1,37},
                                      {3,31},
                                      {5,23},
                                      {7,21},
                                      {11,19},
                                      {13,17},
                                      {17,13},
                                      {19,11},
                                      {21,7},
                                      {23,5},
                                      {31,3},
                                      {37,1},
                                      {24.25448608,4.68637848},
                                      {23.31210136,6.13296223},
                                      {28.24777603,3.68805599},
                                      {26.31826591,4.17043352}
    };
    char *true_datt_values[16][2] = {
                                      {"A","C"},
                                      {"A","C"},
                                      {"A","C"},
                                      {"A","D"},
                                      {"A","D"},
                                      {"A","D"},
                                      {"B","D"},
                                      {"B","D"},
                                      {"B","D"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
    };
    
   // smote(&Data, &Data, &blob, Args);
    
    // Check new number of examples and class proportions
    //fail_unless(Data.meta.num_examples == 16, "Incorrect new number of examples");
   // fail_unless(Data.meta.num_examples_per_class[0] == 8, "Incorrect new number of examples in class 0");
    //fail_unless(Data.meta.num_examples_per_class[1] == 8, "Incorrect new number of examples in class 1");
    
    // Check attribute values for original examples
  //  for (i = 0; i < 12; i++) {
      //  for (j = 0; j < 2; j++) {
       //     fail_unless(av_eqf(Data.float_data[j][Data.examples[i].distinct_attribute_values[j]],
     //                   true_catt_values[i][j]), "Original example continuous att values are not correct");
     //   }
       // for (j = 2; j < 4; j++) {
     //       fail_unless(! strcmp(Data.meta.discrete_attribute_map[j][Data.examples[i].distinct_attribute_values[j]],
   //                     true_datt_values[i][j-2]), "Original example discrete att values are not correct");
     //   }
   // }
    
    // Check attribute values for new examples
    //for (i = 12; i < 16; i++) {
       // for (j = 0; j < 2; j++) {
         //   fail_unless(av_eqf(Data.float_data[j][Data.examples[i].distinct_attribute_values[j]],
 //                       true_catt_values[i][j]), "SMOTEd example continuous att values are not correct");
   //     }
    //    for (j = 2; j < 4; j++) {
 //           fail_unless(! strcmp(Data.meta.discrete_attribute_map[j][Data.examples[i].distinct_attribute_values[j]],
  //                      true_datt_values[i][j-2]), "SMOTEd example discrete att values are not correct");
    //    }
    //}
}
END_TEST

START_TEST(check_new_examples_open)
{
    int i, j;
    CV_Dataset dataset = {0};
    CV_Subset Data = {0};
    Args_Opts Args = {0};
    AV_SortedBlobArray blob;

    set_up_Args_for_read(&Args);
    
    // Stuff for this test
    Args.do_smote = TRUE;
    Args.num_minority_classes = 1;
    Args.minority_classes = (int *)malloc(sizeof(int));
    Args.minority_classes[0] = 1;
    Args.proportions = (float *)malloc(2*sizeof(float));
    Args.proportions[0] = 0.5;
    Args.proportions[1] = 0.5;
    Args.smote_knn = 3;
    Args.smote_Ln = 1;
    Args.smote_type = OPEN_SMOTE;
    Args.random_seed = 4;

    av_initSortedBlobArray(&blob);
    read_training_data(NULL, &dataset, &Data, &blob, &Args);
    
    float true_catt_values[16][2] = {
                                      {1,37},
                                      {3,31},
                                      {5,23},
                                      {7,21},
                                      {11,19},
                                      {13,17},
                                      {17,13},
                                      {19,11},
                                      {21,7},
                                      {23,5},
                                      {31,3},
                                      {37,1},
                                      {20.91031456, 7.13452768},
                                      {35.26171494, 1.49665296},
                                      {19.56562805, 9.86874294},
                                      {35.75735092, 1.46599293}
    };
    char *true_datt_values[16][2] = {
                                      {"A","C"},
                                      {"A","C"},
                                      {"A","C"},
                                      {"A","D"},
                                      {"A","D"},
                                      {"A","D"},
                                      {"B","D"},
                                      {"B","D"},
                                      {"B","D"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","C"},
                                      {"B","D"},
                                      {"B","C"},
                                      {"B","D"},
                                      {"B","C"},
    };

    smote(&Data, &Data, &blob, Args);

    // Check new number of examples and class proportions
    fail_unless(Data.meta.num_examples == 16, "Incorrect new number of examples");
    fail_unless(Data.meta.num_examples_per_class[0] == 8, "Incorrect new number of examples in class 0");
    fail_unless(Data.meta.num_examples_per_class[1] == 8, "Incorrect new number of examples in class 1");


    // Check attribute values for original examples
    for (i = 0; i < 12; i++) {
        for (j = 0; j < 2; j++) {
            fail_unless(av_eqf(Data.float_data[j][Data.examples[i].distinct_attribute_values[j]],
                        true_catt_values[i][j]), "Original example continuous att values are not correct");
        }
        for (j = 2; j < 4; j++) {
            fail_unless(! strcmp(Data.meta.discrete_attribute_map[j][Data.examples[i].distinct_attribute_values[j]],
                        true_datt_values[i][j-2]), "Original example discrete att values are not correct");
        }
    }
    
    // Check attribute values for new examples
    for (i = 12; i < 16; i++) {
        for (j = 0; j < 2; j++) {
            fail_unless(av_eqf(Data.float_data[j][Data.examples[i].distinct_attribute_values[j]],
                        true_catt_values[i][j]), "SMOTEd example continuous att values are not correct");
        }
        for (j = 2; j < 4; j++) {
            fail_unless(! strcmp(Data.meta.discrete_attribute_map[j][Data.examples[i].distinct_attribute_values[j]],
                        true_datt_values[i][j-2]), "SMOTEd example discrete att values are not correct");
        }
    }
}
END_TEST

START_TEST(check_compute_deficits)
{
    // 10 classes with populations: 5, 10, 100, 500, 18, 4, 500, 450, 1000, 750, 600
    // Desired proportions for min classes: 0.1, 0.1, 0.1, 0.1
    
    int i;
    
    CV_Metadata meta;
    Args_Opts args;
    int new_total;
    int *class_deficits;
    
    int nepc[] = {5,10,100,500,18,4,450,1000,750,600};
    float props[] = {0.1,0.1,0.1,0.1};
    int minc[] = {0,1,4,5};
    int truth_class_deficits[] = {562,557,0,0,549,563,0,0,0,0};

    meta.num_classes = 10;
    meta.num_examples = 0;
    for (i = 0; i < meta.num_classes; i++)
        meta.num_examples += nepc[i];
    args.num_minority_classes = 4;
    meta.num_examples_per_class = nepc;
    args.proportions = props;
    args.minority_classes = minc;
    
    compute_deficits(meta, args, &new_total, &class_deficits);
    
    int  truth_new_total = 0;
    for (i = 0; i < meta.num_classes; i++) {
        truth_new_total += nepc[i] + truth_class_deficits[i];
    }
    
    fail_unless(new_total == truth_new_total, "New number of examples not correct");
    fail_unless(! memcmp(class_deficits, truth_class_deficits, meta.num_classes*sizeof(int)), "Deficits not correct");
}
END_TEST

// *********************************************
// ***** Populate the Suite with the tests
// *********************************************

Suite *smote_suite(void)
{
    Suite *suite = suite_create("SMOTE");
    
    TCase *tc_smote = tcase_create(" - SMOTE ");
    
    // general library init/final tests
    suite_add_tcase(suite, tc_smote);
    tcase_add_test(tc_smote, check_compute_deficits);
    tcase_add_test(tc_smote, check_new_examples_closed);
    tcase_add_test(tc_smote, check_new_examples_open);
    
    return suite;
}
