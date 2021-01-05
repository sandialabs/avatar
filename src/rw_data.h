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
#ifndef __RW_DATA__
#define __RW_DATA__

#include "schema.h"
#include "crossval.h"

//Modified by DACIESL June-04-08: Laplacean Estimates
//modified function prototype
int store_predictions_text(CV_Subset test_data, Vote_Cache cache, CV_Matrix matrix, CV_Voting votes, CV_Prob_Matrix prob_matrix, int fold, Args_Opts args);

int add_fold_data(int num_folds, CV_Dataset data, CV_Subset *train, int **fold_pop, Args_Opts args);
int read_opendt_names_file(CV_Dataset *data, CV_Class *class, Args_Opts args);
int read_names_file(CV_Metadata *meta, CV_Class *class, Args_Opts *args, Boolean update_skip_list);
int read_data_file(CV_Dataset *data, CV_Subset *sub, CV_Class *class, AV_SortedBlobArray *blob, char *ext, Args_Opts args);
//int store_predictions_text(CV_Subset test_data, Vote_Cache cache, CV_Matrix matrix, CV_Voting votes, int fold, Args_Opts args);
int _store_predictions_text(CV_Subset test_data, CV_Matrix matrix, Args_Opts args);
void read_metadata(FC_Dataset *ds, CV_Metadata *meta, Args_Opts *args);
void init_att_handling(CV_Metadata meta);
int process_attribute_char_value(char *att_value, int att_index, CV_Metadata meta);
int process_attribute_float_value(float att_value, int att_index);
void create_float_data(CV_Subset *data);
int add_attribute_char_value(char *a_val, CV_Subset *sub, int e_num, int a_num, int all_a, int line, char *file, int truth_col, char **elements);
void add_attribute_float_value(float a_val, CV_Subset *sub, int e_num, int a_num);
void datafile_to_string_array(char *file, int *num_lines, char ***data_lines, int *num_comments, char ***leading_comments);

void read_testing_data(FC_Dataset *ds, CV_Metadata train_meta, CV_Dataset *dataset, CV_Subset *subset, AV_SortedBlobArray *sorted_examples, Args_Opts *args);
void read_training_data(FC_Dataset *ds, CV_Dataset *dataset, CV_Subset *subset, AV_SortedBlobArray *sorted_examples, Args_Opts *args);
void init_fc(Args_Opts args);
void open_exo_datafile(FC_Dataset *ds, char *datafile);
int add_exo_data(FC_Dataset ds, int timestep, CV_Dataset *data, CV_Class *class, Boolean init);
int add_exo_metadata(FC_Dataset ds, int timestep, CV_Metadata *data, CV_Class *class);
void init_predictions(CV_Metadata data, FC_Dataset *dataset, int iter, Args_Opts args);
AV_ReturnCode _copy_and_delete_orig(FC_Mesh mesh, char *var_name, FC_Variable **new_var, double ***data);
int store_predictions(CV_Subset test_data, Vote_Cache cache, CV_Matrix matrix, CV_Voting votes, CV_Prob_Matrix prob_matrix, FC_Dataset dataset, int fold, Args_Opts args);

#endif // __RW_DATA__
