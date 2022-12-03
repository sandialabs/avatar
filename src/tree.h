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
#ifndef __TREE__
#define __TREE__

#include "crossval.h"

//Added by DACIESL June-03-08: Laplacean Estimates
//Function prototypes for Laplacean Estimates support
int * find_class_count(CV_Subset *data);
//void find_class_count(CV_Subset *data, int **class_count);
float *find_class_probs(CV_Subset *data, int *class_count);
void check_tree_version(int fold_num, Args_Opts *args);

void train(CV_Subset *data, DT_Ensemble *ensemble, int fold_num, Args_Opts args);
void train_ivote(CV_Subset train_data, CV_Subset test_data, int fold_num, Vote_Cache *cache, Args_Opts args);
void build_tree(CV_Subset *data, DT_Node **tree, Tree_Bookkeeping *Books, Args_Opts args);
int stop(CV_Subset *data, Args_Opts args);
void find_best_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args);
void find_trt_split( CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args);
void find_ert_split( CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args);
void find_random_forest_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args);
void find_random_attribute_split(CV_Subset *data, DT_Node *tree, int *returned_high, int *returned_low, Args_Opts args);
int is_pure(CV_Subset *data);
int find_best_class(CV_Subset *data);
int errors_guessing_best_class(CV_Subset *data);
//void write_tree_file_header_E(DT_Ensemble ensemble, int fold_num, Args_Opts args);
char* write_tree_file_header(int num_trees, CV_Metadata meta, int fold_num, char *tree_filename, Args_Opts args);
void save_ensemble(DT_Ensemble ensemble, CV_Metadata data, int fold_num, Args_Opts args, int num_classes);
char* build_output_filename(int fold_num, char *filename, Args_Opts args);
void save_tree(DT_Node *tree, int fold_num, int tree_num, Args_Opts args, int num_classes);
void _save_node(DT_Node *tree, int node, FILE *fh, int num_classes);
void copy_dataset_meta(CV_Dataset src, CV_Subset *dest, int population);
void copy_subset_meta(CV_Subset src, CV_Subset *dest, int population);
void point_subset_data(CV_Subset src, CV_Subset *dest);
void copy_subset_data(CV_Subset src, CV_Subset *dest);
void copy_example_metadata(CV_Example src, CV_Example *dest);
void copy_example_data(int num_atts, CV_Example src, CV_Example *dest);
void read_ensemble_metadata(FILE *fh, DT_Ensemble *ensemble, int force_num_trees, Args_Opts *args);
void read_ensemble(DT_Ensemble *ensemble, int fold_num, int force_num_trees, Args_Opts *args);
void _read_tree(char *initial_str, DT_Node **tree, Tree_Bookkeeping *book, FILE *fh, int num_classes);
int check_stopping_algorithm(int init, int part_num, float raw_accuracy, int trees, float *max_raw, char *oob_filename, Args_Opts args);
double _fminf(double x, double y);

void test(CV_Subset test_data, int num_ensembles, DT_Ensemble *ensemble, FC_Dataset dataset, int fold_num, Args_Opts args);
void test_ivote(CV_Subset test_data, Vote_Cache cache, FC_Dataset dataset, int fold_num, Args_Opts args);

int check_tree_validity(DT_Node* tree, int node, int num_classes, int num_nodes);
void check_ensemble_validity(const char * label,DT_Ensemble *ensemble);

#endif

