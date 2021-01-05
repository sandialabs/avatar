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
//Added by DACIESL June-17-08: Performance Metrics
//added function prototypes
void print_performance_metrics(CV_Subset data, CV_Prob_Matrix matrix, char ** class_names);
void calculate_precision(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_recall(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_fmeasure(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_brier_score(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_nce(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_auroc(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_calibration(CV_Subset data, CV_Prob_Matrix matrix, double * L);
void calculate_refinement(CV_Subset data, CV_Prob_Matrix matrix, double * L);
int compare_ref_cal (const void * a, const void * b);


//Added by DACIESL June-05-08: Laplacean Estimates
//added function prototypes
void build_probability_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Prob_Matrix *matrix);
void build_boost_probability_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Prob_Matrix *matrix);
float *find_example_probabilities(DT_Node *tree, CV_Example example, float **xlate, int *leaf_node);
float *_find_example_probabilities(DT_Node *tree, int node, CV_Example example, float **xlate, int *leaf_node);
void build_probability_matrix_for_ivote(Vote_Cache cache, CV_Prob_Matrix *matrix);
void build_boost_probability_matrix_for_ivote(Vote_Cache cache, CV_Prob_Matrix *matrix);

int count_nodes(DT_Node *tree);
void _count_nodes(DT_Node *tree, int node, int *count);
int classify_example(DT_Node *tree, CV_Example example, float **xlate, int *leaf_node);
int _classify_example(DT_Node *tree, int node, CV_Example example, float **xlate, int *leaf_node);
void build_prediction_matrix_for_ivote(CV_Subset data, Vote_Cache cache, CV_Matrix *matrix);
void build_prediction_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Matrix *matrix);
void build_boost_prediction_matrix(CV_Subset data, DT_Ensemble ensemble, CV_Matrix *matrix);
void build_boost_prediction_matrix_for_ivote(Vote_Cache cache, CV_Matrix *matrix);
void concat_ensembles(int num_ensembles, DT_Ensemble *in, DT_Ensemble *out);
float compute_voted_accuracy(CV_Matrix matrix, int ***confusion_matrix, Args_Opts args);
float compute_boosting_accuracy(CV_Matrix matrix, int ***confusion_matrix);
float compute_average_accuracy(CV_Matrix matrix);
void count_class_votes_from_matrix(int example_num, CV_Matrix matrix, int **votes);
int find_best_class_from_matrix(int example_num, CV_Matrix matrix, Args_Opts args, int seed_flag, int clean);
void print_confusion_matrix(int num_classes, int **confusion_matrix, char **class_names);
//void _generate_confusion_matrix(CV_Matrix matrix, int ***confusion_matrix, Args_Opts args);
void print_pred_matrix(char *pre, CV_Matrix matrix);
void print_boosting_pred_matrix(char *pre, CV_Matrix matrix);
char* set_to_NULL(void);
void calculate_accuracy(CV_Subset data, CV_Prob_Matrix matrix, double * L);

