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
#define LOG2 0.69314718055994528622676398299518


//Added by DACIESL June-02-08: HDDT CAPABILITY
//Function prototypes for HDDT support
float best_hellinger_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args);
float compute_hellinger_from_array(int **array, int num, int a, int b);
float compute_hellinger(int **array, int num_attributes, int num_classes);

float best_trt_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, float *cut_threshold, Args_Opts args);
float best_ert_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, float *cut_threshold, Args_Opts args);
float best_c45_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args);
float best_gain_ratio_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args);
float best_gain_split(CV_Subset *data, int att_num, int *returned_high, int *returned_low, Args_Opts args);
//float _log_2(double x);
double dlog_2(double x);
double dlog_2_int(int x);
//float _compute_info(CV_Subset *data);
float compute_info_from_array(int *array, int num);
float compute_split_info(int *array, int num_splits);
float compute_gain(int **array, int num_attributes, int num_classes);
int get_min_examples_per_split(CV_Subset *data, Args_Opts args);

