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
#ifndef __IVOTE__
#define __IVOTE__

typedef struct tree_bookkeeping_struct Tree_Bookkeeping;
typedef struct crossval_dataset_struct CV_Dataset;
typedef struct crossval_sub_dataset_struct CV_Subset;
typedef struct av_sorted_blob_array AV_SortedBlobArray;
typedef struct voting_cache_struct Vote_Cache;
typedef struct dt_node_struct DT_Node;
typedef struct args_and_opts_struct Args_Opts;

void make_bite(CV_Subset *src, CV_Subset *bite, Vote_Cache *cache, Args_Opts args);
double compute_oob_error_rate(DT_Node *tree, CV_Subset train_data, Vote_Cache *cache, Args_Opts args);
double compute_test_error_rate(DT_Node *tree, CV_Subset test_data, Vote_Cache *cache, Args_Opts args);
void initialize_cache(Vote_Cache *lcache, int n_cl, int n_train_ex, int n_test_ex, int free_first);

#endif
