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
#include "crossval.h"
#include "subspaces.h"
#include "tree.h"
#include "util.h"

void apply_random_subspaces(CV_Subset data, CV_Subset *data_rs, Args_Opts args) {
    int i;
    int num_dimensions = (int)(args.random_subspaces * (float)data.meta.num_attributes / 100.0);
    int *array;
    
    if (num_dimensions == 0) {
        fprintf(stderr, "Error: zero dimensions in subspace\n");
        exit(-8);
    }
    
    copy_subset_meta(data, data_rs, -1);
    //printf("data_bite.high(s) is at memory location 0x%lx\n", &data.high);
    //printf("data_rs.high(s)   is at memory location 0x%lx\n", &(data_rs->high));
    point_subset_data(data, data_rs);
    //printf("data_bite.high(a) is at memory location 0x%lx\n", &data.high);
    //printf("data_rs.high(a)   is at memory location 0x%lx\n", &(data_rs->high));
    
    array = (int *)malloc(data_rs->meta.num_attributes * sizeof(int));
    for (i = 0; i < data_rs->meta.num_attributes; i++)
        array[i] = i;
    if (args.use_opendt_shuffle)
        _opendt_shuffle(data_rs->meta.num_attributes, array, args.data_path);
    else
        _knuth_shuffle(data_rs->meta.num_attributes, array);
    
    // We spoof the dataset into thinking these attributes are unsplitable
    for (i = 0; i < data_rs->meta.num_attributes - num_dimensions; i++) {
        data_rs->high[array[i]] = 0;
        data_rs->low[array[i]] = 1;
        data_rs->discrete_used[array[i]] = TRUE;
    }
    
    free(array);
}
