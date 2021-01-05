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
#include "ivote.h"
#include "tree.h"
#include "util.h"
#include "evaluate.h"
#include "att_noising.h"

void compute_noised_oob_error_rate(DT_Node *tree, CV_Subset data, Vote_Cache **cache, Args_Opts args) {
    int i, j;
    int *oob_att_values;
    int *randomized;
    int num_oob_atts;
    int malloc_size;
    
    for (i = 0; i < data.meta.num_attributes; i++) {
        // Get the attribute values for oob examples
        malloc_size = data.meta.num_examples / 3;
        oob_att_values = (int *)malloc(malloc_size * sizeof(int));
        randomized = (int *)malloc(malloc_size * sizeof(int));
        num_oob_atts = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            if (data.examples[j].in_bag == FALSE) {
                num_oob_atts++;
                if (num_oob_atts > malloc_size) {
                    malloc_size *= 2;
                    oob_att_values = (int *)realloc(oob_att_values, malloc_size * sizeof(int));
                    randomized = (int *)realloc(randomized, malloc_size * sizeof(int));
                }
                oob_att_values[num_oob_atts-1] = data.examples[j].distinct_attribute_values[i];
                randomized[num_oob_atts-1] = data.examples[j].distinct_attribute_values[i];
            }
        }
        
        // Randomize them
        _knuth_shuffle(num_oob_atts, randomized);
        
        // Assign the random attribute values to oob examples
        num_oob_atts = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            if (data.examples[j].in_bag == FALSE) {
                num_oob_atts++;
                data.examples[j].distinct_attribute_values[i] = randomized[num_oob_atts-1];
            }
        }
        
        // Compute oob error
        cache[i]->oob_error = compute_oob_error_rate(tree, data, cache[i], args);
        
        // Reset attribute values for oob examples
        num_oob_atts = 0;
        for (j = 0; j < data.meta.num_examples; j++) {
            if (data.examples[j].in_bag == FALSE) {
                num_oob_atts++;
                data.examples[j].distinct_attribute_values[i] = oob_att_values[num_oob_atts-1];
            }
        }
        
        // Free
        free(oob_att_values);
        free(randomized);
    }
}
