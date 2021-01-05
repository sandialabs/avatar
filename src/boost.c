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
#include "crossval.h"
#include "boost.h"
#include "evaluate.h"
#include "tree.h"
#include "av_rng.h"

static struct ParkMiller* rng;
static struct AVDiscRandVar* var;

void init_weighted_rng(int num, double *weights, Args_Opts args) {
    // Initialize random number generator
    rng = malloc(sizeof(struct ParkMiller));
    av_pm_default_init(rng, args.random_seed);
    var = malloc(sizeof(struct AVDiscRandVar));
    var->dist = weights;
    var->n = num;
}

void free_weighted_rng() {
    free(var);
    free(rng);
}

int get_next_weighted_sample() {
    return av_discrete_rand(var);
}

void reset_weights(int num, double *weights) {
    if (var->dist != weights)
    {
      var->dist = weights;
      var->n = num;
    }
}

// Computes epsilon, beta, and new weights
// Returns beta
double update_weights(double **weights, DT_Node *tree, CV_Subset data) {
    int i;
    int leaf_node;
    double epsilon = 0.0;
    short *w;
    w = (short *)calloc(data.meta.num_examples, sizeof(short));
    int num_wrong = 0;
    //printf("\n");
    
    for (i = 0; i < data.meta.num_examples; i++) {
        int this_class = classify_example(tree, data.examples[i], data.float_data, &leaf_node);
        if (this_class != data.examples[i].containing_class_num) {
            //printf("%d was wrong\n", i);
            num_wrong++;
            epsilon += (*weights)[i];
        } else {
            w[i] = 1;
        }
    }
    
    double beta = epsilon / (1.0 - epsilon);
    //printf("\nComputed beta (%g) = %g for %d wrong\n", epsilon, beta, num_wrong);
    
    // Find normalization factor to make new weights a distribution
    double Z = 0;
    for (i = 0; i < data.meta.num_examples; i++)
        Z += (*weights)[i] * (w[i]==1 ? beta : 1.0);
    //printf("Normalization factor = %g\n", Z);
    for (i = 0; i < data.meta.num_examples; i++) {
        //printf("Weight for %3d goes from %g to ", i, (*weights)[i]);
        (*weights)[i] *= (w[i]==1 ? beta : 1.0) / Z;
        //printf("%g\n", (*weights)[i]);
    }
    
    return beta;
}

void get_boosted_set(CV_Subset *src, CV_Subset *bst) {
    int i, j;
    
    copy_subset_meta(*src, bst, src->meta.num_examples);
    copy_subset_data(*src, bst);
    bst->meta.num_examples = src->meta.num_examples;
    for (i = 0; i < src->meta.num_examples; i++) {
        j = get_next_weighted_sample();
        bst->meta.num_examples_per_class[src->examples[j].containing_class_num]++;
        copy_example_data(src->meta.num_attributes, src->examples[j], &(bst->examples[i]));
    }
}

