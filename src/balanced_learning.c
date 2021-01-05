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
#include <math.h>
#include "crossval.h"
#include "balanced_learning.h"
#include "array.h"
#include "tree.h"
#include "util.h"
#include "skew.h"

/*
 * Populates the next clump with examples.
 *
 */

void get_next_balanced_set(int cycle, CV_Subset *src, CV_Subset *bal, Args_Opts args) {
    int i;
    static int num_clumps;
    static int ex_per_part;
    
    
    int *per_class_count;
    per_class_count = (int *)calloc(src->meta.num_classes, sizeof(int));
    
    // First time through, figure out number of clumps and the number of examples per clump
    if (cycle == 0)
        compute_number_of_clumps(src->meta, &args, &num_clumps, &ex_per_part);
    // Cycle should go from 0 to num_clumps then back to 0
    while (cycle >= num_clumps)
        cycle -= num_clumps;
    
    // Initialize the balanced learning dataset
    copy_subset_meta(*src, bal, ex_per_part);
    copy_subset_data(*src, bal);
    for (i = 0; i < src->meta.num_classes; i++)
        bal->meta.num_examples_per_class[i] = _num_per_class_per_clump(i, num_clumps, src->meta, args);
    
    // Go through all examples. If the example's bl_clump_num is within range, add this example to the
    // balanced learning dataset. The bl_clump_num numbers go up from 0 for each class separately, so
    // each class is added independently until the proper number of examples is reached. Basically,
    // take all the example in one class, randomize them, then append the sequence to itself. Then
    // Take the first num_this_class examples for the first set, the second num_this_class examples
    // for the second set, etc.
    for (i = 0; i < src->meta.num_examples; i++) {
        int class = src->examples[i].containing_class_num;
        int num_this_class = src->meta.num_examples_per_class[class];
        int num_this_part = bal->meta.num_examples_per_class[class];
        
        int low1 = cycle * num_this_part;
        int low2 = cycle * num_this_part;
        int high1 = low1 + num_this_part;
        int high2 = low2 + num_this_part;
        if (high2 >= num_this_class && low2 >= num_this_class) {
            low1 %= num_this_class;
            low2 %= num_this_class;
            high1 = low1 + num_this_part;
            high2 = low2 + num_this_part;
        }
        if (high2 >= num_this_class && low2 < num_this_class) {
            high1 = num_this_class;
            low2 = 0;
            high2 = low2 + num_this_part - (high1 - low1);
        }
        
        if ( (src->examples[i].bl_clump_num >= low1 && src->examples[i].bl_clump_num < high1) ||
             (src->examples[i].bl_clump_num >= low2 && src->examples[i].bl_clump_num < high2) ) {
            bal->examples[bal->meta.num_examples++] = src->examples[i];
            per_class_count[class]++;
        }
    }
    
    // Make sure the number of examples added is correct
    if (bal->meta.num_examples != ex_per_part)
        printf("ERROR: Added %d samples instead of %d\n", bal->meta.num_examples, ex_per_part);
    for (i = 0; i < src->meta.num_classes; i++)
        if (bal->meta.num_examples_per_class[i] != per_class_count[i])
            printf("ERROR: Added %d samples in class %d instead of %d\n", per_class_count[i], i, bal->meta.num_examples_per_class[i]);
}

/*
 *  assign_bl_clump_numbers
 *
 *  Assigns a random gid to each example, sorts on class, and then assigns a number from
 *  0 to N-1 for examples in each class. Basically, it assigns a 0-based class-count
 *  to each example.
 *
 *  get_next_balanced_set() uses these counts to assign examples to clumps.
 */
 
void assign_bl_clump_numbers(CV_Metadata meta, AV_SortedBlobArray examples, Args_Opts args) {
    int i;
    int *ran_id;
    int num_partitions;
    int num_examples_per;
    
    compute_number_of_clumps(meta, &args, &num_partitions, &num_examples_per);
    
    ran_id = (int *)malloc(examples.numBlob * sizeof(int));
    for (i = 0; i < examples.numBlob; i++)
        ran_id[i] = i;
    _knuth_shuffle(examples.numBlob, ran_id);
    for (i = 0; i < examples.numBlob; i++) {
        CV_Example *blob = (CV_Example *)examples.blobs[i];
        blob->random_gid = ran_id[i];
    }
    free(ran_id);
    qsort(examples.blobs, examples.numBlob, sizeof(CV_Example *), qsort_example_compare_by_class);
    
    int *class_count;
    int *num_per_part;
    class_count = (int *)calloc(meta.num_classes, sizeof(int));
    num_per_part = (int *)malloc(meta.num_classes * sizeof(int));
    for (i = 0; i < meta.num_classes; i++)
        num_per_part[i] = _num_per_class_per_clump(i, num_partitions, meta, args);
    
    for (i  = 0; i < examples.numBlob; i++) {
        CV_Example *blob = (CV_Example *)examples.blobs[i];
        int class = blob->containing_class_num;
        blob->bl_clump_num = class_count[class];
        class_count[class]++;
    }
}

void __print_bl_clumps(int clump_to_print, CV_Subset data, Args_Opts args) {
    int i, j, k;
    int num_clumps, ex_per_part, start_clump, end_clump;
    CV_Subset clump;
    if (clump_to_print < 0) {
        compute_number_of_clumps(data.meta, &args, &num_clumps, &ex_per_part);
        start_clump = 0;
        end_clump = 2 * num_clumps;
    } else {
        start_clump = clump_to_print;
        end_clump = clump_to_print;
    }
    printf("TOTAL OF %d CLUMPS\n", num_clumps);
    
    for (i = 0; i < 2*num_clumps; i++) {
        if (i < start_clump || i > end_clump)
            continue;
        get_next_balanced_set(i, &data, &clump, args);
        for (j = 0; j < clump.meta.num_examples; j++) {
            printf("CLUMP:%d:SAMPLE:%d:", i, clump.examples[j].fclib_id_num+1);
            for (k = 0; k < clump.meta.num_attributes; k++) {
                if (clump.meta.attribute_types[k] == CONTINUOUS) {
                    printf("%f,", clump.float_data[k][clump.examples[j].distinct_attribute_values[k]]);
                } else if (clump.meta.attribute_types[k] == DISCRETE) {
                    printf("%s,", clump.meta.discrete_attribute_map[k][clump.examples[j].distinct_attribute_values[k]]);
                }
            }
            printf("%s\n", clump.meta.class_names[clump.examples[j].containing_class_num]);
        }
    }
}

