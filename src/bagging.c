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
#include "bagging.h"
#include "tree.h"
#include "util.h"
#include "balanced_learning.h"
#include "array.h"
#include "skew.h"
#include "av_rng.h"

void make_bag(CV_Subset *src, CV_Subset *bag, Args_Opts args, int cleanup) {
    int i, j, k;
    static int count = 0;
    static struct ParkMiller* rng = NULL;

    if (cleanup == 1)
    {
      if (rng != NULL)
      {
        free(rng);
        rng = NULL;
      }
      return;
    }
    
    //static int *samples_seen;
    //if (count == 0)
    //    samples_seen = (int *)calloc(src->meta.num_examples, sizeof(int));
    count++;
    if (args.debug)
        printf("count=%d\n", count);
    int num_in_bag, num_clumps;
    int *ex_per = NULL;
    int *current_ex_per = NULL;
    if (args.majority_bagging) {
        compute_number_of_clumps(src->meta, &args, &num_clumps, &num_in_bag);
        //if (count == 1)
        //    printf("Selecting %d for %d clumps\n", num_in_bag, num_clumps);
        ex_per = (int *)malloc(src->meta.num_classes * sizeof(int));
        current_ex_per = (int *)calloc(src->meta.num_classes, sizeof(int));
        for (i = 0; i < src->meta.num_classes; i++) {
            ex_per[i] = _num_per_class_per_clump(i, num_clumps, src->meta, args);
            //printf("Shooting for %d samples of class %d\n", ex_per[i], i);
        }
        //printf("For a total of %d samples\n", num_in_bag);
    } else {
        num_in_bag = (int)(args.bag_size * (float)src->meta.num_examples / 100.0);
        if (args.debug)
            printf("This bag will have %d samples\n", num_in_bag);
        if (num_in_bag <= 0 || num_in_bag > src->meta.num_examples) {
            fprintf(stderr, "ERROR: invalid number of examples in bag: %d\n", num_in_bag);
            exit(-8);
        }
    }
    
    copy_subset_meta(*src, bag, num_in_bag);
    copy_subset_data(*src, bag);
    bag->meta.num_examples = num_in_bag;
    // Re-initialize to false
    for (i = 0; i < src->meta.num_examples; i++)
        src->examples[i].in_bag = FALSE;
    
    // The first time through, initialize the RNG
    if (rng == NULL)
    {
      rng = malloc(sizeof(struct ParkMiller));
      av_pm_default_init(rng, args.random_seed + args.mpi_rank);
    }
    
    if (args.majority_bagging) {
        j = 0;
        
        for (i = 0; i < src->meta.num_examples; i++) {
            // If this is a minority class, include automatically
          if (find_int(src->examples[i].containing_class_num, args.num_minority_classes, args.minority_classes)) {
                //printf("Copy(1) %d to %d\n", i, j);
                copy_example_data(src->meta.num_attributes, src->examples[i], &(bag->examples[j]));
                //printf("MIN:%d %d\n", count, i);
                //samples_seen[i] = 1;
                src->examples[i].in_bag = TRUE;
                bag->meta.num_examples_per_class[src->examples[i].containing_class_num]++;
                j++;
                current_ex_per[src->examples[i].containing_class_num]++;
            }
        }
        find_int_release();
        // Bag the majority classes
        for (i = j; i < num_in_bag; i++) {
            //if (args.use_opendt_shuffle) {
            //    while (src->examples[(j=_lrand48(args) % src->meta.num_examples)].containing_class_num != best_class) {}
            //} else {
            //int this_class = src->examples[(j=lrand48() % src->meta.num_examples)].containing_class_num;
            //int this_class = src->examples[(j=gsl_rng_uniform_int(R, src->meta.num_examples))].containing_class_num;
            int this_class = src->examples[(j=av_pm_uniform_int(rng, src->meta.num_examples))].containing_class_num;
            while (current_ex_per[this_class] >= ex_per[this_class]) {
                //this_class = src->examples[(j=lrand48() % src->meta.num_examples)].containing_class_num;
                //this_class = src->examples[(j=gsl_rng_uniform_int(R, src->meta.num_examples))].containing_class_num;
                this_class = src->examples[(j=av_pm_uniform_int(rng, src->meta.num_examples))].containing_class_num;
            }
            current_ex_per[this_class]++;
            //}
            //printf("Copy(2) %d to %d\n", j, i);
            //printf("MAJ:%d %d\n", count, j);
            //samples_seen[j] = 1;
            copy_example_data(src->meta.num_attributes, src->examples[j], &(bag->examples[i]));
            src->examples[j].in_bag = TRUE;
            bag->meta.num_examples_per_class[this_class]++;
        }
        //int total = 0;
        //for (i = 0; i < src->meta.num_classes; i++) {
        //    printf("Actually have %d samples of class %d\n", current_ex_per[i], i);
        //    total += current_ex_per[i];
        //}
        //printf("For a total of %d samples\n", total);
        //int total_seen = 0;
        //for (i = 0; i < src->meta.num_examples; i++)
        //    if (samples_seen[i] == 1)
        //        total_seen++;
        //printf("After %3d iterations have seen %3d of %3d samples\n", count, total_seen, src->meta.num_examples);
        
    } else {
        //printf("modulus %d\n", src->meta.num_examples);
        for (i = 0; i < num_in_bag; i++) {
            //if (args.use_opendt_shuffle) {
            //    if (0 && args.debug) {
            //        for (k = 0; k < src->meta.num_attributes; k++)
            //            printf("  bag high/low for att %d: %d/%d\n", k, bag->high[k], bag->low[k]);
            //    }
            //    j = _lrand48(args) % src->meta.num_examples;
            //} else {
            //j = lrand48() % src->meta.num_examples;
            //j = gsl_rng_uniform_int(R, src->meta.num_examples);
            j = av_pm_uniform_int(rng, src->meta.num_examples);
            //}
            if (args.debug)
                printf("Picking data point:%8d\n", j);
            copy_example_data(src->meta.num_attributes, src->examples[j], &(bag->examples[i]));
            src->examples[j].in_bag = TRUE;

        }
    }
    if (args.debug) {
        for (k = 0; k < bag->meta.num_examples; k++)
            printf("Data for Bag:%d Att:0 = %10g\n",
                   k, bag->float_data[0][bag->examples[k].distinct_attribute_values[0]]);
    }

}
