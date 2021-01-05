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
#include "../src/crossval.h"
#include "util.h"
#include "../src/util.h"

int _check_xlicates(long **xlicates, int iterations, int pick, int outof, double *ff) {
    int i, j;
    int num_errors = 0;
    double truth;
    double M = pick;
    double N = outof;
    
    // We have a match if the values agree to the third decimal place.
    // This seems to work well for picking numbers out of 10e6 one time.
    // Picking numbers repeatedly and averaging will get closer but takes more time.
    // My assumption is that it will either be right or way off so this will catch the later.
    for (i = 0; i <= 4; i++) {
        truth = pow((N-1)/N, M-i) * pow(M/N, i+1) / (double)factorial(i);
        double truth_min = truth/ff[i];
        double truth_max = truth*ff[i];
        for (j = 0; j < iterations; j++) {
          //printf("Checking (%d) %.10f <= %.10f <= %.10f", i, truth_min, (double)xlicates[j][i]/N, truth_max);
            if ((double)xlicates[j][i]/N > truth_max || (double)xlicates[j][i]/N < truth_min) {
              //printf(" ERROR\n");
              num_errors++; 
            }
            else {
              //printf(" OK\n");
            }
        }
    }
    
    return num_errors;
}

int _check_class_xlicates(long **xlicates, int iterations, int pick, int outof) {
    int i, j;
    int num_errors = 0;
    double truth;
    double M = pick;
    double N = outof;
    // I'm picking for each clump, so I'm not picking from the total number of class samples,
    // I'm picking from the number of class samples in each clump. So, scale everything by
    // number of class samples in the clump
    double d = N*N/M;
    // Since I don't hit the mark spot on, allow for some fudging. The higher xlicates need
    // more fudge since there are fewer samples in the sample
    double fudge_factor[] = { 1.01, 1.01, 1.08, 1.08, 2 };

    // We have a match if the values agree to the third decimal place.
    // This seems to work well for picking numbers out of 10e6 one time.
    // Picking numbers repeatedly and averaging will get closer but takes more time.
    // My assumption is that it will either be right or way off so this will catch the later.
    for (i = 0; i <= 4; i++) {
        truth = pow((N-1)/N, M-i) * pow(M/N, i+1) / (double)factorial(i);
        // Another fudge: make the higher limit at least equivalent to an xlicate count of 3
        double truth_min = truth/fudge_factor[i];
        double truth_max = truth*fudge_factor[i];
        if (truth_max * d < 3.0)
            truth_max = 3.0/d;
        for (j = 0; j < iterations; j++) {
          //printf("Checking (i=%d) %.10f <= %f/%f=%f <= %.10f: ", i, truth_min, (double)xlicates[j][i], d, (double)xlicates[j][i]/d, truth_max);
            if ((double)xlicates[j][i]/d > truth_max || (double)xlicates[j][i]/d < truth_min) {
              //printf("WRONG\n");
                num_errors++;
            } else {
              //printf("OK\n");
            }
        }
    }
    
    return num_errors;
}

int _check_for_0_and_1(long **xlicates, int iterations, int pick, int outof) {
    int i, j;
    int num_errors = 0;
    double truth;
    double M = pick;
    double N = outof;
    
    for (i = 0; i <= 4; i++) {
        if (i == 0)
            truth = 1.0 - (M/N);
        else if (i == 1)
            truth = M/N;
        else
            truth = 0.0;
        for (j = 0; j < iterations; j++) {
            //printf("Comparing %f with %f for picking %d out %d\n", (double)xlicates[j][i]/N, truth, pick, outof);
            if ((double)xlicates[j][i]/N < truth-0.000001 || (double)xlicates[j][i]/N > truth+0.000001)
                num_errors++;
        }
    }
    
    return num_errors;
}

int _check_no_xlicates(long **xlicates, int iterations, int outof) {
    int i, j;
    int num_errors = 0;
    double truth;
    double N = outof;
    
    // We have a match if the values agree to the third decimal place.
    // This seems to work well for picking numbers out of 10e6 one time.
    // Picking numbers repeatedly and averaging will get closer but takes more time.
    // My assumption is that it will either be right or way off so this will catch the later.
    for (i = 0; i <= 4; i++) {
        truth = i==1?1.0:0.0;
        for (j = 0; j < iterations; j++) {
            //printf("Comparing %f with %f for picking %d out %d\n", (double)xlicates[j][i]/N, truth, pick, outof);
            if ((double)xlicates[j][i]/N < truth-0.000001 || (double)xlicates[j][i]/N > truth+0.000001)
                num_errors++;
        }
    }
    
    return num_errors;
}

void __print_xlicates(long *xlicates) {
    int i;
    printf("TUPLES 0-9:\n");
    for (i = 0; i <= 9; i++)
        printf("  %d: %ld (= %ld samples)\n", i, xlicates[i], xlicates[i]*i);
}

void __print_scaled_xlicates(long *xlicates, int num) {
    int i;
    printf("TUPLES 0-9:\n");
    for (i = 0; i <= 9; i++)
        printf("  %d: %8g\n", i, (double)xlicates[i]/(double)num);
}

void _count_xlicates(CV_Subset data, int size, long **xlicates) {
    long i;
    long *hits;

    if (size == 0)
        hits = (long *)calloc(data.meta.num_examples, sizeof(long));
    else
        hits = (long *)calloc(size, sizeof(long));
    
    *xlicates = (long *)calloc(10, sizeof(long));
    // Count number of times each example number shows up
    for (i = 0; i < data.meta.num_examples; i++)
        hits[data.examples[i].global_id_num]++;
    // Compute number of misses, single occurences, duplicates, triplicates, ...
    for (i = 0; i < data.meta.num_examples; i++)
        if (hits[i] <= 9)
            (*xlicates)[hits[i]]++;
    free(hits);
}

// Converts example number to a per-class example number which go from 0 to num_examples_per_class[]
// class_map[i][j] = n
// The jth example is the nth example in class i.
void _map_classes(CV_Subset data, int ***class_map) {
    int i, j;
    int *class_count;
    class_count = (int *)calloc(data.meta.num_classes, sizeof(int));
    (*class_map) = (int **)malloc(data.meta.num_classes * sizeof(int *));
    for (i = 0; i < data.meta.num_classes; i++) {
        (*class_map)[i] = (int *)malloc(data.meta.num_examples * sizeof(int));
        for (j = 0; j < data.meta.num_examples; j++)
            (*class_map)[i][j] = -1;
    }
    for (j = 0; j < data.meta.num_examples; j++) {
        int this_class = data.examples[j].containing_class_num;
        //printf("Setting class_map[%d][%d] to %d\n", this_class, j, class_count[this_class]);
        (*class_map)[this_class][j] = class_count[this_class]++;
    }
}

void _count_class_xlicates(CV_Subset data, CV_Subset clump, int class, int **class_map, long **class_xlicates) {
    long i;
    long *hits;
    
    hits = (long *)calloc(data.meta.num_examples_per_class[class], sizeof(long));
    *class_xlicates = (long *)calloc(10, sizeof(long));
    for (i = 0; i < clump.meta.num_examples; i++)
        if (clump.examples[i].containing_class_num == class)
            hits[class_map[class][clump.examples[i].global_id_num]]++;
    for (i = 0; i < data.meta.num_examples_per_class[class]; i++)
        if (hits[i] <= 9)
            (*class_xlicates)[hits[i]]++;
    free(hits);
}

