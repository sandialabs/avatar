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
#include <stdio.h>
#include <string.h>
#include "../src/crossval.h"
#include "../src/evaluate.h"
#include "diversity_measures.h"

/*
    Computes a matrix holding per-tree results, truth values, and numbers of errors for each tree
    
    For T trees and N examples:
    matrix[0][n=0..N-1] holds the truth value for example n
    matrix[t=1..T][n=0..N-1] holds the result for tree t classifying example n
    matrix[t=1..T][N] holds the number of errors for tree t
 */

void init_matrix(DT_Ensemble ensemble, CV_Subset data, int ***matrix) {
    int i, j;
    int leaf_node;
    (*matrix) = (int **)malloc((ensemble.num_trees+1) * sizeof(int *));
    for (i = 0; i < ensemble.num_trees+1; i++)
        (*matrix)[i] = (int *)calloc(data.meta.num_examples+1, sizeof(int));
    
    for (j = 0; j < data.meta.num_examples; j++) {
        //printf("Ex %02d ", j+1);
        (*matrix)[0][j] = data.examples[j].containing_class_num;
        for (i = 0; i < ensemble.num_trees; i++) {
            (*matrix)[i+1][j] = classify_example(ensemble.Trees[i], data.examples[j], data.float_data, &leaf_node);
            //printf("%d ", matrix[i][j]);
            // Count number of errors for this tree
            if ((*matrix)[i+1][j] != (*matrix)[0][j])
                (*matrix)[i+1][data.meta.num_examples]++;
        }
        //printf("%d\n", data.examples[j].containing_class_num);
    }
}

double compute_pcdm(int num_trees, int num_examples, int **matrix) {
    int k, n;
    int tally = 0;
    for (k = 0; k < num_examples; k++) {
        int num_correct = 0;
        for (n = 1; n < num_trees+1; n++)
            if (matrix[n][k] == matrix[0][k])
                num_correct++;
        double percent_correct = 100.0 * (double)num_correct / (double)num_trees;
        if (percent_correct >= 10 && percent_correct <= 90)
            tally++;
    }
    
    return (double)tally/(double)num_examples;
}

double compute_interrater_kappa(int num_trees, int num_examples, int **matrix) {
    int k, n;
    int sum_lzj = 0;
    for (k = 0; k < num_examples; k++) {
        int num_correct = 0;
        for (n = 1; n < num_trees+1; n++)
            if (matrix[n][k] == matrix[0][k])
                num_correct++;
        sum_lzj += num_correct * (num_trees - num_correct);
        //printf("Ex %d has l(zj) = %d\n", k, sum_lzj);
    }
    int total_correct = 0;
    for (n = 1; n < num_trees+1; n++)
        total_correct += num_examples - matrix[n][num_examples];
    double avg_accuracy = (double)total_correct / (double)(num_trees * num_examples);
    //printf("Avg Acc = %f\n", avg_accuracy);
    
    return 1.0 - ((double)sum_lzj/(avg_accuracy*(1.0-avg_accuracy)*(double)(num_examples*num_trees*(num_trees-1))));
}

double compute_Q_statistic(int num_trees, int num_examples, int **matrix) {
    int k, m, n;
    double Q_sum = 0.0;
    
    for (n = 1; n < num_trees; n++) {
        for (m = n+1; m < num_trees+1; m++) {
            int N11 = 0;
            int N00 = 0;
            int N01 = 0;
            int N10 = 0;
            for (k = 0; k < num_examples; k++) {
                if (matrix[n][k] == matrix[0][k] && matrix[m][k] == matrix[0][k])
                    N11++;
                else if (matrix[n][k] != matrix[0][k] && matrix[m][k] != matrix[0][k])
                    N00++;
                else if (matrix[n][k] != matrix[0][k] && matrix[m][k] == matrix[0][k])
                    N01++;
                else if (matrix[n][k] == matrix[0][k] && matrix[m][k] != matrix[0][k])
                    N10++;
            }
            double Q = (double)(N11*N00 - N01*N10)/(double)(N11*N00 + N01*N10);
            //printf("For %d/%d: N11=%d N00=%d N01=%d N10=%d: Q=%f\n", n,m,N11,N00,N01,N10,Q);
            Q_sum += Q;
        }
    }
    
    return Q_sum / ((double)(num_trees*(num_trees-1))/2.0);
}

double compute_dietterich_kappa(int num_trees, int num_classes, int num_examples, int **matrix, Args_Opts args) {
    int i, j, k, m, n;
    
    FILE *fh;
    if (args.kappa_plot_data == TRUE) {
        char *kappa_data;
        kappa_data = (char *)malloc((strlen(args.data_path) + strlen(args.base_filestem) + 13) * sizeof(char));
        sprintf(kappa_data, "%s/%s_kappa.plot", args.data_path, args.base_filestem);
        fh = fopen(kappa_data, "w");
    }
    
    double Kappa_sum = 0.0;
    for (n = 1; n < num_trees; n++) {
        for (m = n+1; m < num_trees+1; m++) {
            int Cii = 0;
            int Cij = 0;
            int Cji = 0;
            double theta1 = 0.0;
            double theta2 = 0.0;
            for (i = 0; i < num_classes; i++) {
                for (k = 0; k < num_examples; k++)
                    if (matrix[n][k] == i && matrix[m][k] == i)
                        Cii++;
                for (j = 0; j < num_classes; j++) {
                    for (k = 0; k < num_examples; k++) {
                        if (matrix[n][k] == i && matrix[m][k] == j)
                            Cij++;
                        if (matrix[m][k] == i && matrix[n][k] == j)
                            Cji++;
                    }
                }
                //printf("C%d%d/C%d%d = %d/%d\n", n, m, m, n, Cij, Cji);
                theta2 += (double)(Cij*Cji)/(double)(num_examples*num_examples);
                //printf("theta2 = %f\n", theta2);
                Cij = 0;
                Cji = 0;
            }
            theta1 = (double)Cii/(double)num_examples;
            double this_Kappa = (theta1 - theta2)/(1.0 - theta2);
            if (args.kappa_plot_data == TRUE) {
                fprintf(fh, "%d %d %f %f\n", n, m, this_Kappa,
                                   (double)(matrix[n][num_examples]+matrix[m][num_examples])/(double)(2*num_examples));
            }
            Kappa_sum += this_Kappa;
        }
    }
    
    if (args.kappa_plot_data == TRUE)
        fclose(fh);
    
    return Kappa_sum / ( (double)(num_trees*(num_trees-1)) / 2.0 );
}
