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
#ifndef __PROX_UTILS__
#define __PROX_UTILS__

#include "../src/crossval.h"

typedef struct proximity_matrix_struct {
    int row;
    int col;
    float data;
    struct proximity_matrix_struct *next;
} Prox_Matrix;


void init_node_matrix(const DT_Ensemble* ensemble, 
                      const CV_Subset* data, 
                      int ***matrix);
void assemble_prox_matrix(int num_samples, 
                          int num_trees, 
                          int** n_matrix, 
                          Prox_Matrix **p_matrix, 
                          Boolean print_progress);
void compute_outlier_metric(int num_samples, 
                            int num_classes, 
                            int *class, 
                            Prox_Matrix *p_matrix, 
                            float **outliers,
                            Deviation_Type deviation_type);
void LL_unshift(Prox_Matrix **head_ref, int r, int c, float d);
void LL_push(Prox_Matrix **head_ref, int r, int c, float d);
void LL_reverse(Prox_Matrix *old_list, Prox_Matrix **new_list);
int LL_length(Prox_Matrix *head);
void init_outlier_file(CV_Metadata data, FC_Dataset *dataset, Args_Opts args);
int store_outlier_values(CV_Subset test_data, float *out, float *prox, int *samp, FC_Dataset dataset, Args_Opts args);
int write_proximity_matrix(Prox_Matrix *matrix, Args_Opts args);
void read_proximity_matrix(Prox_Matrix **matrix, Args_Opts args);


typedef struct landmark_struct Landmarks;

/*
 * Measure the remoteness, or outlierness, of each data point relative
 * to the given landmarks and save in the scores array.
 *
 * Remoteness is derived from Leo Breiman's forest proximity.  The
 * proximity of points i and j to each other in a tree ensemble is the
 * fraction of trees in which they land in the same leaf:
 *
 *    prox(i,j) = sum_t I[t.leaf(i) == t.leaf(j)]
 *                -------------------------------
 *                          # trees
 *
 * where I[ ] is the indicator function that returns 1 if true and 0
 * otherwise.  Breiman derived an outlier score, suitable for checking
 * for mislabeled data, from this as follows.  The outlierness of
 * point i, with respect to the other points j in the same class, is:
 *
 *    rawOUT(i) = 1 / (sum_j prox(i,j)^2)
 *
 * The raw outlier scores are then normalized to be comparable across
 * classes.  
 *
 * Here, the points i in data are compared to points j in landmarks
 * without knowledge of labels for points i.  Multiple outlier scores
 * for i are computed, as if it belonged to each of the possible
 * classes c.  Remoteness of i is:
 *
 *    remoteness(i) = min_c outlier(i | label(i) = c)
 *
 * Pre: length(scores) >= data->meta.num_examples
 */
void 
measure_remoteness(
    const DT_Ensemble* ensemble, 
    const Landmarks* landmarks,
    const CV_Subset* data,
    float* scores);

Landmarks* create_landmarks(
    const DT_Ensemble* ensemble, 
    const CV_Subset* data,
    Boolean print_progress);
void free_landmarks(Landmarks* landmarks);

#endif // __PROX_UTILS__
