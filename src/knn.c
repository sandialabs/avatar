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
#include <math.h>
#include "crossval.h"
#include "knn.h"
#include "array.h"

void compute_knn(CV_Subset data, int k, int distance, Smote_Type s_type, Nearest_Neighbors ***knn) {
    int i, j, n;
    double d;
    
    //printf("Computing median of standard deviations\n");
    float median = compute_median_of_stdev(data);
    
    //printf("Computing %d-NN\n", k);
    //printf("Malloc knn to %d\n", data.meta.num_examples);
    (*knn) = (Nearest_Neighbors **)malloc(data.meta.num_examples * sizeof(Nearest_Neighbors *));
    for (i = 0; i < data.meta.num_examples; i++) {
        (*knn)[i] = (Nearest_Neighbors *)malloc(k * sizeof(Nearest_Neighbors));
        for (j = 0; j < k; j++) {
            (*knn)[i][j].neighbor = -1;
            // Initialize all distances to a large number
#if SIZEOF_DOUBLE >= 8
            (*knn)[i][j].distance = pow(2.0,8.0*8.0);
#else
            (*knn)[i][j].distance = pow(2.0,4.0*8.0);
#endif
        }
    }

    for (i = 0; i < data.meta.num_examples; i++) {
        for (j = i+1; j < data.meta.num_examples; j++) {
            
            // If we are doing closed SMOTE and the classes for i and j don't match, skip it
            if (s_type == CLOSED_SMOTE && data.examples[i].containing_class_num != data.examples[j].containing_class_num)
                continue;
            
            d = distance==1?L1_distance(data.examples[i], data.examples[j], data.meta, data.float_data, median)
                           :L2_distance(data.examples[i], data.examples[j], data.meta, data.float_data, median);
            if (d < (*knn)[i][k-1].distance) {
                // Example at j is in i's closest k for now
                // Find it's location and shift everyone else up
                n = k-1;
                while (n > 0 && d < (*knn)[i][n-1].distance) {
                    (*knn)[i][n].neighbor = (*knn)[i][n-1].neighbor;
                    (*knn)[i][n].distance = (*knn)[i][n-1].distance;
                    n--;
                }
                (*knn)[i][n].neighbor = j;
                (*knn)[i][n].distance = d;
            }
            
            if (d < (*knn)[j][k-1].distance) {
                // Example at i is in j's closest k for now
                // Find it's location and shift everyone else up
                n = k-1;
                while (n > 0 && d < (*knn)[j][n-1].distance) {
                    (*knn)[j][n].neighbor = (*knn)[j][n-1].neighbor;
                    (*knn)[j][n].distance = (*knn)[j][n-1].distance;
                    n--;
                }
                (*knn)[j][n].neighbor = i;
                (*knn)[j][n].distance = d;
            }

        }
    }
}

double L1_distance(CV_Example a, CV_Example b, CV_Metadata meta, float **float_data, float median) {
    double sum = 0.0;
    int i;
    for (i = 0; i < meta.num_attributes; i++) {
        if (meta.attribute_types[i] == CONTINUOUS) {
            sum += fabs(float_data[i][a.distinct_attribute_values[i]] - float_data[i][b.distinct_attribute_values[i]]);
        } else if (meta.attribute_types[i] == DISCRETE) {
            if (a.distinct_attribute_values[i] != b.distinct_attribute_values[i])
                sum += median;
        }
    }
    return sum;
}

double L2_distance(CV_Example a, CV_Example b, CV_Metadata meta, float **float_data, float median) {
    double sum = 0.0;
    int i;
    for (i = 0; i < meta.num_attributes; i++) {
        if (meta.attribute_types[i] == CONTINUOUS)
            sum += pow(float_data[i][a.distinct_attribute_values[i]] - float_data[i][b.distinct_attribute_values[i]], 2.0);
        else if (meta.attribute_types[i] == DISCRETE)
            if (a.distinct_attribute_values[i] != b.distinct_attribute_values[i])
                sum += median*median;
    }
    return sqrt(sum);
}

float compute_median_of_stdev(CV_Subset data) {
    int i, j;
    int cont_att;
    float *average;
    float *stdev;
    float return_val;
    
    average = (float *)calloc(data.meta.num_attributes, sizeof(float));
    stdev = (float *)calloc(data.meta.num_attributes, sizeof(float));
    
    // Sum each continuous feature for average
    for (i = 0; i < data.meta.num_examples; i++) {
        cont_att = 0;
        for (j = 0; j < data.meta.num_attributes; j++) {
            if (data.meta.attribute_types[j] == CONTINUOUS) {
                average[cont_att] += data.float_data[j][data.examples[i].distinct_attribute_values[j]];
                cont_att++;
            }
        }
    }
    // Average for each continuous feature
    cont_att = 0;
    for (j = 0; j < data.meta.num_attributes; j++)
        if (data.meta.attribute_types[j] == CONTINUOUS)
            average[cont_att++] /= data.meta.num_examples;
    // Sum of difference squared for stdev
    for (i = 0; i < data.meta.num_examples; i++) {
        cont_att = 0;
        for (j = 0; j < data.meta.num_attributes; j++) {
            if (data.meta.attribute_types[j] == CONTINUOUS) {
                stdev[cont_att] += pow(data.float_data[j][data.examples[i].distinct_attribute_values[j]] - average[cont_att], 2.0);
                cont_att++;
            }
        }
    }
    // Stdev for each continuous feature
    cont_att = 0;
    for (j = 0; j < data.meta.num_attributes; j++) {
        if (data.meta.attribute_types[j] == CONTINUOUS) {
            stdev[cont_att] = sqrt(stdev[cont_att]/(float)data.meta.num_examples);
            cont_att++;
        }
    }
    // Sort for median
    float_array_sort(cont_att, stdev-1);
    if (cont_att % 2 == 0)
        return_val = (stdev[(cont_att/2)-1] + stdev[cont_att/2])/2.0;
    else
        return_val = stdev[cont_att/2];
    
    free(average);
    free(stdev);
    
    return return_val;
}
