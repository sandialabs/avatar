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
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../src/crossval.h"
#include "../src/evaluate.h"
#include "../src/rw_data.h"
#include "../src/array.h"
#include "proximity_utils.h"
#include "../src/safe_memory.h"
#include "../src/datatypes.h"
#include "../src/av_rng.h"
#include "../src/av_stats.h"

// This is typedef'd to Landmarks in header file.
struct landmark_struct {
    uint num_examples;
    uint num_trees;
    // Dense table storing which leaf each reference data point landed in.
    // One row per reference point, and one column per tree.
    // I.e., leaf_nodes[i][t] retrieves leaf for point i in tree t.
    int** leaf_nodes;
    // Class labels for each reference point.  Taken from training labels.
    int* labels;
    // We need the median and deviation for the raw outlier scores, for each class.
    uint num_classes;
    double* median;
    double* deviation;
}; // Landmarks;

// Prototypes for helper functions.
void _find_leaves(const DT_Ensemble* ensemble, const CV_Subset* data, uint i, int* leaves);
double _proximity(uint num_trees, const int* leaves_A, const int* leaves_B);
double* _raw_outlier_scores(uint num_samples, const int* labels, const Prox_Matrix* p_matrix);
void _score_pop_stats(
    uint N, 
    const double* scores, 
    uint L, 
    const int* labels,
    Deviation_Type devtype,
    double** _centroids,
    double** _deviation);
double _standardize(double raw, double centroid, double dev);
void _print_landmarks(FILE* fh, const Landmarks* landmarks);
void _print_nodeIDs(FILE* fh, uint num_trees, const int* nodes);


/*
    Computes a matrix holding the leaf nodes for each sample and tree
 */
void init_node_matrix(const DT_Ensemble* ensemble, 
                      const CV_Subset* data, 
                      int ***matrix) 
{
    uint i = 0;
    const uint num_examples = data->meta.num_examples;
    const uint num_trees = ensemble->num_trees;
    
    // Allocate matrix of leaf node IDs.
    int** nodes = e_calloc(num_examples, sizeof(int*));
    for (i = 0; i < num_examples; ++i) {
        nodes[i] = e_calloc(num_trees, sizeof(int));
    }

    // For each example data point...
    for (i = 0; i < num_examples; ++i) {
        // ... store the leaves it lands in.
        _find_leaves(ensemble, data, i, nodes[i]);
    }

    *matrix = nodes;
}

/*
    Computes the proximity matrix as a linked list.
    The proximity is defined as the number of trees for which two samples end up in the same leaf node
    normalized by the number of trees
 */
void assemble_prox_matrix(int num_samples, 
                          int num_trees, 
                          int** n_matrix, 
                          Prox_Matrix **p_matrix, 
                          Boolean print_progress) 
{
    int i, j;
    int num_pushes = 0;
    int this_percent = 0;
    int last_percent = -1;
    
    if (print_progress == TRUE) {
        fprintf(stderr, "Proximity Matrix computation: 000%%");
        fflush(NULL);
    }
    for (i = 0; i < num_samples; i++) {
        this_percent = 100*(i+1)/num_samples;
        if (print_progress == TRUE && this_percent != last_percent) {
            fprintf(stderr, "\b\b\b\b%3d%%", this_percent);
            fflush(NULL);
            last_percent = this_percent;
        }
        num_pushes = 0;
        for (j = i; j < num_samples; j++) {
            // Measure forest proximity of i and j.
            float prox = (float)_proximity(num_trees, n_matrix[i], n_matrix[j]);
            // Only store pairs for which there was at least one common leaf node
            if (prox > 0) {
                num_pushes++;
                LL_unshift(p_matrix, i, j, prox);
            }
        }
    }
    
    if (print_progress == TRUE) {
        int total = (2 * (LL_length(*p_matrix) - num_samples)) + num_samples;
        fprintf(stderr, "\nThe proximity matrix density is %.2f%%\n", 100.0 * (float)total / (float)(num_samples*num_samples));
    }
}

/*
   Computes the outlier metric for the specified sample
 */
void compute_outlier_metric(int num_samples, 
                            int num_classes, 
                            int *class, 
                            Prox_Matrix *p_matrix, 
                            float **outliers,
                            Deviation_Type deviation_type)
{
    uint c;
    int i;

    double* scores = _raw_outlier_scores((uint)num_samples, class, p_matrix);

    double* centroids = NULL;
    double* deviation = NULL;
    _score_pop_stats(
        (uint)num_samples, 
        scores, 
        (uint)num_classes, 
        class, 
        deviation_type,
        &centroids,
        &deviation);

    // Standardize with median and standard deviation.
    *outliers = e_calloc(num_samples, sizeof(float));
    for (i = 0; i < num_samples; i++) {
        c = class[i];
        (*outliers)[i] = (float)_standardize(scores[i], centroids[c], deviation[c]);
    }

    // Free scratch space.
    free(deviation);
    free(centroids);
    free(scores);
}

/*
   Unshifts a node onto the beginning of the linked list
 */
void LL_unshift(Prox_Matrix **head_ref, int r, int c, float d) {
    Prox_Matrix *new_node;
    // Create new node
    new_node = (Prox_Matrix *)malloc(sizeof(Prox_Matrix));
    new_node->row = r;
    new_node->col = c;
    new_node->data = d;
    // Since we're adding to the front, next will be the current head node
    new_node->next = *head_ref;
    // And the new head node will be the one we just created
    *head_ref = new_node;
}

/*
   Pushes a node onto the end of the linked list
 */
void LL_push(Prox_Matrix **head_ref, int r, int c, float d) {
    Prox_Matrix *current = *head_ref;
    Prox_Matrix *new_node;
    // Create new node
    new_node = (Prox_Matrix *)malloc(sizeof(Prox_Matrix));
    new_node->row = r;
    new_node->col = c;
    new_node->data = d;
    // Since we're pushing onto the end, next will be NULL
    new_node->next = NULL;
    
    if (current == NULL) {
        // Special case for length 0
        *head_ref = new_node;
    } else {
        // Find the end of the list ...
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_node;
    }
}

/*
   Reverse a linked list
   Traverses a list and appends each node to the beginning of a new list
 */
void LL_reverse(Prox_Matrix *old_list, Prox_Matrix **new_list) {
    Prox_Matrix *current = old_list;
    while (current != NULL) {
        LL_unshift(new_list, current->row, current->col, current->data);
        current = current->next;
    }
}

/*
   Returns the number of nodes in the linked list
 */
int LL_length(Prox_Matrix *head) {
    Prox_Matrix *current;
    current = head;
    int count = 0;
    while (current != NULL) {
        count++;
        current = current->next;
    }
    return count;
}

/*
   Initialize the exodus file which will contain the original data plus the 
   outlier and, possibly, the proximity values
 */
void init_outlier_file(CV_Metadata data, FC_Dataset *dataset, Args_Opts args) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j;
    int num_steps;
    char *mesh_name;
    FC_Mesh *meshes;
    FC_Sequence seq;
    FC_Variable ***vars; // vars[0][i][0] is for outlier
                         // vars[1][i][0] is for proximity
    
    // Create dataset
    if (args.sort_line_num >= 0) {
        fc_exitIfErrorPrintf(fc_createDataset("Outlier and Proximity", dataset), "Failed to create Outlier dataset\n");
    } else {
        fc_exitIfErrorPrintf(fc_createDataset("Outlier", dataset), "Failed to create Outlier dataset\n");
    }
    // Malloc meshes
    meshes = (FC_Mesh *)malloc(data.exo_data.num_seq_meshes * sizeof(FC_Mesh));
    // Create 1-step sequence
    fc_exitIfErrorPrintf(fc_createSequence(*dataset, "TestingTimesteps", &seq), "Failed to create new sequence\n");
    double *seq_coords;
    seq_coords = (double *)malloc(args.num_test_times * sizeof(double));
    for (i = 0; i < args.num_test_times; i++)
        seq_coords[i] = (double)i;
    fc_exitIfErrorPrintf(fc_setSequenceCoords(seq, args.num_test_times, FC_DT_DOUBLE, seq_coords),
                         "Failed to create sequence\n");
    free(seq_coords);
    // Malloc vars
    int num_vars = 2;
    vars = (FC_Variable ***)malloc(num_vars * sizeof(FC_Variable **));
    for (i = 0; i < num_vars; i++)
        vars[i] = (FC_Variable **)malloc(data.exo_data.num_seq_meshes * sizeof(FC_Variable*));

    for (i = 0; i < data.exo_data.num_seq_meshes; i++) {
        // Get name of mesh
        if (fc_getMeshName(data.exo_data.seq_meshes[i], &mesh_name) != FC_SUCCESS) {
            mesh_name = (char *)malloc(1024 * sizeof(char));
            sprintf(mesh_name, "Mesh%06d", i);
        }
        // Create new mesh with same name
        fc_exitIfErrorPrintf(fc_createMesh(*dataset, mesh_name, &meshes[i]),
                             "Failed to create mesh %d for Outlier\n", i);
        free(mesh_name);
        // Set coordinates
        int dim, vert, elems;
        double *coords;
        FC_ElementType elem_type;
        int *conns;
        fc_exitIfErrorPrintf(fc_getMeshDim(data.exo_data.seq_meshes[i], &dim),
                             "Failed to get mesh dim %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_getMeshNumVertex(data.exo_data.seq_meshes[i], &vert),
                             "Failed to get vertices %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_getMeshCoordsPtr(data.exo_data.seq_meshes[i], &coords),
                             "Failed to get mesh coords %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_setMeshCoords(meshes[i], dim, vert, coords),
                             "Failed to set mesh coords %d for Outlier\n", i);

        // Set element connectivities
        fc_exitIfErrorPrintf(fc_getMeshNumElement(data.exo_data.seq_meshes[i], &elems),
                             "Failed to get num elements %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_getMeshElementType(data.exo_data.seq_meshes[i], &elem_type),
                             "Failed to get element type %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_getMeshElementConnsPtr(data.exo_data.seq_meshes[i], &conns),
                             "Failed to get mesh conns %d for Outlier\n", i);
        fc_exitIfErrorPrintf(fc_setMeshElementConns(meshes[i], elem_type, elems, conns),
                             "Failed to set mesh conns %d for Outlier\n", i);
        
        // Create variables for fold_number, truth, and prediction
        fc_exitIfErrorPrintf(fc_createSeqVariable(meshes[i], seq, "Outlier Measure", &num_steps, &vars[0][i]),
                             "Failed to create outlier_measure var for mesh %d\n", i);
        // Create variables for probabilities if requested
        if (args.sort_line_num >= 0) {
            fc_exitIfErrorPrintf(fc_createSeqVariable(meshes[i], seq, "Proximity", &num_steps,
                                 &vars[1][i]), "Failed to create proximity var for mesh %d\n", i);
        }
        
        // Initialize everything to -1
        double *init;
        int num_data_pts = data.global_offset[i+1] - data.global_offset[i];
        init = (double *)malloc(num_data_pts * sizeof(double));
        for (j = 0; j < num_data_pts; j++)
            init[j] = -1.0;
        for (j = 0; j < args.num_test_times; j++) {
            fc_exitIfErrorPrintf(fc_setVariableData(vars[0][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                    FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                    "Failed to initialize outlier_measure data %d for Outlier\n", i);
            if (args.sort_line_num >= 0) {
                fc_exitIfErrorPrintf(fc_setVariableData(vars[1][i][j], num_data_pts, 1, data.exo_data.assoc_type,
                                                        FC_MT_SCALAR, FC_DT_DOUBLE, init),
                                                        "Failed to initialize proximity data %d for Outlier\n", i);
            }
        }
        free(init);

    }
    fc_exitIfErrorPrintf(fc_writeDataset(*dataset, args.prox_sorted_file, FC_FT_EXODUS),
                                         "Failed to write the outlier exodus file\n");
    #else
    av_missingFCLIB();
    #endif
}

/*
   Writes the outlier measures and, possibly, the proximity values to the
   already-initialized exodus file
 */
int store_outlier_values(CV_Subset test_data, float *out, float *prox, int *samp,  FC_Dataset dataset, Args_Opts args) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j;
    FC_Variable ***vars; // vars[0][i][j] is for outlier
                         // vars[1][i][j] is for proximity
    FC_Mesh *meshes;
    int num_meshes, num_vars;
    double **data;
    
    // Get the meshes from the dataset
    fc_exitIfErrorPrintf(fc_getMeshes(dataset, &num_meshes, &meshes), "Failed to get meshes from Outlier dataset");
    if (num_meshes != test_data.meta.exo_data.num_seq_meshes) {
        fprintf(stderr, "Expected %d but got %d meshes from Outlier dataset\n",
                        test_data.meta.exo_data.num_seq_meshes, num_meshes);
        return 0;
    }
    
    // Malloc vars
    num_vars = 2;
    vars = (FC_Variable ***)malloc(num_vars * sizeof(FC_Variable **));
    for (i = 0; i < num_vars; i++)
        vars[i] = (FC_Variable **)malloc(test_data.meta.exo_data.num_seq_meshes * sizeof(FC_Variable*));
    
    for (i = 0; i < test_data.meta.exo_data.num_seq_meshes; i++) {
        int num_data_pts = test_data.meta.global_offset[i+1]-test_data.meta.global_offset[i];
        //int global_id_base = fclib2global(i, 0, test_data.num_fclib_seq, test_data.meta.global_offset);
        //int mesh = i % test_data.meta.exo_data.num_seq_meshes;
        //int timestep = i / test_data.meta.exo_data.num_seq_meshes;
        
        data = (double **)malloc(args.num_test_times * sizeof(double *));        
        for (j = 0; j < args.num_test_times; j++)
            data[j] = (double *)malloc(num_data_pts * sizeof(double));
        
        // Update outlier var
        fc_exitIfErrorPrintf(_copy_and_delete_orig(meshes[i], "Outlier Measure", &vars[0][i], &data),
                             "Error copying outlier_measure variable");
        for (j = 0; j < test_data.meta.num_examples; j++) {
            if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                // This fclib sequence corresponds to the exodus mesh we are working on
                
                // Get the timestep that this fclib sequence corresponds to
                int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                //printf("Setting outlier value for %d to %f\n", test_data.examples[j].fclib_id_num, out[test_data.examples[j].global_id_num]);
                data[timestep][test_data.examples[j].fclib_id_num] = out[test_data.examples[j].global_id_num];
            }
        }
        for (j = 0; j < args.num_test_times; j++)
            fc_exitIfErrorPrintf(fc_setVariableData(vars[0][i][j], num_data_pts, 1, test_data.meta.exo_data.assoc_type,
                                            FC_MT_SCALAR, FC_DT_DOUBLE, (void *)data[j]),
                                            "Failed to set outlier_measure data for timestep %d on mesh %d\n", j, i);
        
        // Update proximity var
        if (args.sort_line_num >= 0) {
            // Init all proximities to -1 so we only need to update those with valid values
            for (j = 0; j < test_data.meta.num_examples; j++) {
                if (test_data.examples[j].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                    int timestep = test_data.examples[j].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                    data[timestep][test_data.examples[j].fclib_id_num] = -1.0;
                }
            }
            
            fc_exitIfErrorPrintf(_copy_and_delete_orig(meshes[i], "Proximity", &vars[1][i], &data),
                                                      "Error copying proximity variable");
            for (j = 0; j < test_data.meta.num_examples; j++) {
                int sample = samp[j];
                if (test_data.examples[sample].fclib_seq_num % test_data.meta.exo_data.num_seq_meshes == i) {
                    // Get the timestep that this fclib sequence corresponds to
                    int timestep = test_data.examples[sample].fclib_seq_num / test_data.meta.exo_data.num_seq_meshes;
                    if (sample == args.sort_line_num) {
                        data[timestep][test_data.examples[sample].fclib_id_num] = 1.0;
                    } else if (sample > -1) {
                        data[timestep][test_data.examples[sample].fclib_id_num] = prox[j];
                    }
                    // If sample == -1, then the proximity is -1 and the variable has already been
                    // initialized to -1 do nothing needs to be set
                }
            }
            for (j = 0; j < args.num_test_times; j++)
                fc_exitIfErrorPrintf(fc_setVariableData(vars[1][i][j], num_data_pts, 1,
                                                 test_data.meta.exo_data.assoc_type, FC_MT_SCALAR,
                                                 FC_DT_DOUBLE, (void *)data[j]),
                                                 "Failed to set proximity data for timestep %d on mesh %d\n", j, i);
        }
        
        for (j = 0; j < args.num_test_times; j++)
            free(data[j]);
        free(data);
        
    }
    fc_exitIfErrorPrintf(fc_rewriteDataset(dataset, args.prox_sorted_file, FC_FT_EXODUS),
                                           "Failed to write the predictions exodus file: %s\n", args.prox_sorted_file);
    
    return 1;
    #else
    av_missingFCLIB();
    return 0;
    #endif
}

int write_proximity_matrix(Prox_Matrix *matrix, Args_Opts args) {
    Prox_Matrix *current = matrix;
    FILE *pm;
    if ((pm = fopen(args.prox_matrix_file, "w")) == NULL) {
        fprintf(stderr, "ERROR: Could not write the proximity matrix file\n");
        return 0;
    }
    int count = 0;
    while (current != NULL) {
        count++;
        fprintf(pm, "%d,%d,%f\n", current->row, current->col, current->data);
        current = current->next;
    }
    fclose(pm);
    return count;
}

void read_proximity_matrix(Prox_Matrix **matrix, Args_Opts args) {
    int i;
    FILE *pm;
    unsigned int max = 128;
    char strbuf[max];
    Boolean last_line_partial = FALSE;
    char *data_line = NULL;
    int num_nodes = 0;
    Prox_Matrix *temp = NULL;
    
    if ((pm = fopen(args.prox_matrix_file, "r")) == NULL) {
        fprintf(stderr, "ERROR: Could not read the proximity matrix file '%s'\n", args.prox_matrix_file);
        exit(-1);
    }
    while (fgets(strbuf, max, pm) != NULL) {
        // If the last line is partial, append strbuf to it
        if (last_line_partial == TRUE) {
            data_line = (char *)realloc(data_line, (strlen(data_line)+strlen(strbuf)+1)*sizeof(char));
            strcat(data_line, strbuf);
        }
        else if (last_line_partial == FALSE) {
            // If the last line is complete, remove \n, parse and add to linked list
            if (num_nodes > 0) {
                data_line[strlen(data_line)-1] = '\0';
                int num_tokens;
                char **tokens;
                parse_delimited_string(',', data_line, &num_tokens, &tokens);
                if (num_tokens != 3)
                    fprintf(stderr, "Line %d does not have the correct number of entries ... skipping\n", num_nodes+1);
                else
                    LL_unshift(&temp, atoi(tokens[0]), atoi(tokens[1]), atof(tokens[2]));
                for (i = 0; i < num_tokens; i++)
                    free(tokens[i]);
                free(tokens);
            }
            num_nodes++;
            // Start over
            free(data_line);
            data_line = strdup(strbuf);
        }
        // Set last_line_parital
        if (strbuf[strlen(strbuf)-1] == '\n')
            last_line_partial = FALSE;
        else
            last_line_partial = TRUE;
    }
    // Handle last line
    {
        int num_tokens;
        char **tokens;
        parse_delimited_string(',', data_line, &num_tokens, &tokens);
        if (num_tokens != 3)
            fprintf(stderr, "Line %d does not have the correct number of entries ... skipping\n", num_nodes+1);
        else
            LL_unshift(&temp, atoi(tokens[0]), atoi(tokens[1]), atof(tokens[2]));
    }
    LL_reverse(temp, matrix);
}


void 
measure_remoteness(
    const DT_Ensemble* ensemble, 
    const Landmarks* landmarks,
    const CV_Subset* data,
    float* scores)
{
    assert((uint)(ensemble->num_trees) == landmarks->num_trees);
    const uint num_landmarks = landmarks->num_examples;
    const uint num_trees = landmarks->num_trees;
    const uint num_points = data->meta.num_examples;
    const uint num_classes = ensemble->num_classes;
    const Boolean verbose = FALSE;

    int* leaves = e_calloc(num_trees, sizeof(int));
    double* class_score = e_calloc(num_classes, sizeof(double));

    uint i, j;

    // Measure remoteness for each data point i...
    for (i = 0; i < num_points; ++i) {
        // Drop point i into ensemble, and record the leaves it lands in.
        _find_leaves(ensemble, data, i, leaves);
        if (verbose) {
            fprintf(stdout, "TEST %d: ", i);
            _print_nodeIDs(stdout, num_trees, leaves);
        }

        // Compute sum of squared proximity scores, for i to all classes.
        memset(class_score, 0, sizeof(class_score[0]) * num_classes);
        for (j = 0; j < num_landmarks; ++j) {
            // ... compute prox(i, j)
            double prox = _proximity(num_trees, leaves, landmarks->leaf_nodes[j]);
            if (verbose && j == 0) {
                _print_nodeIDs(stdout, num_trees, landmarks->leaf_nodes[j]);
                fprintf(stdout,    "PROX=%f\n", prox);
            }

            // ... and update accum. squared prox of i to j's class
            class_score[ landmarks->labels[j] ] += prox*prox;
        }

        if (verbose) {
            fprintf(stdout, "** SUM_PROX =");
            for (j = 0; j < num_classes; ++j) {
                fprintf(stdout, " %f", class_score[j]);
            }
            fprintf(stdout, "; TRUTH=%d\n", data->examples[i].containing_class_num);
        }

        // Derive raw outlier scores from the accum. proximities.
        for (j = 0; j < num_classes; ++j) {
            if (class_score[j] > 0) {
                class_score[j] = 1.0 / class_score[j];
            }
            else {
                class_score[j] = INFINITY;
            }
        }
        if (verbose) {
            fprintf(stdout, "** RAW_OUT =");
            for (j = 0; j < num_classes; ++j) {
                fprintf(stdout, " %f", class_score[j]);
            }
            fprintf(stdout, "; TRUTH=%d\n", data->examples[i].containing_class_num);
        }

        // Normalize outlier scores.
        for (j = 0; j < num_classes; ++j) {
            class_score[j] = _standardize(class_score[j], landmarks->median[j], landmarks->deviation[j]);
        }
        if (verbose) {
            fprintf(stdout, "** OUT =");
            for (j = 0; j < num_classes; ++j) {
                fprintf(stdout, " %f", class_score[j]);
            }
            fprintf(stdout, "; TRUTH=%d\n", data->examples[i].containing_class_num);
        }


        // Return the smallest outlier score as the remoteness score.
        scores[i] = INFINITY;
        for (j = 0; j < num_classes; ++j) {
            if (class_score[j] < scores[i]) {
                scores[i] = (float)(class_score[j]);
            }
        }

        // Truncate remoteness scores to be 0, at bottom end.
        if (scores[i] < 0) {
            scores[i] = 0;
        }
    }

    free(class_score);
    free(leaves);
}

Landmarks* 
create_landmarks(
    const DT_Ensemble* ensemble, 
    const CV_Subset* data,
    Boolean print_progress)
{
    uint i = 0;
    Landmarks* landmarks = e_calloc(1, sizeof(Landmarks));
    const uint num_examples = landmarks->num_examples = data->meta.num_examples;
    const uint num_trees = landmarks->num_trees = ensemble->num_trees;
    const uint num_classes = landmarks->num_classes = ensemble->num_classes;
    
    // Build database of how reference points fall into tree leaves.
    int** nodes = NULL;
    init_node_matrix(ensemble, data, &nodes);
    landmarks->leaf_nodes = nodes;

    // Record ground truth label for each example.
    int* labels = e_calloc(num_examples, sizeof(int));
    landmarks->labels = labels;
    for (i = 0; i < num_examples; ++i) {
        labels[i] = data->examples[i].containing_class_num;
    }

    // Fill in the median and deviation value for each class.
    // This is based on the distribution of raw outlier scores of
    // the reference data.  Thus, we need the proximity matrix of
    // the reference data.
    
    Prox_Matrix* p_matrix = NULL;
    assemble_prox_matrix((int)num_examples, (int)num_trees, nodes, &p_matrix, print_progress);
    
    double* scores = _raw_outlier_scores(num_examples, landmarks->labels, p_matrix);

    _score_pop_stats(
        num_examples,
        scores,
        num_classes,
        landmarks->labels,
        ABSOLUTE_DEVIATION,
        &(landmarks->median),
        &(landmarks->deviation));

    // Clean up temporary variables.
    free(scores);

    //_print_landmarks(stdout, landmarks);
    return landmarks;
}

void 
free_landmarks(Landmarks* landmarks)
{
    if (landmarks != NULL) {
        free(landmarks->labels);  landmarks->labels = NULL;
        uint i;
        for (i = 0; i < landmarks->num_examples; ++i) {
            free(landmarks->leaf_nodes[i]);
            landmarks->leaf_nodes[i] = NULL;
        }
        free(landmarks->leaf_nodes);  landmarks->leaf_nodes = NULL;
        landmarks->num_trees = 0;
        landmarks->num_examples = 0;
        
        landmarks->num_classes = 0;
        free(landmarks->median);  landmarks->median = NULL;
        free(landmarks->deviation);  landmarks->deviation = NULL;
    }
}


//////////////////////////// PRIVATE HELPER FUNCTIONS ////////////////////////////

void _find_leaves(const DT_Ensemble* ensemble, const CV_Subset* data, uint i, int* leaves)
{
    const uint num_trees = ensemble->num_trees;
    uint t;
    for (t = 0; t < num_trees; ++t) {
        // ... classify example and ID of leaf example lands in.
        classify_example(
            ensemble->Trees[t], 
            data->examples[i], 
            data->float_data,
            &(leaves[t]));
    }
}

/*
 * Compute forest proximity between two points A and B.
 *
 * Pre: leaves_A and leaves_B list the leaf IDs that A and B fall
 *      into, respectively.
 */
double _proximity(uint num_trees, const int* leaves_A, const int* leaves_B)
{
    uint t;
    uint overlap = 0;
    for (t = 0; t < num_trees; ++t) {
        if (leaves_A[t] == leaves_B[t]) {
            overlap += 1;
        }
    }
    return (double)overlap / (double)num_trees;
}

/*
 * Compute unnormalized outlier scores for a set of data points, based
 * on their pair-wise forest proximity to each other.
 *
 * N.B. Caller is responsible for calling free() on the returned scores.
 */
double* 
_raw_outlier_scores(
    uint num_samples, 
    const int* labels,
    const Prox_Matrix* p_matrix)
{
    uint i;
    double* scores = e_calloc(num_samples, sizeof(double));

    // Step through the proximity matrix and handle each pair of samples
    const Prox_Matrix* current;
    current = p_matrix;
    while (current != NULL) {
        // Skip diagonal elements
        if (current->row != current->col) {
            // Check that both row and col samples have same class
            if (labels[current->row] == labels[current->col]) {
                // Update outliers vector for row and col
                scores[current->row] += current->data * current->data;
                scores[current->col] += current->data * current->data;
            }
        }
        current = current->next;
    }

    // Convert scores from proximity to outliers.
    for (i = 0; i < num_samples; ++i) {
        scores[i] = 1.0 / scores[i];
    }

    return scores;
}

/*
 * Compute statistics for the per class raw outlier scores.
 *
 * Arguments:
 *   N --- total number of values in scores
 *   scores --- Array of raw outlier scores.
 *   L --- Number of possible classes.
 *   labels --- Array of N labels, corresponding to the N scores.
 *   devtype --- type of deviation (around centroid) to compute
 *   centroids --- Returned per-class centroids.
 *   deviation --- Returned per-class deviations.
 *
 * N.B. Caller responsible for calling free() on *centroids and *deviation.
 */
void
_score_pop_stats(
    uint N, 
    const double* scores, 
    uint L, 
    const int* labels,
    Deviation_Type devtype,
    double** _centroids,
    double** _deviation)
{
    uint i, c;
    double* centroids = e_calloc(L, sizeof(double));
    double* deviation = e_calloc(L, sizeof(double));

    //
    // For each class, we need a vector of raw outlier scores.
    //

    // First figure out number of scores per class.
    uint* counts = e_calloc(L, sizeof(uint));
    for (i = 0; i < N; ++i) {
        counts[ labels[i] ] += 1;
    }

    // Next, copy scores into per class vectors.
    double** per_class_outliers = e_calloc(L, sizeof(double*));
    for (c = 0; c < L; ++c) {
        per_class_outliers[c] = e_calloc(counts[c], sizeof(double));
    }
    memset(counts, 0, sizeof(uint) * L);
    for (i = 0; i < N; ++i) {
        c = labels[i];
        per_class_outliers[c][ counts[c] ] = scores[i];
        counts[c] += 1;
    }

    //
    // Compute centroids of scores in each class.
    // 
    for (c = 0; c < L; ++c) {
        if (counts[c] > 0) {
            qsort(per_class_outliers[c],
                  sizeof(per_class_outliers[c])/sizeof(*per_class_outliers[c]),
                  sizeof(*per_class_outliers[c]),
                  av_stats_comp_double);
            centroids[c] = av_stats_median_from_sorted_data(per_class_outliers[c], counts[c]);
        }
        else {
            centroids[c] = 0.0;
        }
    }

    //
    // Compute per-class deviations.
    //
    for (c = 0; c < L; ++c) {
        if (counts[c] > 0) {
            switch (devtype) {
            case STANDARD_DEVIATION:
                // REVIEW-2012-04-09-ArtM: This finds std dev around mean, but we use the median to center data.
                // Weird.  Do we want std dev. around the median instead, or should we return the mean
                // for centering data?
                deviation[c] = av_stats_sd(per_class_outliers[c], counts[c]);
                break;
            case ABSOLUTE_DEVIATION:
                // REVIEW-2012-04-09-ArtM: This finds abs dev around mean, but:
                // A) we use the median to center data, and
                // B) Breiman's formulation is abs dev around the median.
                // So this should probably be changed.  Leaving for now for backwards compatibility.
                deviation[c] = av_stats_median_from_sorted_data(per_class_outliers[c], counts[c]);
                break;
            default:
                fprintf(stderr, "error: unexpected deviation type in _score_pop_stats()\n");
                break;
            };
        }
        else {
            deviation[c] = 0.0;
        }
    }


    // Clean up scratch space.
    for (c = 0; c < L; ++c) {
        free(per_class_outliers[c]);
        per_class_outliers[c] = NULL;
    }
    free(per_class_outliers);
    free(counts);

    // Return computed statistics.
    *_centroids = centroids;
    *_deviation = deviation;
}

double 
_standardize(double raw, double centroid, double dev)
{
    return (raw - centroid) / dev;
}

void _print_landmarks(FILE* fh, const Landmarks* landmarks)
{
    const uint num_rows = landmarks->num_examples;
    const uint num_cols = landmarks->num_trees;
    uint i;

    assert(num_cols > 0 || num_rows == 0);
    fprintf(fh, "********** LANDMARK TABLE ************\n");
    for (i = 0; i < num_rows; ++i) {
        fprintf(fh, "C%d: ", landmarks->labels[i]);
        _print_nodeIDs(fh, num_cols, landmarks->leaf_nodes[i]);
    }
}

void _print_nodeIDs(FILE* fh, uint num_trees, const int* nodes)
{
    uint t;
    fprintf(fh, "%d", nodes[0]);
    for (t = 1; t < num_trees; ++t) {
        fprintf(fh, " %d", nodes[t]);
    }
    fprintf(fh, "\n");
}
