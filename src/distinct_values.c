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
#include "crossval.h"
#include "distinct_values.h"
#include "util.h"
#include "av_utils.h"

void _tree_to_array(float *array, BST_Node *tree, int node, int *me) {
    if (tree[node].left != -1)
        _tree_to_array(array, tree, tree[node].left, me);
    array[(*me)++] = tree[node].value;
    if (tree[node].right != -1)
        _tree_to_array(array, tree, tree[node].right, me);
}

void tree_to_array(float *array, BST_Node *tree) {
    int me = 0;
    _tree_to_array(array, tree, 0, &me);
}

short tree_insert(BST_Node **tree, Tree_Bookkeeping *books, float new_value) {
    int x = 0;
    Boolean keep_going = TRUE;
    while (keep_going) {
        if (new_value < (*tree)[x].value) {
            if ((*tree)[x].left == -1) {
                // Can't go any further -- create new node
                keep_going = FALSE;
                (*tree)[x].left = books->next_unused_node;
                
                // Make sure we have a node allocated
                while (books->num_malloced_nodes <= books->next_unused_node) {
                    books->num_malloced_nodes *= 2;
                    *tree = (BST_Node *)realloc(*tree, books->num_malloced_nodes * sizeof(BST_Node));
                }
                (*tree)[books->next_unused_node].value = new_value;
                (*tree)[books->next_unused_node].left = -1;
                (*tree)[books->next_unused_node].right = -1;
                
                books->next_unused_node++;
            } else {
                x = (*tree)[x].left;
            }
        } else if (new_value > (*tree)[x].value) {
            if ((*tree)[x].right == -1) {
                // Can't go any further -- create new node                
                keep_going = FALSE;
                (*tree)[x].right = books->next_unused_node;
                
                // Make sure we have a node allocated
                while (books->num_malloced_nodes <= books->next_unused_node) {
                    books->num_malloced_nodes *= 2;
                    *tree = (BST_Node *)realloc(*tree, books->num_malloced_nodes * sizeof(BST_Node));
                }
                (*tree)[books->next_unused_node].value = new_value;
                (*tree)[books->next_unused_node].left = -1;
                (*tree)[books->next_unused_node].right = -1;
                
                books->next_unused_node++;                
            } else {
                x = (*tree)[x].right;
            }
        } else {
            // We've seen this value already
            return 0;
        }
    }
    
    return 1;
}

int translate(float *array, float value, int low, int high) {
    int match = (low + high)/2;
    //    printf("   low -> match -> high = %d -> %d -> %d\n", low, match, high);
    //    printf("   value -> value = %10.6e -> %10.6e\n", value, array[match]);
    if(isnan(value)) {
      printf("ERROR: Nan value detected\n");
      exit(1);
    }
       
    if (value == array[match])
        return match;
    else {
        if (value > array[match])
            return translate(array, value, match, high);
        else
            return translate(array, value, low, match);
    }
}

int translate_discrete(char **map, int num_ids, char *value) {
    int i;
    //printf("Looking for %s ...\n", value);
    for (i = 0; i < num_ids; i++) {
        //printf("  ... '%s'?\n", map[i]);
        if (! strcmp(map[i], value))
            return i;
    }
    return -1;
}

void create_cv_subset(CV_Dataset data, CV_Subset *train) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j, k;
    AV_ReturnCode rc;
    void *data_ptr;
    int num_data_points;
    BST_Node **tree;
    Tree_Bookkeeping *books;
    
    train->meta.num_classes = data.meta.num_classes;
    train->meta.num_attributes = data.meta.num_attributes;
    train->meta.num_fclib_seq = data.meta.num_fclib_seq;
    train->meta.num_examples = data.meta.num_examples;
    train->meta.class_names = data.meta.class_names;
    train->meta.attribute_names = data.meta.attribute_names;
    train->meta.attribute_types = data.meta.attribute_types;
    train->meta.global_offset = data.meta.global_offset;
    train->examples = data.examples;
    free(train->meta.num_examples_per_class);
    train->meta.num_examples_per_class = (int *)malloc(data.meta.num_classes * sizeof(int));
    for (i = 0; i < data.meta.num_classes; i++)
        train->meta.num_examples_per_class[i] = data.meta.num_examples_per_class[i];
    //printf("Copying %d sequences\n", data.meta.exo_data.num_seq_meshes);
    if (data.meta.exo_data.num_seq_meshes > 0) {
        train->meta.exo_data.num_seq_meshes = data.meta.exo_data.num_seq_meshes;
        train->meta.exo_data.seq_meshes = (FC_Mesh *)malloc(train->meta.exo_data.num_seq_meshes * sizeof(FC_Mesh));
        for (i = 0; i < train->meta.exo_data.num_seq_meshes; i++)
            train->meta.exo_data.seq_meshes[i] = data.meta.exo_data.seq_meshes[i];
        train->meta.exo_data.assoc_type = data.meta.exo_data.assoc_type;
    }
    
    // Allocate one tree and one set of high/low for each attribute
    tree = (BST_Node **)malloc(train->meta.num_attributes * sizeof(BST_Node *));
    books = (Tree_Bookkeeping *)malloc(train->meta.num_attributes * sizeof(Tree_Bookkeeping));
    train->low = (int *)malloc(train->meta.num_attributes * sizeof(int));
    train->high = (int *)malloc(train->meta.num_attributes * sizeof(int));

    for (i = 0; i < train->meta.num_fclib_seq; i++) {
        for (j = 0; j < train->meta.num_attributes; j++) {
            
            // Initialize the tree for each attribute on the first sequence only
            if (i == 0) {
                books[j].num_malloced_nodes = 1000;
                books[j].next_unused_node = 1;
                books[j].current_node = 0;
                tree[j] = (BST_Node *)malloc(books[i].num_malloced_nodes * sizeof(BST_Node));
                //tree[j] = NULL;
            }
            
            // Get data pointer for this attribute
            rc = fc_getVariableDataPtr(data.meta.exo_data.variables[i][j], &data_ptr);
            av_exitIfErrorPrintf(rc, "Failed to get data pointer for %s on mesh %d\n", data.meta.attribute_names[j], i);
            
            // Step through data and insert into tree
            rc = fc_getVariableNumDataPoint(data.meta.exo_data.variables[i][j], &num_data_points);
            for (k = 0; k < num_data_points; k++) {
                if (books[j].current_node == 0) {
                    tree[j][0].value = *((double *)(data_ptr) + k);
                    tree[j][0].left = -1;
                    tree[j][0].right = -1;
                    train->low[j] = 0;
                    train->high[j] = 0;
                    
                    // current_node == 0 only triggers the initialization. Otherwise we don't care about it.
                    // Increment it only so the initialization is not repeated
                    books[j].current_node++;
                } else {
                    train->high[j] += tree_insert(&tree[j], &books[j], *((double *)(data_ptr) + k));
                }
            }
        }
    }
    
    // Create the float array to translate int back to float for each attribute
    train->float_data = (float **)malloc(train->meta.num_attributes * sizeof(float *));
    for (i = 0; i < train->meta.num_attributes; i++) {
        train->float_data[i] = (float *)malloc((train->high[i] + 1)*sizeof(float));
        tree_to_array(train->float_data[i], tree[i]);
        free(tree[i]);
    }
    free(tree);
    #else
    av_missingFCLIB();
    #endif
}

void populate_distinct_values_from_dataset(CV_Dataset data, CV_Subset *sub, AV_SortedBlobArray *blob) {
    #ifdef HAVE_AVATAR_FCLIB
    int i, j, k;
    int gid;
    FC_ReturnCode rc;
    
    for (i = 0; i < data.meta.num_fclib_seq; i++) {
        for (j = 0; j < data.meta.num_attributes; j++) {
            // Get data pointer for this mesh and attribute
            void *data_ptr;
            rc = fc_getVariableDataPtr(data.meta.exo_data.variables[i][j], &data_ptr);
            av_exitIfErrorPrintf(rc, "Failed to get data pointer for sequence %d and attribute %d\n", i, j);
            
            for (k = 0; k < data.meta.global_offset[i+1] - data.meta.global_offset[i]; k++) {
                gid = fclib2global(i, k, data.meta.num_fclib_seq, data.meta.global_offset);
                if (j == 0) {
                    sub->examples[gid].distinct_attribute_values =
                                                            (int *)malloc(data.meta.num_attributes * sizeof(int));
                    rc = av_addBlobToSortedBlobArray(blob, &sub->examples[gid],
                                                     cv_example_compare_by_seq_id);
                    if (rc < 0) {
                        av_exitIfErrorPrintf(rc, "Failed to add example %d to SBA\n", gid);
                    } else if (rc == 0) {
                        fprintf(stderr, "Example %d already exists in SBA\n", gid);
                    }
                }
                sub->examples[gid].distinct_attribute_values[j] =
                                      translate(sub->float_data[j], *((double *)data_ptr + k), 0, sub->high[j] + 1);
            }
        }
    }
    #else
    av_missingFCLIB();
    #endif
}

