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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "av_utils.h"
#include "crossval.h"
#include "memory.h"

void clear_CV_Metadata(CV_Metadata* meta, Data_Format format, Boolean read_folds)
{
    int i, j;

    if (meta == NULL)
        return;

    free(meta->global_offset);

    // Free data about classes.
    for (i = 0; i < meta->num_classes; ++i) {
        free(meta->class_names[i]);  
    }
    free(meta->class_names);
    free(meta->num_examples_per_class);

    // Free data about attributes.
    for (i = 0; i < meta->num_attributes; ++i) {
        free(meta->attribute_names[i]);
        if (meta->attribute_types[i] == DISCRETE) {
            for (j = 0; j < meta->num_discrete_values[i]; ++j) {
                free(meta->discrete_attribute_map[i][j]);
            }
            free(meta->discrete_attribute_map[i]);
        }
    }
    free(meta->attribute_names);
    if (format == AVATAR_FORMAT) {
        // REVIEW-2012-03-27-ArtM: Why are these not freed if format == EXODUS?
        free(meta->num_discrete_values);
        free(meta->discrete_attribute_map);
    }
    free(meta->attribute_types);
    free(meta->Missing);

    #ifdef HAVE_AVATAR_FCLIB
    // Free exo_data.
    for (i = 0; i < meta->exo_data.num_seq_meshes; i++) {
        for (j = 0; j < meta->num_attributes; j++) {
            FC_ReturnCode rc = fc_releaseVariable(meta->exo_data.variables[i][j]);
            if (rc != FC_SUCCESS)
               fprintf(stderr, "Error releasing seq var %d/%d: %d\n", i, j, rc);
        }
        free(meta->exo_data.variables[i]);
    }
    if (format == EXODUS_FORMAT && !read_folds) {
        // REVIEW-2012-03-27-ArtM: Is this actually correct?
        free(meta->exo_data.variables);
    }

    for (i = 0; i < meta->exo_data.num_seq_meshes; i++) {
        FC_ReturnCode rc = fc_deleteMesh(meta->exo_data.seq_meshes[i]);
        if (rc != FC_SUCCESS)
           fprintf(stderr, "Failed to delete mesh %d: %d\n", i, rc);
    }
    if (meta->exo_data.num_seq_meshes > 0) {
        // REVIEW-2012-03-27-ArtM: Is this okay?
        // Also, if data structure were initialized to 0 when unused,
        // then we should be able to call free without checking for > 0.
        free(meta->exo_data.seq_meshes);  
    }
    #endif
    memset(meta, 0, sizeof(CV_Metadata));
}

//Call this to free the metadata of a CV_Subset,
//where arbitrary pointer members may alias members of the
//corresponding CV_Dataset's metadata (aliased). aliased is not affected.
//
//Does NOT free exo_data.
void free_CV_Metadata_Aliasing(CV_Metadata* meta, CV_Metadata* aliased, Data_Format format, Boolean read_folds)
{
  int i,j;
  if (meta == NULL)
    return;
  if(meta->global_offset != aliased->global_offset)
    free(meta->global_offset);
  // Free data about classes.
  if(meta->class_names != aliased->class_names)
  {
    for (i = 0; i < meta->num_classes; ++i) {
      free(meta->class_names[i]);  
    }
    free(meta->class_names);
  }
  if(meta->num_examples_per_class != aliased->num_examples_per_class)
    free(meta->num_examples_per_class);
  // Free data about attributes.
  if(meta->attribute_names && meta->attribute_names != aliased->attribute_names)
  {
    for (i = 0; i < meta->num_attributes; ++i)
      free(meta->attribute_names[i]);
    free(meta->attribute_names);
  }
  if(meta->discrete_attribute_map &&
     meta->discrete_attribute_map != aliased->discrete_attribute_map)
  {
    for (i = 0; i < meta->num_attributes; ++i) {
      if (meta->attribute_types[i] == DISCRETE) {
        for (j = 0; j < meta->num_discrete_values[i]; ++j) {
          free(meta->discrete_attribute_map[i][j]);
        }
        free(meta->discrete_attribute_map[i]);
      }
    }
    free(meta->discrete_attribute_map);
  }
  if(meta->num_discrete_values != aliased->num_discrete_values)
    free(meta->num_discrete_values);
  if(meta->attribute_types != aliased->attribute_types)
    free(meta->attribute_types);
  if(meta->Missing != aliased->Missing)
    free(meta->Missing);
}

void free_DT_Ensemble(DT_Ensemble ensemble, CV_Mode mode) {
    //printf("free_DT_Ensemble\n");
    int i;
    for (i = 0; i < ensemble.num_trees; i++)
    {
      if(ensemble.Trees[i])
        free_DT_Node(ensemble.Trees[i], ensemble.Books[i].next_unused_node);
    }
    //Cosmin added if statements below
    if (ensemble.Books != NULL) {
        free(ensemble.Books);
        ensemble.Books = NULL;
    }
    if (ensemble.Trees != NULL) {
        free(ensemble.Trees);
        ensemble.Trees = NULL;
    }
    if (ensemble.boosting_betas != NULL) {
        free(ensemble.boosting_betas);
        ensemble.boosting_betas = NULL ;
    }
    if (ensemble.attribute_types != NULL) {
        free(ensemble.attribute_types);
        ensemble.attribute_types = NULL;
    }
    if (ensemble.num_training_examples_per_class != NULL) {
        free(ensemble.num_training_examples_per_class);
        ensemble.num_training_examples_per_class = NULL;
    }
    if (ensemble.Missing != NULL) {
        free(ensemble.Missing);
        ensemble.Missing = NULL;
    }
    //if (mode == TRAIN_MODE)
    //    free(ensemble.weights);
}

//Modified by DACIESL June-03-08: Laplacean Estimates
//Added trees[i].branch_type == LEAF case to free class_count and class_prob variables
void free_DT_Node(DT_Node *trees, int num_nodes) {
    int i;
    for (i = 0; i < num_nodes; i++) {
        if (trees[i].branch_type != LEAF && trees[i].num_branches > 0)
            free(trees[i].Node_Value.branch);
        else if (trees[i].branch_type == LEAF) {
	  free(trees[i].class_count);
	  free(trees[i].class_probs);
        }
    }
    free(trees);
}

void free_Vote_Cache(Vote_Cache cache, Args_Opts args) {
    //printf("free_Vote_Cache\n");
    int j;
    if (args.do_training) {
        free(cache.best_train_class);
        for (j = 0; j < cache.num_train_examples; j++) {
            free(cache.oob_class_votes[j]);
            free(cache.oob_class_weighted_votes[j]);
	}
        free(cache.oob_class_votes);
        free(cache.oob_class_weighted_votes);

        free(cache.best_test_class);
        for (j = 0; j < cache.num_test_examples; j++) {
            free(cache.class_votes_test[j]);
            free(cache.class_weighted_votes_test[j]);
	}
        free(cache.class_votes_test);
        free(cache.class_weighted_votes_test);
    }
}

void free_CV_Class(CV_Class class) {
    //printf("free_CV_Class\n");
    int i;
    free(class.class_frequencies);
    for (i = 0; i < class.num_classes; i++)
        free(class.class_names[i]);
    free(class.class_names);
    free(class.class_var_name);
    free(class.thresholds);
}

void free_CV_Dataset(CV_Dataset data, Args_Opts args) {
    //free(data.examples); // Already freed in free_CV_Subset?
    clear_CV_Metadata(&(data.meta), args.format, args.read_folds);
}

void free_CV_Subset(CV_Subset* sub, Args_Opts args, CV_Mode mode) {
    //printf("free_CV_Subset\n");
    int i;
    if(sub->examples)
    {
      for (i = 0; i < sub->meta.num_examples; i++) {
          if(sub->examples[i].distinct_attribute_values) {
              free(sub->examples[i].distinct_attribute_values);
              sub->examples[i].distinct_attribute_values = NULL;
          }
      }
      free(sub->examples);
      sub->examples = NULL;
    }
    free(sub->high);
    free(sub->low);
    sub->high = NULL;
    sub->low = NULL;
    if(sub->float_data)
    {
      for (i = 0; i < sub->meta.num_attributes; i++) {
        if (sub->float_data[i]){
          free(sub->float_data[i]);
          sub->float_data[i] = NULL;
        }
      }
      free(sub->float_data);
      sub->float_data = NULL;
    }
    if (args.format == AVATAR_FORMAT)
      free(sub->discrete_used);
    sub->discrete_used = NULL;
}

// Sort of the same as free_CV_Subset except float_data is not touched since this is not
// copied but pointed to by CV_Subset so the original must remain
void free_CV_Subset_inter(CV_Subset* sub, Args_Opts args, CV_Mode mode) {
    //printf("free_CV_Subset\n");
    //int i;
    //for (i = 0; i < sub.num_examples; i++)
    //    free(sub.examples[i].distinct_attribute_values);
    free(sub->examples);
    sub->examples = NULL;
    //if (mode == TRAIN_MODE && ! args.do_ivote && args.random_subspaces == 0)
    //    free(sub.weights);
    free(sub->high);
    free(sub->low);
    free(sub->discrete_used);
    sub->high = NULL;
    sub->low = NULL;
    sub->discrete_used = NULL;
    if (args.format == EXODUS_FORMAT && ! args.read_folds) {
        //free(sub.exo_data.seq_meshes);
        //free(sub.global_offset); causes double free
        //for (i = 0; i < sub.num_fclib_seq; i++) causes seg fault
        //    free(sub.exo_data.seq_variables[i]);
        //free(sub.exo_data.seq_variables);
    }
}

