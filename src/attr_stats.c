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
#include "crossval.h"
#include "attr_stats.h"

typedef struct attr_info {
  double mass;
  double mass_no_dup;
  double path_mass;
  double deviance;
  double depth;
} Attr_Influence;

// This is typedef'd to Attr_Stats in header file.
struct attr_stats_struct {
  int num_attr; // number of active attributes (non-skipped)
  Attr_Influence* importance;
  int total_num_attr; // number of input columns, minus 1 for label column
  char** names;       // names of the inputs
  Boolean* skipped;   // bookkeepping for which input columns were skipped
}; // Attr_Stats;

// Prototypes for helper functions.
void _tally_subtree_stats(const DT_Node *tree, 
			  int node, 
			  int num_classes, 
			  int depth, 
			  double *subtree_path_mass,
			  Attr_Stats *stats);
void _format_depth(FILE *out, int depth);
void _write_class_dist(FILE *out, int num_classes, const int *distribution);
void _write_subtree(FILE *out, const DT_Node *tree, int node, int num_classes, int depth);
void _write_ensemble(FILE *out, const DT_Ensemble *ensemble);



/* void */
/* reset_attr_stats(Attr_Stats* stats) */
/* { */
/*   assert(stats != NULL); */
/*   Attr_Influence empty = {0, 0, 0, 0, 0}; */
/*   int i = 0; */
/*   for (i = 0; i < stats->num_attr; ++i) { */
/*     stats->importance[i] = empty; */
/*   } */
/* } */

void 
compute_feature_imp(const DT_Ensemble *ensemble, Attr_Stats *stats)
{
  assert(ensemble != NULL);
  assert(stats != NULL);
  int i;
  Attr_Influence denom = {0, 0, 0, 0, 0};

  fprintf(stderr, "info: only path_mass and depth implemented at moment\n");
  fprintf(stderr, "info: feature importances may not be reliable for boosted ensembles\n");

  // For each tree in ensemble...
  for (i = 0; i < ensemble->num_trees; ++i) {
    // ...compute feature importance stats from tree structure and training counts
    double root_path_mass = 0.0;
    _tally_subtree_stats(ensemble->Trees[i], 
			 0, 
			 ensemble->num_classes, 
			 0, 
			 &root_path_mass, 
			 stats);
  }

  // Normalize importances to sum to 1.
  for (i = 0; i < stats->num_attr; ++i) {
    denom.path_mass += stats->importance[i].path_mass;
    denom.depth += stats->importance[i].depth;
  }
  for (i = 0; i < stats->num_attr; ++i) {
    stats->importance[i].path_mass /= denom.path_mass;
    stats->importance[i].depth /= denom.depth;
  }
  
  // Following is useful for testing with a single tree.
  //_write_ensemble(stderr, ensemble);
}

void 
write_feature_imp(FILE *fh, const Attr_Stats *stats)
{
  assert(fh != NULL);
  assert(stats != NULL);

  int i, nextActiveFeature;
  Attr_Influence skipped_attr_influence = {0, 0, 0, 0, 0};

  fprintf(fh, "# Feature Mass Mass_No_Dup Path Deviance Depth\n");
  nextActiveFeature = 0;
  for (i = 0; i < stats->total_num_attr; ++i) {
    const char *name = stats->names[i];
    Attr_Influence* imp = NULL;

    if (stats->skipped[i]) {
      imp = &skipped_attr_influence;
    }
    else {
      imp = &(stats->importance[nextActiveFeature]);
      ++nextActiveFeature;
    }

    fprintf(fh, "%s: %g %g %g %g %g\n",
	    name == NULL ? "NULL" : name,
	    imp->mass,
	    imp->mass_no_dup,
	    imp->path_mass,
	    imp->deviance,
	    imp->depth);
  }
}

Attr_Stats* 
malloc_attr_stats(
  const CV_Metadata* meta, 
  int num_skipped_features, 
  const int* skipped_features)
{
  assert(meta != NULL);
  int i;
  
  Attr_Stats* stats = (Attr_Stats*)malloc(sizeof(Attr_Stats));
  if (stats == NULL) {
    fprintf(stderr, "error: failed to allocate space for attribute stats\n");
    exit(-1);
  }

  // Initialize space.
  stats->num_attr = meta->num_attributes - num_skipped_features;
  stats->importance = (Attr_Influence*)calloc(stats->num_attr, sizeof(Attr_Influence));
  if (stats->importance == NULL) {
    fprintf(stderr, "error: failed to allocate space for attribute importance\n");
    free(stats);
    exit(-1);
  }

  // Copy attribute names.
  stats->total_num_attr = meta->num_attributes;
  stats->names = (char**)calloc(stats->total_num_attr, sizeof(char*));
  if (stats->names == NULL) {
    fprintf(stderr, "error: failed to allocate space for attribute names\n");
    exit(-1);
  }
  for (i = 0; i < stats->total_num_attr; ++i) {
    stats->names[i] = av_strdup(meta->attribute_names[i]);
    if (stats->names[i] == NULL) {
      fprintf(stderr, "error: failed to allocate space for attribute names\n");
      exit(-1);
    }
  }

  // Set up data structures for tracking which attributes skipped.
  stats->skipped = (Boolean*)calloc(stats->total_num_attr, sizeof(Boolean));
  if (stats->skipped == NULL) {
    fprintf(stderr, "error: failed to allocate space for marking attributes skipped\n");
    exit(-1);
  }
  for (i = 0; i < num_skipped_features; ++i) {
    int offset = skipped_features[i] - 1; // -1 to convert to 0-based indexing
    stats->skipped[offset] = TRUE;
  }

  return stats;
}

void
free_attr_stats(Attr_Stats *stats)
{
  int i;
  if (stats) {
    free(stats->importance);
    stats->importance = NULL;
    stats->num_attr = 0;

    free(stats->skipped);
    stats->skipped = NULL;

    for (i = 0; i < stats->total_num_attr; ++i) {
      free(stats->names[i]);
      stats->names[i] = NULL;
    }
    free(stats->names);
    stats->names = NULL;
    stats->total_num_attr = 0;

    free(stats);
  }
}


/***** PRIVATE HELPER FUNCTIONS FOR MODULE *****/

void 
_tally_subtree_stats(const DT_Node *tree, 
		     int node, 
		     int num_classes, 
		     int depth, // depth of the subtree's root in total tree
		     double *subtree_path_mass,
		     Attr_Stats *stats)
{
  if (tree[node].branch_type == LEAF) {
    // Base case: this subtree is a leaf node.

    // How much mass of training data reached this leaf?
    int leaf_mass = 0;
    int k;
    int *distribution = tree[node].class_count;
    if (distribution == NULL || distribution[0] == -1) {
      // Something is clearly wrong...
      fprintf(stderr, "error: cannot compute feature importance b/c leaf node(s) lack class counts\n");
      exit(-1);
    }

    for (k = 0; k < num_classes; ++k) {
      leaf_mass += distribution[k];
    }

    // Tally the path mass.  For path mass, the leaf mass is evenly
    // distributed across all the nodes in the path from the root 
    // to the leaf.  The rationale is that each internal node on the
    // path contributed equally to sorting the training cases that
    // reached this leaf.
    //
    // Two facts help:
    // 1) A node's depth equals the number of feature tests above it.
    //    => a leaf's depth equals the # of internal nodes above it.
    // 2) The path mass of an internal node N equals the sum of the path
    // masses for subtrees rooted at N.  (Proof by induction.)
    //
    // Therefore, this subtree's path mass is leaf_mass / depth.
    *subtree_path_mass = ((double)leaf_mass) / depth;
  }
  else {
    // General case: the node has child nodes.
    // Recursively compute stats for all children.
    int t, attr;
    for (t = 0; t < tree[node].num_branches; ++t) {
      double child_path_mass = 0.0;
      int child = tree[node].Node_Value.branch[t];
      _tally_subtree_stats(tree, child, num_classes, depth+1, &child_path_mass, stats);
      // The total path mass for this subtree is the sum of the
      // children's path masses.  (See note above in base case.)
      *subtree_path_mass += child_path_mass;
    }
    
    // Update the stats for this node's feature.
    attr = tree[node].attribute;
    stats->importance[attr].path_mass += *subtree_path_mass;
    stats->importance[attr].depth += 1.0 / (depth + 1);
  }
}


void
_write_ensemble(FILE *out, const DT_Ensemble *ensemble)
{
  assert(ensemble != NULL);
  assert(out != NULL);

  int i;
  for (i = 0; i < ensemble->num_trees; ++i) {
    fprintf(out, "******************* TREE %d *******************\n", i+1);
    _write_subtree(out, ensemble->Trees[i], 0, ensemble->num_classes, 0);
    fprintf(out, "\n");
  }
}

void
_write_subtree(FILE *out, const DT_Node *tree, int node, int num_classes, int depth)
{
  int i;

  if (tree[node].branch_type == LEAF) {
    _format_depth(out, depth);
    fprintf(out, "LEAF Class %d", tree[node].Node_Value.class_label);
    _write_class_dist(out, num_classes, tree[node].class_count);
    fprintf(out, "\n");
  }
  else {
    // TODO: this causes seg fault b/c class_count not initialized to anything for non-leaf nodes
    //  (I.e., not NULL, not a valid array, nothing... boo

    // non-leaf node
    if (tree[node].attribute_type == CONTINUOUS) {
      _format_depth(out, depth);
      fprintf(out, "SPLIT CONTINUOUS ATT# %d < %#6g", tree[node].attribute, tree[node].branch_threshold);
      //_write_class_dist(out, num_classes, tree[node].class_count);
      fprintf(out, "\n");

      _write_subtree(out, tree, tree[node].Node_Value.branch[0], num_classes, depth+1);

      _format_depth(out, depth);
      fprintf(out, "SPLIT CONTINUOUS ATT# %d >= %#6g", tree[node].attribute, tree[node].branch_threshold);
      //_write_class_dist(out, num_classes, tree[node].class_count);
      fprintf(out, "\n");

      _write_subtree(out, tree, tree[node].Node_Value.branch[1], num_classes, depth+1);
    } 
    else if (tree[node].attribute_type == DISCRETE) {
      for (i = 0; i < tree[node].num_branches; i++) {
	_format_depth(out, depth);
	fprintf(out, "SPLIT DISCRETE ATT# %d VAL# %d / %d", tree[node].attribute, i+1, tree[node].num_branches);
	//_write_class_dist(out, num_classes, tree[node].class_count);
	fprintf(out, "\n");

	_write_subtree(out, tree, tree[node].Node_Value.branch[i], num_classes, depth+1);
      }
    }
  }
}

void
_format_depth(FILE *out, int depth)
{
  while (depth > 0) {
    fprintf(out, "| ");
    --depth;
  }
}

void
_write_class_dist(FILE *out, int num_classes, const int *distribution)
{
  int k;

  if (distribution == NULL || distribution[0] == -1) {
    // class counts not available
    return;
  }

  fprintf(out, " (");
  for (k = 0; k < num_classes; ++k) {
    fprintf(out, "%d%s", distribution[k], k==(num_classes-1)?")":" ");    
  }
}


