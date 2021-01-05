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
/**
 * \file connect_comp_regions.c
 * \brief Thresholds a variable and then computes connected component regions.
 *
 * $Source: /home/Repositories/avatar/avatar/tools/connect_comp_regions.c,v $
 * $Revision: 1.19 $ 
 * $Date: 2006/10/06 05:25:52 $
 *
 * \description
 *    Usage: connect_comp_regions dataset "variable name" "op" threshold_value
 *    where "op: is the threshold operator such as ">", "<=", "==", etc. 
 *    Quotes are required around the op string, but can be dropped around
 *    the variable name if there are no spaces or punctuation in the name.
 *
 *    Example: connect_comp_regions -f data.ex2 -V Osaliency -o ">" -t 0.5
 *
 * \modifications
 *   - 04-DEC-2005 KAB: First version.
 *   - 07-MAR-2006 KAB: Removed the -r option to do smoothing.
 *                      Other programs provide this service.
 *   - 04-APR-2006 KAB: Added -r option back in at Larry S.'s request
 *                      Using new bounding box and proximity routines
 *   - 25-JUN-2006 KAB: Added handling of displaced meshes
 *   - 28-AUG-2006 KAB: Added the doLocalSmooth option
 *   - 14-SEP-2006 KAB: Fixed logic in finding close neighbors. Some regions were
 *                      not compared to all other regions and so were ignored for
 *                      joining.
 */
#include <math.h>
#include <string.h>
#include "fc.h"
#include "fcP.h"
#include "array.h"
#include "regions.h"
//#include "displacements.h"
#include <sys/param.h>
#ifndef _GNU_SOURCE
  #include "getopt.h"
#else
  #include <getopt.h>
#endif

struct Global_Args_t {
    double max_sep_dist;
    int do_smoothing;
    int use_displ_mesh;
    int got_displ_var;
    float radius;
    int do_local_smooth;
    char *displ_var;
    int use_bb;
    int do_join;
    char *file_name;
    char *var_name;
    char *operator;
    double threshold;
} Args;

int find_ccrs(int step_id, int num, FC_Variable* vars, FC_Variable **displ, timestep *ts, int **n_lut, int *e_offset);
int _process_opts(int argc, char **argv);
void _display_usage( void );

typedef struct Segments_Metadata_struct {
    int segment_id;
    int mesh_number;
    int on_mesh_segment_number;
    int segment_label;
} Seg_MD;

static FC_VerbosityLevel verbose_level = FC_QUIET;

int main(int argc, char** argv) {
    FC_ReturnCode rc;
    int i, j;
    int num_mesh, num_seq_var, num_step;
    FC_Dataset ds;
    FC_Sequence sequence;
    FC_Mesh *seq_meshes;
    FC_Variable** seq_vars; // array of seq vars (num_mesh long)
    FC_Variable** displ;
    FC_Variable** smooth_vars;
    int num_comp;
    int *num_seq_datapoints;
    int *seq_num_steps;
	FC_VertexBin ** bins;
	FC_VertexBin*** displ_bins;
    int **exo_node_lut;
    int *exo_element_offset;
        
    region_file_metadata metadata;
    timestep *ts;
    
    // handle arguments
    _process_opts(argc, argv);
    
    // init library and load dataset
    rc = fc_setLibraryVerbosity(verbose_level);
    fc_exitIfError(rc);
    rc = fc_initLibrary();
    fc_exitIfError(rc);
    rc = fc_loadDataset(Args.file_name, &ds);
    if (rc != FC_SUCCESS) {
        fprintf(stderr, "Could not load dataset: '%s'\n", Args.file_name);
        fc_exitIfError(rc);
    }

    // get number of meshes and setup var arrays
    rc = fc_getMeshes(ds, &num_mesh, &seq_meshes);
    fc_exitIfError(rc);
    seq_vars = (FC_Variable**)malloc(num_mesh*sizeof(FC_Variable*));
    exo_node_lut = (int**)malloc(num_mesh*sizeof(int*));
    exo_element_offset = (int*)malloc(num_mesh*sizeof(int));
    seq_num_steps = (int*)malloc(num_mesh*sizeof(int));
    num_seq_var = 0;
    for (i = 0; i < num_mesh; i++) {
        FC_Variable **sv;
        int nsv, *nspv;
        fc_getSeqVariableByName(seq_meshes[i], Args.var_name, &nsv, &nspv, &sv);
//        &seq_num_steps[num_seq_var], &seq_vars[num_seq_var]);
//        if (seq_vars[num_seq_var]) {
        if (rc == FC_SUCCESS && nsv == 1) {
            seq_vars[num_seq_var] = sv[0];
            seq_num_steps[num_seq_var] = nspv[0];
            if (seq_num_steps[num_seq_var] != seq_num_steps[0])
                fc_exitIfErrorPrintf(FC_ERROR, "Sequence variable must have the same number of steps on all meshes");
            _fc_getMeshExodusGlobalNodalIDsPtr(seq_meshes[i], &exo_node_lut[i]);
            _fc_getMeshExodusGlobalElementIDOffset(seq_meshes[i], &exo_element_offset[i]);
            num_step = seq_num_steps[num_seq_var];
            seq_meshes[num_seq_var] = seq_meshes[i];
            num_seq_var++;
        }
    }
    free(seq_num_steps);
    
    // did we find anything?
    if (num_seq_var < 1)
        fc_exitIfErrorPrintf(FC_ERROR, "Failed to find variable '%s' in dataset '%s'", Args.var_name, Args.file_name);

    // make sure seq variable makes sense
    if (num_seq_var > 0) {
        num_seq_datapoints = (int*)malloc(sizeof(int)*num_seq_var);
        FC_Sequence temp_sequence;
        fc_getSequenceFromSeqVariable(num_step, seq_vars[0], &sequence);
        fc_exitIfError(rc);
        for (i = 0; i < num_seq_var; i++) {
            fc_getSequenceFromSeqVariable(num_step, seq_vars[i], &temp_sequence);
            if (!FC_HANDLE_EQUIV(temp_sequence, sequence)) 
                fc_exitIfErrorPrintf(FC_ERROR, "Abort because variable '%s' was found "
                                               "on more than one sequence", Args.var_name);
            fc_getVariableInfo(seq_vars[i][0], &num_seq_datapoints[i], &num_comp, NULL, NULL, NULL);
            if (num_comp != 1)
                fc_exitIfErrorPrintf(FC_ERROR, "Cannot threshold variable '%s' because"
                                               " it has %d component\n", Args.var_name, num_comp);
        }
    }
    
    if (Args.use_displ_mesh) {
        FC_Variable **sv;
        int nsv, *nspv;
        displ = (FC_Variable **)malloc(num_mesh * sizeof(FC_Variable *));
        for (i = 0; i < num_mesh; i++) {
            if (Args.got_displ_var) {
                rc = fc_getSeqVariableByName(seq_meshes[i], Args.displ_var, &nsv, &nspv, &sv);
                if (rc != FC_SUCCESS || nsv != 1) {
                    fprintf(stderr, "Did not find displacment var '%s' on mesh %d\n", Args.displ_var, i);
                    exit(-1);
                }
                displ[i] = sv[0];
            } else {
                // Try DISPL
                rc = fc_getSeqVariableByName(seq_meshes[i], "DISPL", &nsv, &nspv, &sv);
                if (rc == FC_SUCCESS && nsv == 1) {
                    displ[i] = sv[0];
                } else {
                    free(nspv);
                    // Try displ_
                    rc = fc_getSeqVariableByName(seq_meshes[i], "displ_", &nsv, &nspv, &sv);
                    if (rc == FC_SUCCESS && nsv == 1) {
                        displ[i] = sv[0];
                    } else {
                        free(nspv);
                        // Try DIS
                        rc = fc_getSeqVariableByName(seq_meshes[i], "DIS", &nsv, &nspv, &sv);
                        if (rc == FC_SUCCESS && nsv == 1) {
                            displ[i] = sv[0];
                        } else {
                            fprintf(stderr, "Did not find displacment var on mesh %d\n", i);
                            exit(-1);
                        }
                    }
                }
            }
            free(nspv);
        }
    }
        //get_displacement_var(ds, Args.displ_var, &displ);
	
    // Smooth if asked
    if (Args.do_smoothing) {
        if (num_seq_var > 0) {
            if (Args.use_displ_mesh) {
                displ_bins = (FC_VertexBin***)malloc(num_seq_var * sizeof(FC_VertexBin **));
                for (i = 0; i < num_seq_var; i++) {
					displ_bins[i] = (FC_VertexBin**)malloc(num_step * sizeof(FC_VertexBin *));
					for (j = 0; j < num_step; j++) {
						displ_bins[i][j] = NULL;
						rc = fc_createDisplacedMeshVertexBin(seq_meshes[i], displ[i][j], &displ_bins[i][j]);
						fc_exitIfErrorPrintf(rc, "Failed to create displaced bin for mesh %d step %d\n", i, j);
					}
				}
				
				rc = fc_displacedGeomSmoothSeqVariable(num_seq_var, seq_meshes, num_step, displ, displ_bins, num_step,
				                                       seq_vars, Args.radius, Args.do_local_smooth, &smooth_vars);
				fc_exitIfErrorPrintf(rc, "Failed to smooth with displacement the variable");
				
				for (i = 0; i < num_seq_var; i++) {
					for (j = 0; j < num_step; j++) {
						free(displ_bins[i][j]);
					}
					free(displ_bins[i]);
				}
				free(displ_bins);
			} else {
				bins = (FC_VertexBin**)malloc(num_seq_var * sizeof(FC_VertexBin *));
				for (i = 0; i < num_seq_var; i++) {
					bins[i] = NULL;
					rc = fc_createMeshVertexBin(seq_meshes[i], &bins[i]);
					fc_exitIfErrorPrintf(rc, "Failed to create mesh vertex bin for mesh %d\n", i);
				}
				
				rc = fc_geomSmoothSeqVariable(num_seq_var, seq_meshes, bins, num_step, seq_vars,
                                              Args.radius, Args.do_local_smooth, &smooth_vars);
				fc_exitIfErrorPrintf(rc, "Failed to smooth the variable");
				
				for (i = 0; i < num_seq_var; i++)
					free(bins[i]);
				free(bins);
			}
		} else {
            fprintf(stderr, "No variable found to smooth\n");
        }
	} else {
		smooth_vars = seq_vars;
	}
	
    // Determine a characteristic length scale of the undeformed mesh
    // for use as a threshold for determining "close" regions
    
    // Use the maximum edge length on all meshes as this characteristic length
    if (Args.max_sep_dist < 0.0) {
        FC_Variable edge_lengths;
        FC_MathType math_type;
        FC_DataType data_type;
        double var_min, var_max;
        double min_var_min = -1.0;
        
        for (i = 0; i < num_seq_var; i++) {
            rc = fc_getEdgeLengths(seq_meshes[i], &edge_lengths);
            fc_exitIfErrorPrintf(rc, "Failed to get edge lengths");
            rc = fc_getVariableInfo(edge_lengths, NULL, NULL, NULL, &math_type, &data_type);
            if (data_type != FC_DT_DOUBLE && math_type != FC_MT_SCALAR) {
                fprintf(stderr, "edge_lengths variable not what I expected\n");
                fprintf(stderr, "Cannot determine max sep dist automatically\n");
                exit(-1);
            } else {
                rc = fc_getVariableMinMax(edge_lengths, &var_min, NULL, &var_max, NULL);
                if (Args.max_sep_dist < 0.0 || var_max > Args.max_sep_dist) {
                    Args.max_sep_dist = var_max;
                    if (verbose_level >= FC_DEBUG_MESSAGES)
                        fprintf(stderr, "Found new max: %f\n", var_max);
                }
                if (min_var_min < 0.0 || var_min < min_var_min) {
                    min_var_min = var_min;
                    if (verbose_level >= FC_DEBUG_MESSAGES)
                        fprintf(stderr, "Found new min: %f\n", var_min);
                }
            }
        }        
    }
    if (verbose_level >= FC_LOG_MESSAGES)
        fprintf(stderr, "Using %f for max_sep_dist\n", Args.max_sep_dist);
    
    ts = (timestep*)malloc(num_step * sizeof(timestep));
    
    // Set metadata
    metadata.file_name = (char*)malloc( (strlen(Args.file_name) + 1) * sizeof(char) );
    strcpy(metadata.file_name, Args.file_name);
    metadata.var_name = (char*)malloc( (strlen(Args.var_name) + 1) * sizeof(char));
    strcpy(metadata.var_name, Args.var_name);
    metadata.threshold_op = (char*)malloc( (strlen(Args.operator) + 1) * sizeof(char));
    strcpy(metadata.threshold_op, Args.operator);
    metadata.threshold_val = Args.threshold;
    metadata.joining = Args.do_join;
    metadata.max_separation = Args.max_sep_dist;
    metadata.smoothing = Args.do_smoothing ? ( Args.use_displ_mesh ? DISPLACED : UNDISPLACED ) : NONE;
    if (metadata.smoothing != NONE)
        metadata.smoothing_radius = Args.radius;
    else
        metadata.smoothing_radius = -1.0;
    metadata.num_timesteps = num_step;
    
    // then do the seq vars by step
    if (num_seq_var > 0) {
        for (i = 0; i < num_step; i++) {
            ts[i].timestep = i;
            
            // get vars for just this step
            FC_Variable temp_vars[num_seq_var];
            for (j = 0; j < num_seq_var; j++)
                temp_vars[j] = smooth_vars[j][i];
            // do it
            rc = find_ccrs(i, num_seq_var, temp_vars, displ, ts, exo_node_lut, exo_element_offset);
            fc_exitIfError(rc);

        }
    }

    // Print out results
    print_region_data(ts, metadata, NULL);
    
    // cleanup
    for (i = 0; i < num_seq_var; i++) {
        free(seq_vars[i]);
        if (Args.do_smoothing && Args.use_displ_mesh)
            free(displ[i]);
    }
    free(seq_vars);
    if (Args.do_smoothing && Args.use_displ_mesh)
        free(displ);
    free(metadata.file_name);
    free(metadata.var_name);
    free(metadata.threshold_op);
    for (i = 0; i < num_step; i++) {
        for (j = 0; j < ts[i].num_regions; j++)
            free(ts[i].regions[j].members);
        free(ts[i].regions);
    }
    free(ts);
    free(num_seq_datapoints);
    free(seq_meshes);
    free(exo_node_lut);
    free(exo_element_offset);
    free(Args.file_name);
    free(Args.var_name);
    free(Args.operator);
    fc_finalLibrary();
  
    exit(0);
}

// Find the thresholded regions & print output.
int find_ccrs(int step_id, int num_vars, FC_Variable* vars, FC_Variable** displ,
              timestep *ts, int** n_lut, int *e_offset) {
    FC_ReturnCode rc; 
    int i, j, k, n;
    int num_dim, num_segment;
    //int numBox = 0;
    FC_Coords *lowpoints, *highpoints;
    FC_Coords temp_lowpoint, temp_highpoint;
    FC_Subset** segments;        // segments
    int overlap_flag;
    FC_Subset subset;
    char *subset_name = "temp_subset";
    int segment_count = 0;
    int join_regions;
    Seg_MD *Metadata = NULL;
    
    if (verbose_level >= FC_LOG_MESSAGES)
        fprintf(stderr, "Timestep: %d\n", step_id);

    // Initialize these arrays. We'll realloc them as needed later
    if (Args.use_bb)
        lowpoints =  highpoints =  NULL;
    
    segments = (FC_Subset**)malloc(num_vars * sizeof(FC_Subset*));
        
    for (i = 0; i < num_vars; i++) {
        // get segments
        rc = fc_createThresholdSubset(vars[i], Args.operator, Args.threshold, subset_name, &subset);				
        if (rc != FC_SUCCESS)
            return rc;
        rc = fc_segment(subset, 0, &num_segment, &segments[i]);
        if (rc != FC_SUCCESS)
            return rc;
        fc_deleteSubset(subset);

        // If none, move on
        if (num_segment < 1)
            continue;
        
        Metadata = (Seg_MD *)realloc(Metadata, (segment_count + num_segment) * sizeof(Seg_MD));
        if (Metadata == NULL)
            fc_exitIfError(FC_MEMORY_ERROR);

        if (Args.use_bb) {
            // Extend bb array
            lowpoints = realloc(lowpoints, (segment_count + num_segment)*sizeof(FC_Coords));
            highpoints = realloc(highpoints, (segment_count + num_segment)*sizeof(FC_Coords));
            if (! lowpoints ||  !highpoints)
                fc_exitIfError(FC_MEMORY_ERROR);
        }

        for (j = 0; j < num_segment; j++) {
            // Initialize segment_id to sequentially increasing number
            Metadata[j + segment_count].segment_id = j + segment_count;
            // Initialize on_mesh_segment_number to sequential on mesh
            Metadata[j + segment_count].mesh_number = i;
            Metadata[j + segment_count].on_mesh_segment_number = j;
            // Initialize segment_label to  sequentially increasing
            Metadata[j + segment_count].segment_label = j + segment_count;
        }

        for (j = 0; j < num_segment; j++) {

            if (Args.use_bb) {
                // get bb
                if (Args.use_displ_mesh)
                    rc = fc_getDisplacedSubsetBoundingBox(segments[i][j], displ[i][step_id], &num_dim,
                                                          &temp_lowpoint, &temp_highpoint);
                else
                    rc = fc_getSubsetBoundingBox(segments[i][j], &num_dim, &temp_lowpoint, &temp_highpoint);
                
                // Extend by msd/2 all around
                for (k = 0; k < num_dim; k++) {
                    temp_lowpoint[k] -= Args.max_sep_dist / 2.0;
                    temp_highpoint[k] += Args.max_sep_dist / 2.0;
                }
                if (rc != FC_SUCCESS)
                    return rc;
            }

            // potentially combine with another segment
            join_regions = 0;
            if (Args.do_join) {
                for (k = 0; k < segment_count + j; k++) {
                    int kMesh = Metadata[k].mesh_number;
                    int kSegment = Metadata[k].on_mesh_segment_number;
                    
                    if (verbose_level >= FC_DEBUG_MESSAGES)
                        fprintf(stderr, "Considering %d/%d and %d/%d\n", i, j, kMesh, kSegment);
                    
                    if (Args.use_bb) {
                        fc_getBoundingBoxesOverlap(num_dim, lowpoints[k], highpoints[k],
                                                   temp_lowpoint, temp_highpoint, &overlap_flag, NULL, NULL);
                        if (overlap_flag == 0) {
                            if (verbose_level >= FC_DEBUG_MESSAGES)
                                fprintf(stderr, "Should NOT check %d/%d and %d/%d\n", i, j, kMesh, kSegment);
                        }
                    }
                    // If not using bb's or the bb's overlap, check for neighborishness ...
                    if (! Args.use_bb || overlap_flag > 0) {
                        if (verbose_level >= FC_DEBUG_MESSAGES)
                            fprintf(stderr, "AM checking %d/%d and %d/%d\n", i, j, kMesh, kSegment);
                        
                        int num_pairs;
                        if (Args.use_displ_mesh)
                            rc = fc_getDisplacedSubsetsProximity(segments[i][j], displ[i][step_id],
                                                                segments[kMesh][kSegment], displ[kMesh][step_id],
                                                                Args.max_sep_dist, &num_pairs, NULL, NULL);
                        else
                            rc = fc_getSubsetsProximity(segments[i][j], segments[kMesh][kSegment],
                                                        Args.max_sep_dist, &num_pairs, NULL, NULL);
                        fc_exitIfErrorPrintf(rc, "Failed to get proximity for regions %d:%d and %d:%d\n",
                                                 i, j, kMesh, kSegment);
                        if (num_pairs > 0) {
                            // Assign to the neighbor the current region's label
                            if (verbose_level >= FC_DEBUG_MESSAGES)
                                fprintf(stderr, "  fc_gDSP() says to    JOIN (assign label %d to region num %d)\n",
                                                Metadata[j + segment_count].segment_label, kSegment);
                            int old_label = Metadata[k].segment_label;
                            Metadata[k].segment_label = Metadata[j + segment_count].segment_label;
                            // Update all other segments with neighbor's old label to the new one
                            for (n = 0; n < segment_count + j; n++) {
                                if (Metadata[n].segment_label == old_label) {
                                    if (verbose_level >= FC_DEBUG_MESSAGES)
                                        fprintf(stderr, "  update region %d from label %d to label %d)\n",
                                                Metadata[n].on_mesh_segment_number, Metadata[n].segment_label,
                                                Metadata[k].segment_label);
                                    Metadata[n].segment_label = Metadata[k].segment_label;
                                }
                            }
                        }
                    }

                }

            }

            if (Args.use_bb) {
                // Add to bb list
                for (k = 0; k < 3; k++) {
                    lowpoints[segment_count][k] = temp_lowpoint[k];
                    highpoints[segment_count][k] = temp_highpoint[k];
                }
            }
                
        }
        
        segment_count += num_segment;

    }
    
    // Store segments for this timestep
    
    FC_SortedIntArray seg_labels;
    fc_exitIfErrorPrintf(fc_initSortedIntArray(&seg_labels), "Failed to init SortedIntArray for labels");
    
    ts[step_id].num_regions = segment_count;
    ts[step_id].regions = (region *)malloc(segment_count * sizeof(region));
    for (i = 0; i < segment_count; i++) {
        ts[step_id].regions[i].number = Metadata[i].segment_id;
        ts[step_id].regions[i].mesh = Metadata[i].mesh_number;
        fc_addIntToSortedIntArray(&seg_labels, Metadata[i].segment_label);
        int subset_size, *subset_members;
        rc = fc_getSubsetMembersAsArray(segments[Metadata[i].mesh_number][Metadata[i].on_mesh_segment_number],
                                        &subset_size, &subset_members);
        fc_exitIfErrorPrintf(rc, "Failed to get region members for %d:%d\n",
                                 step_id, ts[step_id].num_regions - 1);
        ts[step_id].regions[i].size = subset_size;
        ts[step_id].regions[i].members = (int *)malloc(subset_size*sizeof(int));
        for (k = 0; k < subset_size; k++) {
            FC_AssociationType t;
            fc_getVariableAssociationType(vars[Metadata[i].mesh_number], &t);
            if (t == FC_AT_VERTEX)
                ts[step_id].regions[i].members[k] = n_lut[Metadata[i].mesh_number][subset_members[k]] + 1;
            else if (t == FC_AT_ELEMENT)
                ts[step_id].regions[i].members[k] = subset_members[k] + e_offset[Metadata[i].mesh_number] + 1;
        }
        free(subset_members);
    }
    
    // Figure out the number of labels and shift down to 0 and make consecutive
    ts[step_id].num_labels = seg_labels.numVal;
    for (i = 0; i < seg_labels.numVal; i++) {
        int label = seg_labels.vals[i];
        for (j = 0; j < segment_count; j++)
            if (Metadata[j].segment_label == label)
                ts[step_id].regions[j].label = i;
    }
    
    // cleanup
    fc_freeSortedIntArray(&seg_labels);
    if (Args.use_bb) {
        free(lowpoints);
        free(highpoints);
    }
    for (i = 0; i < num_vars; i++) {
        free(segments[i]);
    }
    free(segments);
    free(Metadata);

    return FC_SUCCESS;
}

static const char *opt_string = "bD:d:f:hlno:r:t:uV:v";

static const struct option long_opts[] = {
    {"use_bb", no_argument, NULL, 'b'},
    {"displ_variable", required_argument, NULL, 'D'},
    {"separation_distance", required_argument, NULL, 'd'},
    {"filename", required_argument, NULL, 'f'},
    {"help", no_argument, NULL, 'h'},
    {"doLocalSmooth", no_argument, NULL, 'l'},
    {"no_join", no_argument, NULL, 'n'},
    {"operator", required_argument, NULL, 'o'},
    {"radius", required_argument, NULL, 'r'},
    {"threshold", required_argument, NULL, 't'},
    {"undisplaced", no_argument, NULL, 'u'},
    {"variable", required_argument, NULL, 'V'},
    {"verbose", no_argument, NULL, 'v'},
    {NULL, no_argument, NULL, 0}
};

int _process_opts(int argc, char **argv) {
    
    int opt = 0;
    int long_index = 0;
    
    // Initialize
    Args.max_sep_dist = -1.0;
    Args.do_smoothing = 0;
    Args.do_local_smooth = 0;
    Args.use_displ_mesh = 1;
    Args.got_displ_var = 0;
    Args.displ_var = NULL;
    Args.use_bb = 0;
    Args.do_join = 1;
    Args.file_name = NULL;
    
    while( (opt = getopt_long( argc, argv, opt_string, long_opts, &long_index )) != -1 ) {
        switch(opt) {
            case 'b':
                Args.use_bb = 1;
                break;
            case 'D':
                Args.displ_var = strdup(optarg);
                Args.got_displ_var = 1;
                break;
            case 'd':
                Args.max_sep_dist = (double)atof(optarg);
                break;
            case 'f':
                Args.file_name = strdup(optarg);
                break;
            case 'h':
                _display_usage();
                break;
            case 'l':
                Args.do_local_smooth = 1;
                break;
            case 'n':
                Args.do_join = 0;
                break;
            case 'o':
                Args.operator = strdup(optarg);
                break;
            case 'r':
                Args.radius = atof(optarg);
                Args.do_smoothing = 1;
                break;
            case 't':
                Args.threshold = (double)atof(optarg);
                break;
            case 'u':
                Args.use_displ_mesh = 0;
                break;
            case 'V':
                Args.var_name = strdup(optarg);
                break;
            case 'v':
                verbose_level++;
                break;
            default:
                _display_usage();
                break;
        }
    }
    
    return 1;
}

void _display_usage( void ) {
    printf("usage: connect_comp_regions [options] -f dataset -V \"var name\" -o \"op\" -t thresh_value \n");
    printf("options: \n");
    printf("   --no_join|-n            : Do not join close regions\n");
    printf("   --separation|-d F       : Use F for the maximum separation distance (msd) instead\n");
    printf("                             of the default which is the maximum edge length in the\n");
    printf("                             dataset. Two regions with a minimum nodal separation less\n");
    printf("                             then or equal to the msd will be given the same region label\n");
    printf("   --use_bb|-b             : Compute the bounding box for each region and try to join\n");
    printf("                             only those regions whose bounding boxes overlap\n");
    printf("                             [ defaults to 'no' ]\n");
    printf("   --displ_variable|-D STR : Use the variable STR for calculating mesh displacment\n");
    printf("                             If the displacement variable is one of 'DISPL', 'displ_', 'DIS',\n");
    printf("                             the -D option does not need to be specified.\n");
    printf("   --undisplaced|-u        : Do smoothing, subsetting, proximity tests with un-displaced mesh\n");
    printf("   --radius|-r F           : smooth the variable by averaging each data\n");
    printf("                             point with all points within a radius F of it.\n");
    printf("   --doLocalSmooth         : if true, only points on local mesh are used for smoothing\n");
    printf("   --help|-h               : print this help message\n");
    printf("   --verbose|-v            : verbose: print warning and error messages\n");
    printf("                             May be specified multiple times to increase verbosity\n");
    printf("\n");
    printf("Prints to stdout information about the regions in the dataset\n");
    printf("for which the given variable satisfies the threshold operation.\n");
    printf("\"op\" is an operator such as \">\", \"<=\", \"==\", etc.\n");
    printf("Quotes are required around the op string, but can be dropped\n");
    printf("around the variable name if it contains no spaces or punctuation\n");
    printf("\n");
    printf("Example: connect_comp_regions -f data.ex2 -V Osaliency -o \">\" -t 0.5\n");
    printf("\n");
    exit(-1);
}
