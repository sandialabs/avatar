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
#include <string.h>
#include <math.h>

#include "array.h"
//#include "base.h"
#include "regions.h"


void free_region_data(timestep *timesteps, region_file_metadata metadata) {
    int i, j;
    for (i = 0; i < metadata.num_timesteps; i++) {
        for (j = 0; j < timesteps[i].num_regions; j++) {
            free(timesteps[i].regions[j].members);
        }
        free(timesteps[i].regions);
    }
    free(timesteps);
    free(metadata.file_name);
    free(metadata.var_name);
    free(metadata.threshold_op);

}


void print_region_data(timestep *reg, region_file_metadata meta, FILE *filename) {
    
    int i, j, k;
    
    FILE *stream;
    
    if (filename != NULL) {
        stream = filename;
    } else {
        stream = stdout;
    }
    
    fprintf(stream, "dataset %s\nvariable %s\nthreshold %s %f\n", 
                    meta.file_name, meta.var_name, meta.threshold_op, meta.threshold_val);
    fprintf(stream, "joining %s\nmax_separation %f\n", meta.joining?"on":"off", meta.max_separation);
    fprintf(stream, "smoothing %d\n", meta.smoothing);
    if (meta.smoothing)
        fprintf(stream, "smoothing_radius %f\n", meta.smoothing_radius);
    fprintf(stream, "number_of_timesteps %d\n",  meta.num_timesteps);
    for (i = 0; i < meta.num_timesteps; i++) {
        timestep ts = reg[i];
        fprintf(stream, "timestep %d\nnumber_of_regions %d\nnumber_of_unique_labels %d\n",
                        ts.timestep, ts.num_regions, ts.num_labels);
        for (j = 0; j < reg[i].num_regions; j++) {
            region r = reg[i].regions[j];
            fprintf(stream, "region_number %d\nregion_label %d\nmesh_number %d\nregion_size %d\nregion_members",
                            r.number, r.label, r.mesh, r.size);
            for (k = 0; k < r.size; k++) {
                fprintf(stream, " %d", r.members[k]);
            }
            fprintf(stream, "\n");
        }
    }
    
}


int read_region_data(FILE *file, timestep **timesteps, region_file_metadata *metadata) {
    
    char strbuf[1024];
    int i;
    int temp_timesteps, temp_regions;
    int this_timestep, this_region;
    
    // Initialize some optional values
    (*metadata).smoothing_radius = -1.0;

    while (fscanf(file, "%s", strbuf) > 0) {
        if (! strcmp(strbuf, "dataset") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).file_name = malloc( (strlen(strbuf) + 1) * sizeof(char) );
            strcpy((*metadata).file_name, strbuf);
        } else if (! strcmp(strbuf, "variable") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).var_name = malloc( (strlen(strbuf) + 1) * sizeof(char) );
            strcpy((*metadata).var_name, strbuf);
        } else if (! strcmp(strbuf, "threshold") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).threshold_op = malloc( (strlen(strbuf) + 1) * sizeof(char) );
            strcpy((*metadata).threshold_op, strbuf);
            if (fscanf(file, "%s", strbuf) > 0) {
                (*metadata).threshold_val = atof(strbuf);
            }
        } else if (! strcmp(strbuf, "joining") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).joining = strcmp(strbuf, "off") ? 1 : 0;
        } else if (! strcmp(strbuf, "max_separation") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).max_separation = atof(strbuf);
        } else if (! strcmp(strbuf, "smoothing") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).smoothing = atoi(strbuf);
        } else if (! strcmp(strbuf, "smoothing_radius") && fscanf(file, "%s", strbuf) > 0) {
            (*metadata).smoothing_radius = atof(strbuf);
        } else if (! strcmp(strbuf, "number_of_timesteps") && fscanf(file, "%s", strbuf) > 0) {
            temp_timesteps = atoi(strbuf);
            (*metadata).num_timesteps = temp_timesteps;
            *timesteps = (timestep *)malloc(temp_timesteps * sizeof(timestep));
        } else if (! strcmp(strbuf, "timestep") && fscanf(file, "%s", strbuf) > 0) {
            this_timestep = atoi(strbuf);
            (*timesteps)[this_timestep].timestep = this_timestep;
        } else if (! strcmp(strbuf, "number_of_regions") && fscanf(file, "%s", strbuf) > 0) {
            temp_regions = atoi(strbuf);
            (*timesteps)[this_timestep].num_regions = temp_regions;
            (*timesteps)[this_timestep].regions = (region*)malloc(temp_regions * sizeof(region));
        } else if (! strcmp(strbuf, "number_of_unique_labels") && fscanf(file, "%s", strbuf) > 0) {
            (*timesteps)[this_timestep].num_labels = atoi(strbuf);
        } else if (! strcmp(strbuf, "region_number") && fscanf(file, "%s", strbuf) > 0) {
            this_region = atoi(strbuf);
            (*timesteps)[this_timestep].regions[this_region].number = this_region;
        } else if (! strcmp(strbuf, "region_label") && fscanf(file, "%s", strbuf) > 0) {
            (*timesteps)[this_timestep].regions[this_region].label = atoi(strbuf);
        } else if (! strcmp(strbuf, "mesh_number") && fscanf(file, "%s", strbuf) > 0) {
            (*timesteps)[this_timestep].regions[this_region].mesh = atoi(strbuf);
        } else if (! strcmp(strbuf, "region_size") && fscanf(file, "%s", strbuf) > 0) {
            int size = atoi(strbuf);
            (*timesteps)[this_timestep].regions[this_region].size = size;
            (*timesteps)[this_timestep].regions[this_region].members = (int *)malloc(size * sizeof(int));
        } else if (! strcmp(strbuf, "region_members")) {
            for (i = 0; i < (*timesteps)[this_timestep].regions[this_region].size; i++) {
                if (fscanf(file, "%s", strbuf) > 0) {
                    (*timesteps)[this_timestep].regions[this_region].members[i] = atoi(strbuf);
                } else {
                    return(-4);
                }
            }
        } else {
            fprintf(stderr, "Got '%s'\n", strbuf);
            return(-3);
        }
    }

    return(0);
}


void clone_region_data(timestep *input_t, region_file_metadata input_m,
                                timestep **output_t, region_file_metadata *output_m) { 

    int i, j, k;
    
    // Copy metadata
    output_m->file_name = malloc( (strlen(input_m.file_name) + 1) * sizeof(char) );
    strcpy(output_m->file_name, input_m.file_name);
    output_m->var_name = malloc( (strlen(input_m.var_name) + 1) * sizeof(char) );
    strcpy(output_m->var_name, input_m.var_name);
    output_m->threshold_op = malloc( (strlen(input_m.threshold_op) + 1) * sizeof(char) );
    strcpy(output_m->threshold_op, input_m.threshold_op);    
    output_m->threshold_val = input_m.threshold_val;
    output_m->joining = input_m.joining;
    output_m->max_separation = input_m.max_separation;
    output_m->smoothing = input_m.smoothing;
    output_m->smoothing_radius = input_m.smoothing_radius;
    output_m->num_timesteps = input_m.num_timesteps;
    
    // Copy timestep data
    *output_t = (timestep *)malloc(input_m.num_timesteps * sizeof(timestep));
    for (i = 0; i < input_m.num_timesteps; i++) {
        (*output_t)[i].timestep = input_t[i].timestep;
        (*output_t)[i].num_regions = input_t[i].num_regions;
        (*output_t)[i].num_labels = input_t[i].num_labels;
        (*output_t)[i].regions = (region*)malloc(input_t[i].num_regions * sizeof(region));
        for (j = 0; j < input_t[i].num_regions; j++) {
            (*output_t)[i].regions[j].number = input_t[i].regions[j].number;
            (*output_t)[i].regions[j].label  = input_t[i].regions[j].label;
            (*output_t)[i].regions[j].mesh   = input_t[i].regions[j].mesh;
            (*output_t)[i].regions[j].size   = input_t[i].regions[j].size;
            (*output_t)[i].regions[j].members = (int *)malloc(input_t[i].regions[j].size * sizeof(int));
            for (k = 0; k < input_t[i].regions[j].size; k++) {
                (*output_t)[i].regions[j].members[k] = input_t[i].regions[j].members[k];
            }
        }
    }

}

int compare_region_data(timestep *input1_t, region_file_metadata input1_m,
                        timestep *input2_t, region_file_metadata input2_m) { 

    int c, i, j ,k;
    
    // Compare metadata
    c = strcmp(input1_m.file_name, input2_m.file_name);
    if (c != 0) return c;
    c =strcmp(input1_m.var_name, input2_m.var_name);
    if (c != 0) return c;
    c =strcmp(input1_m.threshold_op, input2_m.threshold_op);    
    if (c != 0) return c;
    if (input1_m.threshold_val != input2_m.threshold_val)
        return (input1_m.threshold_val > input2_m.threshold_val ? 1 : -1);
    if (input1_m.joining != input2_m.joining) {
        fprintf(stderr, "joining %d %d\n", input1_m.joining, input2_m.joining);
        return (input1_m.joining > input2_m.joining ? 1 : -1);
    }
    if (input1_m.max_separation != input2_m.max_separation)
        return (input1_m.max_separation > input2_m.max_separation ? 1 : -1);
    if (input1_m.smoothing != input2_m.smoothing)
        return (input1_m.smoothing > input2_m.smoothing ? 1 : -1);
    if (input1_m.smoothing_radius != input2_m.smoothing_radius)
        return (input1_m.smoothing_radius > input2_m.smoothing_radius ? 1 : -1);
    if (input1_m.num_timesteps != input2_m.num_timesteps)
        return (input1_m.num_timesteps > input2_m.num_timesteps ? 1 : -1);
    
    // Compare timestep data
    for (i = 0; i < input2_m.num_timesteps; i++) {
        if (input1_t[i].timestep != input2_t[i].timestep)
            return (input1_t[i].timestep > input2_t[i].timestep ? 1 : -1);
        if (input1_t[i].num_regions != input2_t[i].num_regions)
            return (input1_t[i].num_regions > input2_t[i].num_regions ? 1 : -1);
        if (input1_t[i].num_labels != input2_t[i].num_labels)
            return (input1_t[i].num_labels > input2_t[i].num_labels ? 1 : -1);
        for (j = 0; j < input2_t[i].num_regions; j++) {
            if (input1_t[i].regions[j].number != input2_t[i].regions[j].number)
                return (input1_t[i].regions[j].number > input2_t[i].regions[j].number ? 1 : -1);
            if (input1_t[i].regions[j].label != input2_t[i].regions[j].label)
                return (input1_t[i].regions[j].label > input2_t[i].regions[j].label ? 1 : -1);
            if (input1_t[i].regions[j].mesh != input2_t[i].regions[j].mesh)
                return (input1_t[i].regions[j].mesh > input2_t[i].regions[j].mesh ? 1 : -1);
            if (input1_t[i].regions[j].size != input2_t[i].regions[j].size)
                return (input1_t[i].regions[j].size > input2_t[i].regions[j].size ? 1 : -1);
            for (k = 0; k < input2_t[i].regions[j].size; k++) {
                if (input1_t[i].regions[j].members[k] != input2_t[i].regions[j].members[k])
                    return (input1_t[i].regions[j].members[k] > input2_t[i].regions[j].members[k] ? 1 : -1);
            }
        }
    }

    return 0;

}

void get_region_data_max(timestep *timesteps, region_file_metadata metadata, region_max_values *max_vals) {

    int i, j;
    
    int num_timesteps = metadata.num_timesteps;
    for (i = 0; i < num_timesteps; i++) {
        max_vals->number = (timesteps[i].num_regions > max_vals->number) ?
                           timesteps[i].num_regions : max_vals->number;
        max_vals->label = (timesteps[i].num_labels > max_vals->label) ?
                          timesteps[i].num_labels : max_vals->label;
        for (j = 0; j < timesteps[i].num_regions; j++) {
            max_vals->size = (timesteps[i].regions[j].size > max_vals->size) ? 
                             timesteps[i].regions[j].size : max_vals->size;
            max_vals->mesh = (timesteps[i].regions[j].mesh > max_vals->mesh) ? 
                             timesteps[i].regions[j].mesh : max_vals->mesh;
        }
    }
}

int get_num_meshes(timestep ts) {
    int i;
    int max_mesh = -1;
    for (i = 0; i < ts.num_regions; i++) {
        if (ts.regions[i].mesh > max_mesh)
            max_mesh = ts.regions[i].mesh;
    }
    return(max_mesh + 1);
}

int get_num_labels(timestep ts) {
    int i;
    int max_label = -1;
    for (i = 0; i < ts.num_regions; i++) {
        if (ts.regions[i].label > max_label)
            max_label = ts.regions[i].label;
    }
    return(max_label + 1);
}

region_max_values init_max_vals(void) {
    region_max_values max_vals;
    
    max_vals.number = -1;
    max_vals.label  = -1;
    max_vals.mesh   = -1;
    max_vals.size   = -1;
    
    return max_vals;
}

int label_to_number_list(int label, int mesh, timestep ts, char** s_region_list, int** region_list) {
    int i;
    int sizeof_str = 0;
    
    int current_alloc = 512;
    int delta_alloc = 512;
    *region_list = (int *)malloc(current_alloc * sizeof(int));

    int num_regions = 0;
    
    for (i = 0; i < ts.num_regions; i++) {
        // Only worry about regions labeled as 'label'
        if (ts.regions[i].label != label)
            continue;
        // If 'mesh' is > -1, only worry about regions on mesh number 'mesh'
        if (mesh > -1 && ts.regions[i].mesh != mesh)
            continue;
        
        // Add this region label to the list
        num_regions++;
        if (num_regions > current_alloc)
            current_alloc += delta_alloc;
        *region_list = (int *)realloc( *region_list, current_alloc * sizeof(int) );
        if ( *region_list == NULL )
            return(-1);
        
        (*region_list)[(num_regions) - 1] = ts.regions[i].number;
        if (ts.regions[i].number > 0)
            sizeof_str += (int)log10(ts.regions[i].number) + 1;
        else
            sizeof_str += 1;
        //printf("Adding %d to list and increasing string length to %d\n",ts.regions[i].number,sizeof_str); 
    }
    
    // Sort region numbers
    int_array_sort(num_regions, (*region_list)-1);
    
    // Generate and return a character representation
    if (num_regions > 0) {
        sizeof_str *= 2;
        *s_region_list = (char *)malloc(sizeof_str * sizeof(char));
        sprintf(*s_region_list, "%d", (*region_list)[0]);
        for (i = 1; i < num_regions; i++)
            sprintf(*s_region_list, "%s,%d", *s_region_list, (*region_list)[i]);
    }

    return(num_regions);
}
