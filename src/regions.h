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
#include <stdio.h>

typedef enum {
    NONE = 0,
    DISPLACED = 1,
    UNDISPLACED = 2
} Smoothing_Type;

struct region_struct {
    int number;
    int label;
    int mesh;
    int size;
    int *members;
}; typedef struct region_struct region;

struct timestep_struct {
    int timestep;
    int num_regions;
    int num_labels;
    region *regions;
}; typedef struct timestep_struct timestep;

struct region_file_metadata_struct {
    char *file_name;
    char *var_name;
    char *threshold_op;
    float threshold_val;
    int joining;
    float max_separation;
    Smoothing_Type smoothing;
    float smoothing_radius;
    int num_timesteps;
}; typedef struct region_file_metadata_struct region_file_metadata;

struct region_max_values_struct {
    int number;
    int label;
    int mesh;
    int size;
}; typedef struct region_max_values_struct region_max_values;

int read_region_data(FILE *file, timestep **regions, region_file_metadata *metadata);
void free_region_data(timestep *regions, region_file_metadata metadata);
void print_region_data(timestep *regions, region_file_metadata metadata, FILE *filename);
void clone_region_data(timestep *in_t, region_file_metadata in_m, timestep **out_t, region_file_metadata *out_m);
int compare_region_data(timestep *in1_t, region_file_metadata in1_m, timestep *in2_t, region_file_metadata in2_m);
void get_region_data_max(timestep *timesteps, region_file_metadata metadata, region_max_values *max_vals);
region_max_values init_max_vals(void);
int label_to_number_list(int label, int mesh, timestep ts, char** s_region_list, int** region_list);
int get_num_meshes(timestep ts);
int get_num_labels(timestep ts);

