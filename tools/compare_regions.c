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
#include <string.h>

#include "fc.h"
#include "array.h"
#include "regions.h"

int main(int argc, char** argv) {
    int i, j, k, m;
    int tt, ct;
    FC_ReturnCode rc;
    
    char *truth_file_name;
    char *comp_file_name;
    FILE *truthFile;
    FILE *compFile;
    
    FILE *intersect_file, *inTnotC_file;
    
    timestep *truth_timesteps;
    region_file_metadata truth_metadata;
    int num_truth_timesteps;
    timestep *comp_timesteps;
    region_file_metadata comp_metadata;
    int num_comp_timesteps;
    
    timestep *region_intersect_t, *region_inTnotC_t;
    region_file_metadata region_intersect_m, region_inTnotC_m;
    
    if (argc < 3) {
        printf("usage: %s truth_file comparison_file\n", argv[0]);
        exit(-1);
    }
    truth_file_name = argv[1];
    comp_file_name = argv[2];
    
    truthFile = fopen(truth_file_name, "r");
    if (truthFile == NULL) {
        fprintf(stderr, "Error: Unable to read truth file '%s'\n", truth_file_name);
        exit(8);
    }
    compFile = fopen(comp_file_name, "r");
    if (compFile == NULL) {
        fprintf(stderr, "Error: Unable to read comparison file '%s'\n", comp_file_name);
        exit(8);
    }
    
    intersect_file = fopen("region.intersect", "w");
    if (intersect_file == NULL) {
        fprintf(stderr, "Error: Unable to write intersection file\n");
        exit(8);
    }
    inTnotC_file = fopen("region.inTnotC", "w");
    if (inTnotC_file == NULL) {
        fprintf(stderr, "Error: Unable to write inTnotC file\n");
        exit(8);
    }
    
    // Read region data
    rc = read_region_data(truthFile, &truth_timesteps, &truth_metadata);
    fc_exitIfError(rc);
    rc = read_region_data(compFile, &comp_timesteps, &comp_metadata);
    fc_exitIfError(rc);
    fclose(truthFile);
    fclose(compFile);
    
    // Copy metadata
    clone_region_data(truth_timesteps, truth_metadata, &region_intersect_t, &region_intersect_m);
    clone_region_data(truth_timesteps, truth_metadata, &region_inTnotC_t, &region_inTnotC_m);

    num_truth_timesteps = truth_metadata.num_timesteps;
    num_comp_timesteps = comp_metadata.num_timesteps;
    
    // Generate overlap matrices for each timestep
    tt = 0;
    ct = 0;
    while (tt < num_truth_timesteps && ct < num_comp_timesteps) {

        if (truth_timesteps[tt].timestep == comp_timesteps[ct].timestep) {
            // Compare regions for this timestep
            int num_tr = truth_timesteps[tt].num_regions;
            region *tr = truth_timesteps[tt].regions;
            int num_cr = comp_timesteps[ct].num_regions;
            region *cr = comp_timesteps[ct].regions;
            for (i = 0; i < num_tr; i++) {

                // Initialize sizes
                region_intersect_t[tt].regions[i].size = 0;
                int *Comp_Union;
                int size_u = 0;
                Comp_Union = malloc(sizeof(int));
                Comp_Union[0] = -1;

                for (j = 0; j < num_cr; j++) {
                    // Only compare regions that are on the same mesh
                    if (tr[i].mesh == cr[j].mesh) {
                    
                        int *Intersection;
                        int size_i = 
                            array_intersection(tr[i].members, tr[i].size, cr[j].members, cr[j].size, &Intersection);
                        if (size_i > 0) {
                            k = region_intersect_t[tt].regions[i].size;
                            region_intersect_t[tt].regions[i].size += size_i;
                            region_intersect_t[tt].regions[i].members = 
                                realloc(region_intersect_t[tt].regions[i].members,
                                        region_intersect_t[tt].regions[i].size * sizeof(int));
                            for (m = 0; m < size_i; m++) {
                                region_intersect_t[tt].regions[i].members[m + k] = Intersection[m];
                            }
                        }
                        free(Intersection);

                        int *Union;
                        if (length_of_uid(Comp_Union) > 0) {
                            size_u =
                                array_union(Comp_Union, length_of_uid(Comp_Union), cr[j].members, cr[j].size, &Union);
                            Comp_Union = realloc(Comp_Union, (size_u + 1) * sizeof(int));
                            memcpy(Comp_Union, Union, (size_u + 1) * sizeof(int));
                            free(Union);
                        } else {
                            size_u = cr[j].size;
                            Comp_Union = realloc(Comp_Union, (size_u + 1) * sizeof(int));
                            memcpy(Comp_Union, cr[j].members, size_u * sizeof(int));
                            Comp_Union[size_u] = -1;
                        }
                        
                    }
                }
                
                // Get diffs between current truth region and union of all comp regions
                int *InTnotC;
                int size_d = array_diff(tr[i].members, tr[i].size, Comp_Union, size_u, &InTnotC);
                if (size_d > 0) {
                    region_inTnotC_t[tt].regions[i].size = size_d;
                    region_inTnotC_t[tt].regions[i].members = 
                        realloc(region_inTnotC_t[tt].regions[i].members, size_d * sizeof(int));
                    memcpy(region_inTnotC_t[tt].regions[i].members, InTnotC, size_d * sizeof(int));
                } else {
                    region_inTnotC_t[tt].regions[i].size = 0;
                }
                free(InTnotC);
                free(Comp_Union);
            }
            
            tt++;
            ct++;
        } else if (comp_timesteps[ct].timestep < truth_timesteps[tt].timestep) {
            // Handle timestep for the comparison data that is not in the truth data
            ct++;
        } else if (truth_timesteps[tt].timestep < comp_timesteps[ct].timestep) {
            // Handle timestep for the truth data that is not in the comparison data
            tt++;
        }
    }
    // Handle extra truth timesteps
    for (i = tt; i < num_truth_timesteps; i++) {
    }
    // Handle extra comparison timesteps
    for (i = ct; i < num_comp_timesteps; i++) {
    }
    
    print_region_data(region_intersect_t, region_intersect_m, intersect_file);
    print_region_data(region_inTnotC_t,   region_inTnotC_m,   inTnotC_file);
    
    // clean up
    fclose(intersect_file);
    fclose(inTnotC_file);

    free_region_data(truth_timesteps, truth_metadata);
    free_region_data(comp_timesteps,  comp_metadata);

    free_region_data(region_intersect_t, region_intersect_m);
    free_region_data(region_inTnotC_t,   region_inTnotC_m);
    
    exit(0);
}

